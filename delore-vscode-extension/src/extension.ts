import * as vscode from 'vscode';
import * as crypto from 'crypto';
import * as logger from './utils/logger';

import dotenv from 'dotenv';

import { FuncState, LineState } from './model/state.model';
import { CCppConfigDocumentSymbolProvider } from './views/customDocumentSymbolProvider';
import { processedNoSpace, processedOneSpace } from './utils/sanitize';
import { InMemoryRepository } from './repositories/inMemory.repository';
import {
  ModelTreeDataProvider,
  OutlineTreeDataProvider
} from './views/customTreeDataProvider';

import path, { basename } from 'path';
import {
  EXTENSION_ID,
  SUPPORTED_LANGUAGES,
  resourceManager
} from './constants/config';
import { activateDeloreCommand } from './commands/activateDelore.command';
import { isLeft, unwrapEither } from './utils/either';
import {
  getVisibleTextEditor,
  isFileOpen,
  isFileOpenAndVisible
} from './views/apiWrapper';
import { instructionsV2 } from './constants/systemMessage';
import {
  LLM_COMMAND_ID,
  promptGithubCopilotService
} from './services/prompt.service';
import { revealLineCommandHandler } from './commands/revealLine.command';
import { onDidRenameFilesEventHandler } from './events/onDidRenameFilesEventHandler.event';
import { syncRevealEventHandler } from './events/syncRevealEventHandler';

const symbolKinds = [
  'file',
  'module',
  'namespace',
  'package',
  'class',
  'method',
  'property',
  'field',
  'constructor',
  'enum',
  'interface',
  'function',
  'variable',
  'constant',
  'string',
  'number',
  'boolean',
  'array',
  'object',
  'key',
  'null',
  'enumMember',
  'struct',
  'event',
  'operator',
  'type parameter'
];

export function activate(context: vscode.ExtensionContext) {
  const absPathToLocalEnvEither = resourceManager.getAbsPathToLocalEnv(
    context.extensionPath
  );

  if (isLeft(absPathToLocalEnvEither)) {
    const err = unwrapEither(absPathToLocalEnvEither);
    logger.debugError(err.type, '\n', err.msg);
    return;
  }

  const absPathToLocalEnv = unwrapEither(absPathToLocalEnvEither);
  dotenv.config({
    path: absPathToLocalEnv,
    encoding: 'utf8'
  });

  // Debug
  // console.log(context.extensionPath);
  // console.log(context.extensionUri.fsPath); // same as above

  // NOTE: Instead of registering tree data provider, you can also create a tree view and specify tree data provider using this function: vscode.window.createTreeView(), this is more dynamic.

  /* ==================================================== */
  /* Initialize 3 Model View + 1 Custom Outline View      */
  /* ==================================================== */

  const outlineTreeDataProvider = new OutlineTreeDataProvider(
    context.extensionPath
  );
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(
      'customOutlineView',
      outlineTreeDataProvider
    )
  );

  // NOTE: remember to change the view id in `package.json`
  const detectionModelTreeDataProvider = new ModelTreeDataProvider(
    'detection',
    context.extensionPath
  );
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(
      'detectionModelView',
      detectionModelTreeDataProvider
    )
  );

  const localizationModelTreeDataProvider = new ModelTreeDataProvider(
    'localization',
    context.extensionPath
  );
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(
      'localizationModelView',
      localizationModelTreeDataProvider
    )
  );

  const repairationModelTreeDataProvider = new ModelTreeDataProvider(
    'repairation',
    context.extensionPath
  );
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(
      'repairationModelView',
      repairationModelTreeDataProvider
    )
  );

  /* ==================================================== */
  /* HANDLE EVENT                                         */
  /* ==================================================== */

  // Preserve funcState and tempState when a file is relocated or renamed
  context.subscriptions.push(onDidRenameFilesEventHandler());

  // Auto scroll when clicking on custom outlines
  context.subscriptions.push(revealLineCommandHandler());

  // Auto scroll temp editor when source code editor scroll and vice versa
  context.subscriptions.push(syncRevealEventHandler(context));

  const handleChangeEditor = async (editor?: vscode.TextEditor) => {
    if (!editor) {
      logger.debugInfo(`Delore not supported empty editor!`);
      return;
    }

    if (
      !SUPPORTED_LANGUAGES.includes(path.extname(editor.document.uri.fsPath))
    ) {
      logger.debugInfo(`Delore not supported this file extension!`);
      return;
    }

    let symbols: vscode.DocumentSymbol[] = [];
    try {
      symbols = await vscode.commands.executeCommand(
        'vscode.executeDocumentSymbolProvider',
        editor.document.uri
      );
    } catch (error) {
      logger.debugError(
        `Retrieve symbol list for editor: ${basename(editor.document.uri.fsPath)} failed! Retry...`
      );
      await handleChangeEditor(editor); // retry until you got it
      return;
    }

    // if empty file
    if (symbols.length === 0) {
      return;
    }

    if (!symbols) {
      await handleChangeEditor(editor); // retry until you got it
      return;
    }

    const funcsOrNulls = symbols?.map((symbol) => {
      const symbolKind = symbolKinds[symbol.kind];

      // for now, only 'function' is supported
      if (symbolKind !== 'function') return null;

      // in some cases, if you write new function before an existing function, the function will reset
      // check if first line is function name
      // if the function loses its name, it will vanished into thin air
      const name = symbol.name;
      const match = name.match(/[a-zA-Z_][a-zA-Z0-9_]*/);
      const nameNoArguments = match ? match[0] : null;

      // logger.debugSuccess(nameNoArguments);

      if (!nameNoArguments) {
        logger.debugError('Check regex matching string or function name!');
        return;
      }

      const unprocessedRange = symbol.range;
      let startPosition = unprocessedRange.start;

      // weird behavior: if the line before the startPosition is not a blank line (no newline), the startPosition is one line after the real position

      // if (
      //   twoLineBeforeStartPosition >= 0 &&
      //   editor.document.lineAt(twoLineBeforeStartPosition).text.length !== 0
      // ) {
      //   const lineBeforeStartPosition = twoLineBeforeStartPosition + 1;
      //   startPosition = new vscode.Position(
      //     lineBeforeStartPosition,
      //     startPosition.character
      //   );
      // }

      // console.log(startPosition);

      let isNameFound: boolean = false;
      for (
        let line = unprocessedRange.start.line;
        line <= unprocessedRange.end.line;
        ++line
      ) {
        const textLine = editor.document.lineAt(line);

        // check if line is out-of-bound
        if (textLine.isEmptyOrWhitespace) {
          vscode.window.showErrorMessage(
            `Line number ${line + 1} is out of bounds!`
          );

          logger.debugError(
            `Line number ${line} is out of bounds in editor:  ${basename(editor.document.uri.fsPath)}.`
          );
          return null;
        }

        const textLineContent = textLine.text;

        if (textLineContent.includes(nameNoArguments)) {
          isNameFound = true;
          startPosition = textLine.range.start;
          break;
        }
      }

      if (!isNameFound) {
        vscode.window.showErrorMessage(
          `Function's name not found! Check the syntax of the function between line ${unprocessedRange.start.line + 1} and ${unprocessedRange.end.line + 1}!`
        );

        logger.debugError(
          `Function's name not found! Check the syntax of the function between line ${unprocessedRange.start.line + 1} and ${unprocessedRange.end.line + 1} in editor: ${basename(editor.document.uri.fsPath)}`
        );
        return null;
      }

      const processedRange = new vscode.Range(
        startPosition,
        unprocessedRange.end
      );

      // logger.debugSuccess(processedRange);

      const unprocessedContentFunc = editor.document.getText(processedRange);

      const processedContent = processedOneSpace(unprocessedContentFunc);
      const processedContentHash = crypto
        .createHash('md5')
        .update(processedContent)
        .digest('hex');

      const lines: LineState[] = [];
      const startLineNum = symbol.range.start.line;
      const endLineNum = symbol.range.end.line;

      // not very functional, meh, who cares
      for (let i = startLineNum; i <= endLineNum; ++i) {
        const textLine = editor.document.lineAt(i);

        const numOnEditor = textLine.lineNumber;
        const unprocessedContent = textLine.text;
        const processedContent = textLine.text.trim();
        const startCharOnEditor = textLine.range.start.character;
        const endCharOnEditor = textLine.range.end.character;
        lines.push({
          numOnEditor,
          unprocessedContent,
          processedContent,
          startCharOnEditor,
          endCharOnEditor
        });
      }

      const func: FuncState = {
        name,
        unprocessedContentFunc,
        processedContentFunc: processedContent,
        processedContentFuncHash: processedContentHash,
        lines,
        isRunDelore: false, // by default, every new Func hasn't run through Delore yet
        detectionResults: [], // reset
        localizationResults: [], // reset
        repairationResults: [] // reset
      };

      return func;
    });

    const funcs = funcsOrNulls.filter(
      (funcOrNull) => funcOrNull !== null
    ) as FuncState[];

    // there is no function inside file
    if (!funcs) {
      const updateEither =
        InMemoryRepository.getInstance().updateAllFuncsInOneEditor(
          editor.document.uri.fsPath,
          []
        );

      if (isLeft(updateEither)) {
        const err = unwrapEither(updateEither);
        logger.debugError(err.type, '\n', err.msg);
        return;
      }

      logger.debugSuccess(
        `Function: ${handleChangeEditor.name}
        Msg: Add empty funcs for file successfully!`
      );

      // Update tree view
      outlineTreeDataProvider.refresh();
      return;
    }

    // there is at least 1 function inside file
    const updateEither =
      InMemoryRepository.getInstance().updateAllFuncsInOneEditor(
        editor.document.uri.fsPath,
        funcs
      );

    if (isLeft(updateEither)) {
      const err = unwrapEither(updateEither);
      logger.debugError(err.type, '\n', err.msg);
      return;
    }

    logger.debugSuccess(
      `File: ${basename(module.filename)}
      Function: ${handleChangeEditor.name}
      Msg: Add/Update funcs successfully!`
    );

    // Update tree view
    outlineTreeDataProvider.refresh();

    // Auto bring temp file to front
    const tempStateEither = InMemoryRepository.getInstance().getTempInOneEditor(
      editor.document.uri.fsPath
    );

    if (isLeft(tempStateEither)) {
      const err = unwrapEither(tempStateEither);
      logger.debugError(err.type, '\n', err.msg);
      return;
    }

    const tempState = unwrapEither(tempStateEither);
    if (!tempState) {
      logger.debugInfo(
        `File havent run Localization Service so it didn't have temp file!`
      );
      return;
    }

    // if (!isFileOpen(tempState.fsPath)) {
    //   logger.debugSuccess('test');
    //   await openFileBeside(
    //     tempState.fsPath,
    //     editor.document.uri.fsPath,
    //     editor.viewColumn
    //   );
    //   return;
    // }

    // if (
    //   isFileOpen(tempState.fsPath) &&
    //   !isFileOpenAndVisible(tempState.fsPath)
    // ) {
    //   await resetTempVisibility(
    //     tempState.fsPath,
    //     editor.document.uri.fsPath,
    //     editor.viewColumn
    //   );
    //   return;
    // }

    return;
  };

  // Auto update custom outlines when changing editor
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(handleChangeEditor)
  );

  // Auto update tree view when content in text editor change
  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument(
      async (changeEvent: vscode.TextDocumentChangeEvent) => {
        // TODO: listen only to changes in text document in active text editor
        if (changeEvent.document === vscode.window.activeTextEditor?.document) {
          await handleChangeEditor(vscode.window.activeTextEditor);

          // ! Change method, don't update directly onto the current editor
          // await handleUpdateDecoration(vscode.window.activeTextEditor)
          // await handleFuncContentChanged(vscode.window.activeTextEditor, [
          //   ...changeEvent.contentChanges
          // ]);
        }

        // PREPARE: listen for text document outside text editor?
      }
    )
  );

  // TODO: populate editor content into custom outline View
  // ! Not what I need
  // context.subscriptions.push(
  //   vscode.languages.registerDocumentSymbolProvider(
  //     {
  //       language: 'c',
  //       scheme: 'file'
  //     },
  //     new CCppConfigDocumentSymbolProvider()
  //   )
  // );
  // context.subscriptions.push(
  //   vscode.languages.registerDocumentSymbolProvider(
  //     {
  //       language: 'cpp',
  //       scheme: 'file'
  //     },
  //     new CCppConfigDocumentSymbolProvider()
  //   )
  // );

  /* ==================================================== */
  /* RUN WHEN VSCODE START UP                             */
  /* ==================================================== */

  const MAX_RETRIES = 5;
  const DELAY = 2000;
  let attempts = 0;

  const intervalId = setInterval(async () => {
    const editor = vscode.window.activeTextEditor;
    if (editor || attempts >= MAX_RETRIES) {
      clearInterval(intervalId);
      if (editor) {
        try {
          await handleChangeEditor(editor);
        } catch (err) {
          logger.debugError(err);
          logger.debugError('Failed to initialize custom outline tree view!');
        }
      } else {
        logger.debugError('No active editor found after maximum attempts!');
      }
    } else {
      attempts++;
    }
  }, DELAY);

  /* ==================================================== */
  /* TESTING                                              */
  /* ==================================================== */

  // defines what the command will do
  // context.subscriptions.push(
  //   // Register the command handler for the /meow followup
  //   vscode.commands.registerTextEditorCommand(
  //     LLM_COMMAND_ID,
  //     async (textEditor: vscode.TextEditor) => {
  //       await vscode.window.withProgress(
  //         {
  //           location: vscode.ProgressLocation.Notification,
  //           title: 'Running Delore x GitHub Copilot',
  //           cancellable: true
  //         },
  //         async (progress, token) => {
  //           token.onCancellationRequested(() => {
  //             logger.notifyInfo(
  //               'User cancelled Delore x GitHub Copilot service!'
  //             );
  //           });

  //           progress.report({ message: 'Send chat request...' });

  //           // // input
  //           const text = textEditor.document.getText();

  //           // // list of msgs
  //           const messages = [
  //             new vscode.LanguageModelChatSystemMessage(instructionsV2),
  //             new vscode.LanguageModelChatUserMessage(text)
  //           ];

  //           // res
  //           let chatResponse: vscode.LanguageModelChatResponse | undefined;
  //           try {
  //             chatResponse = await vscode.lm.sendChatRequest(
  //               LANGUAGE_MODEL_ID,
  //               messages,
  //               {},
  //               new vscode.CancellationTokenSource().token
  //             );
  //           } catch (err) {
  //             // making the chat request might fail because
  //             // - user consent not given
  //             // - model does not exist
  //             // - quote limits exceeded
  //             // - other issues

  //             // other error
  //             logger.debugError(err);

  //             if (err instanceof vscode.LanguageModelError) {
  //               logger.debugError(err.name, '\n', err.message, '\n', err.code);
  //             }
  //             return;
  //           }

  //           // Clear the editor content before inserting new content
  //           await textEditor.edit((edit) => {
  //             const start = new vscode.Position(0, 0);
  //             const end = new vscode.Position(
  //               textEditor.document.lineCount - 1,
  //               textEditor.document.lineAt(
  //                 textEditor.document.lineCount - 1
  //               ).text.length
  //             );
  //             edit.delete(new vscode.Range(start, end));
  //           });

  //           progress.report({ message: 'Write chat response...' });

  //           // Stream the code into the editor as it is coming in from the Language Model
  //           try {
  //             for await (const fragment of chatResponse.stream) {
  //               await textEditor.edit((edit) => {
  //                 const lastLine = textEditor.document.lineAt(
  //                   textEditor.document.lineCount - 1
  //                 );
  //                 const position = new vscode.Position(
  //                   lastLine.lineNumber,
  //                   lastLine.text.length
  //                 );
  //                 edit.insert(position, fragment);
  //               });
  //             }
  //           } catch (err) {
  //             // async response stream may fail, e.g network interruption or server side error
  //             await textEditor.edit((edit) => {
  //               const lastLine = textEditor.document.lineAt(
  //                 textEditor.document.lineCount - 1
  //               );
  //               const position = new vscode.Position(
  //                 lastLine.lineNumber,
  //                 lastLine.text.length
  //               );
  //               edit.insert(position, (<Error>err).message);
  //             });
  //           }

  //           progress.report({ message: 'Finished.' });
  //         }
  //       );
  //     }
  //   )
  // );

  context.subscriptions.push(
    vscode.commands.registerTextEditorCommand(
      `${EXTENSION_ID}.activateDelore`,
      activateDeloreCommand(context.extensionPath, outlineTreeDataProvider)
    )
  );
}

// This method is called when your extension is deactivated
export function deactivate() {}
