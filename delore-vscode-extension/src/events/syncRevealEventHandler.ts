import * as vscode from 'vscode';
import * as logger from '../utils/logger';
import { InMemoryRepository } from '../repositories/inMemory.repository';
import { isLeft, isRight, unwrapEither } from '../utils/either';
import { basename, extname } from 'path';
import { SUPPORTED_LANGUAGES } from '../constants/config';
import {
  getVisibleTextEditor,
  isFileOpenAndVisible
} from '../views/apiWrapper';

export const syncRevealEventHandler = (context: vscode.ExtensionContext) => {
  return vscode.window.onDidChangeTextEditorVisibleRanges((event) => {
    const editor = event.textEditor;
    const editorFsPath = editor.document.uri.fsPath;
    const editorExt = extname(editorFsPath);

    logger.debugSuccess(`Ext: ${editorExt}`);
    logger.debugSuccess(`Visible Ranges: `, ...event.visibleRanges);

    if (!SUPPORTED_LANGUAGES.includes(editorExt)) {
      logger.debugInfo(`Editor ${basename(editorFsPath)} not supported!`);
      return;
    }

    const tempEither =
      InMemoryRepository.getInstance().getTempInOneEditor(editorFsPath);

    // either editor not support, or temp state not existed (not run DeLoRe)
    if (isLeft(tempEither)) {
      const err = unwrapEither(tempEither);
      logger.debugInfo(err.type, '\n', err.msg);
      return;
    }

    // TODO: check unwrapEither behavior, it ignored null value
    const tempState = unwrapEither(tempEither);
    if (tempState === null) {
      logger.debugInfo(
        `Editor: ${basename(editorFsPath)} have not run Delore before! Can not sync reveal!`
      );
      return;
    }

    const tempFsPath = tempState.fsPath;
    const tempEditor = vscode.window.visibleTextEditors.find(
      (editor) => editor.document.uri.fsPath === tempFsPath
    );

    if (!tempEditor) {
      // temp editor not visible
      logger.debugInfo(
        `Editor: ${basename(editorFsPath)} temp file ${basename(tempFsPath)} is not visible! Make it visible first!`
      );
      return;
    }

    const funcsEither =
      InMemoryRepository.getInstance().getFuncsInOneEditor(editorFsPath);

    if (isLeft(funcsEither)) {
      const err = unwrapEither(funcsEither);
      logger.debugError(err.type, '\n', err.msg);
      return; // nuke
    }

    const funcs = unwrapEither(funcsEither);
    const startVisibleLine = event.visibleRanges[0].start.line;

    for (const func of funcs) {
      const startLine = func.lines[0].numOnEditor;
      const endLine = func.lines[func.lines.length - 1].numOnEditor;

      // skip all funcs that passed through endLine
      if (startVisibleLine >= endLine) {
        logger.debugSuccess('A Start line: ', startLine);
        logger.debugSuccess('A Start visible line: ', startVisibleLine);
        logger.debugSuccess('A End line: ', endLine);
        continue;
      }

      if (
        startVisibleLine >= startLine &&
        startVisibleLine <= endLine
        // func.name !== context.globalState.get('last-func-navigation') // avoid scroll to func 2 but still at func 1
      ) {
        logger.debugSuccess('B Start line: ', startLine);
        logger.debugSuccess('B Start visible line: ', startVisibleLine);
        logger.debugSuccess('B End line: ', endLine);

        const processedStartVisiblePosition = new vscode.Position(
          // based on observation
          Math.min(
            event.visibleRanges[0].start.line + 5,
            editor.document.lineCount - 1
          ),
          event.visibleRanges[0].start.character
        );

        const processedVisibleRange = new vscode.Range(
          processedStartVisiblePosition,
          event.visibleRanges[0].end
        );

        tempEditor.revealRange(
          processedVisibleRange,
          vscode.TextEditorRevealType.Default
        );
        // context.globalState.update('last-func-navigation', func.name);
        break;
      }
    }
    return;
  });
};
