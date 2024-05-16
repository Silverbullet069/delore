import path from 'path';
import * as vscode from 'vscode';

import * as logger from '../utils/logger';
import { randomUUID } from 'crypto';
import {
  FuncState,
  LocalizationModelLinesOutput,
  LocalizationModelOutput,
  RepairationModelLinesOutput,
  TempState
} from '../model/state.model';
import {
  Either,
  isLeft,
  makeLeft,
  makeRight,
  unwrapEither
} from '../utils/either';
import { InMemoryRepository } from '../repositories/inMemory.repository';
import { isFileOpen, isFileOpenAndVisible } from '../views/apiWrapper';
import { TextEncoder } from 'util';
import { areStringsSimilar } from '../utils/misc';
import { processedNoSpace } from '../utils/sanitize';

const LINE_VUL_BKG_COLOR = 'rgba(255, 0, 0, 0.5)';
const LINE_REPAIR_BKG_COLOR = 'rgba(0, 255, 0, 0.5)';

type DisplayResultErrorType =
  | 'VSCODE_API_ERROR'
  | 'REPOSITORY_GET_FUNCS_ERROR'
  | 'REPOSITORY_GET_TEMP_ERROR'
  | 'REPOSITORY_UPDATE_TEMP_ERROR'
  | 'FUNC_IS_EMPTY'
  | 'TEMP_IS_EMPTY'
  | 'MAKE_FILE_VISIBLE_ERROR'
  | 'VUL_AND_REPAIR_ARRAY_NOT_EQUAL_LENGTH'
  | 'VUL_AND_REPAIR_ARRAY_NOT_IDENTICAL_NUM';

type DisplayResultError = {
  type: DisplayResultErrorType;
  msg: string;
};

const calculateNewLines = (
  pos1: vscode.Position,
  pos2: vscode.Position
): string => {
  let newLines = '';
  for (let lineNum = pos1.line; lineNum < pos2.line; ++lineNum) {
    newLines += '\n';
  }
  return newLines;
};

// display for both detection, localization and repairation
export const displayResultService = async (
  extensionPath: string,
  srcCodeEditor: vscode.TextEditor
): Promise<Either<DisplayResultError, 'SUCCESS'>> => {
  const srcCodeEditorFsPath = srcCodeEditor.document.uri.fsPath;

  const tempStateEither =
    InMemoryRepository.getInstance().getTempInOneEditor(srcCodeEditorFsPath);

  if (isLeft(tempStateEither)) {
    const err = unwrapEither(tempStateEither);
    return makeLeft({
      type: 'REPOSITORY_GET_TEMP_ERROR',
      msg: `${err.type}\n${err.msg}` // no need stack trace
    });
  }

  // if undefined, this func run Delore the first time
  const tempState = unwrapEither(tempStateEither);

  try {
    // if temp state existed, extract its Uri
    // otherwise, create new Uri
    const tempUri =
      tempState !== null
        ? vscode.Uri.file(tempState.fsPath) // use existing file
        : vscode.Uri.file(
            path.join(extensionPath, `/temp/${randomUUID()}.txt`)
          ); // create new file

    // dispose decorations (if existed)
    tempState?.vulDecoration.dispose();
    tempState?.repairDecoration.dispose();

    // create new decorations
    const vulDecoration = vscode.window.createTextEditorDecorationType({
      backgroundColor: LINE_VUL_BKG_COLOR
    });

    const repairDecoration = vscode.window.createTextEditorDecorationType({
      backgroundColor: LINE_REPAIR_BKG_COLOR
    });

    // Create empty txt file or overwriting existing txt file
    await vscode.workspace.fs.writeFile(tempUri, new TextEncoder().encode(' ')); // avoid VSCode's untitiled placeholder

    // Open the text document if closed
    const document = await vscode.workspace.openTextDocument(tempUri);

    // Show text document on editor
    const editor = await vscode.window.showTextDocument(document, {
      viewColumn:
        srcCodeEditor.viewColumn === vscode.ViewColumn.Two
          ? vscode.ViewColumn.One
          : srcCodeEditor.viewColumn === vscode.ViewColumn.One
            ? vscode.ViewColumn.Two
            : srcCodeEditor.viewColumn === undefined
              ? vscode.ViewColumn.One
              : vscode.ViewColumn.Beside, // if neither, open on side
      preview: false, // keep document open even if you open another document,
      preserveFocus: false // switch focus to new editor
    });

    // if it's read-only (from last run), reset it to write-able to override data
    await vscode.commands.executeCommand(
      'workbench.action.files.resetActiveEditorReadonlyInSession'
    );

    // Clear the content of document
    await editor.edit((editBuilder) => {
      const fullRange = new vscode.Range(
        document.lineAt(0).range.start,
        document.lineAt(document.lineCount - 1).range.end
      );
      editBuilder.replace(fullRange, ' '); // avoid VSCode's untitled placeholder
    });

    // Extract funcs content
    const funcsEither =
      InMemoryRepository.getInstance().getFuncsInOneEditor(srcCodeEditorFsPath);
    if (isLeft(funcsEither)) {
      const err = unwrapEither(funcsEither);
      return makeLeft({
        type: 'REPOSITORY_GET_FUNCS_ERROR',
        msg: `${err.type}\n${err.msg}` // no need stack trace
      });
    }

    const funcs = unwrapEither(funcsEither);

    // using for of can return makeLeft (not very 'functional' thou)
    // write funcs content into new file

    // extract last line that get written to calculate the number of new lines prepending before inserting
    let lastFuncLinePosition = new vscode.Position(0, 0);
    for (const func of funcs) {
      if (func.lines.length === 0 || !func.unprocessedContent) {
        return makeLeft({
          type: 'FUNC_IS_EMPTY',
          msg: `Check again your function: ${func.name}\n${new Error().stack}`
        });
      }

      // you can't use update only the funcs that has their content changed since the whole files get override
      // or maybe i can? if I used the replace()?
      // nah, vscode can't replace different block of code all at once

      const firstLine = func.lines[0];
      const insertPosition = new vscode.Position(
        firstLine.numOnEditor,
        firstLine.startCharOnEditor
      );

      logger.debugInfo(insertPosition);

      const prependNewLines = calculateNewLines(
        lastFuncLinePosition,
        insertPosition
      );

      // write into file
      await editor.edit((editBuilder) => {
        editBuilder.insert(
          insertPosition,
          prependNewLines + func.unprocessedContent
        );
      });

      const lastLine = func.lines[func.lines.length - 1];
      lastFuncLinePosition = new vscode.Position(
        lastLine.numOnEditor,
        lastLine.startCharOnEditor
      );
    }
    await document.save(); // insert many, save once

    /* ================================================== */

    // localization UI logic
    // highlight and add marker to vulnerable lines
    const vulOnEditorLineNumsAndContents = funcs
      .reduce(
        (prev: any[], currentFunc: FuncState) => [
          ...prev,
          ...(currentFunc.mergeLocalizationResult?.lines.reduce(
            (prevLines: any[], currentLine: LocalizationModelLinesOutput) => [
              // PREPARE: currentLine is any, consider giving the lines inside model output, a type
              ...prevLines,
              currentLine.isVulnerable
                ? ({
                    numOnEditor: currentFunc.lines[currentLine.num].numOnEditor,
                    content: currentLine.content
                  } satisfies {
                    numOnEditor: number;
                    content: string; // consider change to number of models identify this line as vul
                  })
                : null // only highlight the line with vulnerables
            ],
            []
          ) ?? [])
        ],
        []
      )
      .filter((line) => line !== null);

    const vulLineRanges = vulOnEditorLineNumsAndContents.map(
      (numAndContent) => editor.document.lineAt(numAndContent.numOnEditor).range
    );

    editor.setDecorations(vulDecoration, vulLineRanges);

    /* ================================================== */

    // repairation UI logic
    // highlight and add marker to repair lines
    const repairOnEditorLineNumsAndFixes = funcs.reduce(
      (prev: any[], currentFunc: FuncState) => [
        ...prev,
        ...(currentFunc.mergeRepairationResult?.lines.reduce(
          (prevLines: any[], currentLine: RepairationModelLinesOutput) => [
            ...prevLines,
            {
              numOnEditor: currentFunc.lines[currentLine.num].numOnEditor,
              fix: currentLine.fix
            } satisfies {
              numOnEditor: number;
              fix: string;
            }
          ],
          []
        ) ?? [])
      ],
      []
    );

    logger.debugSuccess(
      'repairOnEditorLineNumAndFixes',
      '\n',
      repairOnEditorLineNumsAndFixes
    );

    if (
      repairOnEditorLineNumsAndFixes.length !==
      vulOnEditorLineNumsAndContents.length
    ) {
      return makeLeft({
        type: 'VUL_AND_REPAIR_ARRAY_NOT_EQUAL_LENGTH',
        msg: `Check again your list of vul lines: ${vulOnEditorLineNumsAndContents} and list of repair lines: ${repairOnEditorLineNumsAndFixes}`
      });
    }

    const length = repairOnEditorLineNumsAndFixes.length;
    for (let i = 0; i < length; ++i) {
      if (
        vulOnEditorLineNumsAndContents[i].numOnEditor !==
        repairOnEditorLineNumsAndFixes[i].numOnEditor
      ) {
        return makeLeft({
          type: 'VUL_AND_REPAIR_ARRAY_NOT_IDENTICAL_NUM',
          msg: `Check again your list of vul lines: ${vulOnEditorLineNumsAndContents} and list of repair lines: ${repairOnEditorLineNumsAndFixes}`
        });
      }
    }

    // main UI logic for repairation
    const repairLineRanges: vscode.Range[] = [];

    // using for..of for outer async/await
    for (const lineNumAndFix of repairOnEditorLineNumsAndFixes) {
      const oldEndLinePosition = editor.document.lineAt(
        lineNumAndFix.numOnEditor
      ).range.end;

      await editor.edit((editBuilder) => {
        editBuilder.insert(oldEndLinePosition, lineNumAndFix.fix);
      });

      logger.debugSuccess(
        'repair content',
        lineNumAndFix.numOnEditor,
        lineNumAndFix.fix
      );

      const newEndLinePostion = editor.document.lineAt(
        lineNumAndFix.numOnEditor
      ).range.end;

      repairLineRanges.push(
        new vscode.Range(oldEndLinePosition, newEndLinePostion)
      );
    }

    editor.setDecorations(repairDecoration, repairLineRanges);

    /* ================================================== */

    // mark editor read-only
    await vscode.commands.executeCommand(
      'workbench.action.files.setActiveEditorReadonlyInSession'
    );

    /* ================================================== */

    // return focus to source code editor
    // NOTE: you must use vscode.Uri.file, don't use document
    await vscode.window.showTextDocument(vscode.Uri.file(srcCodeEditorFsPath), {
      viewColumn: srcCodeEditor.viewColumn,
      preview: false,
      preserveFocus: false
    });

    /* ================================================== */

    // update temp uri associates with editor fs path
    const newTempState: TempState = {
      fsPath: tempUri.fsPath,
      vulDecoration: vulDecoration,
      repairDecoration: repairDecoration
    };

    const updateTempEither =
      InMemoryRepository.getInstance().updateTempInOneEditor(
        srcCodeEditorFsPath,
        newTempState
      );

    if (isLeft(updateTempEither)) {
      const err = unwrapEither(updateTempEither);
      return makeLeft({
        type: 'REPOSITORY_UPDATE_TEMP_ERROR',
        msg: `${err.type}\n${err.msg}`
      });
    }

    /* ================================================== */

    return makeRight('SUCCESS');
  } catch (err) {
    return makeLeft({
      type: 'VSCODE_API_ERROR',
      msg: `${err}`
    });
  }
};
