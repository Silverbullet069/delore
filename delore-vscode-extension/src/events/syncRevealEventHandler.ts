import * as vscode from 'vscode';
import * as logger from '../utils/logger';
import { InMemoryRepository } from '../repositories/inMemory.repository';
import { isLeft, unwrapEither } from '../utils/either';
import { basename, extname } from 'path';
import { SUPPORTED_LANGUAGES } from '../constants/config';

export const syncRevealEventHandler = () => {
  return vscode.window.onDidChangeTextEditorVisibleRanges((event) => {
    const editor = event.textEditor;
    const editorFsPath = editor.document.uri.fsPath;

    if (!SUPPORTED_LANGUAGES.includes(extname(editorFsPath))) {
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
    const tempState = unwrapEither(tempEither);
    const tempFsPath = tempState.fsPath;

    const editors = vscode.window.visibleTextEditors;
    const tempEditor = editors.find(
      (editor) => editor.document.uri.fsPath === tempFsPath
    );

    // temp editor not visible
    if (!tempEditor) {
      logger.debugInfo(
        `Editor: ${basename(editorFsPath)} have not run Delore before! Can not sync reveal!`
      );
      return;
    }

    if (editors.includes(editor) && editors.includes(tempEditor)) {
    }
    return;
  });
};
