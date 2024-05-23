import * as vscode from 'vscode';
import * as logger from '../utils/logger';

import {
  getVisibleTextEditor,
  isFileOpenAndVisible
} from '../views/apiWrapper';
import { isLeft, unwrapEither } from '../utils/either';

export const revealLineCommandHandler = (): vscode.Disposable => {
  return vscode.commands.registerCommand(
    'delore.revealLine',
    (editor: vscode.TextEditor, range: vscode.Range, tempFsPath?: string) => {
      if (!editor) {
        logger.debugError(`There is no source code editor!`);
        return;
      }

      editor.revealRange(range, vscode.TextEditorRevealType.AtTop);
      logger.debugSuccess('Navigate source code!');

      // all of this are optional
      // also scroll if specified
      if (tempFsPath && isFileOpenAndVisible(tempFsPath)) {
        const tempEditorEither = getVisibleTextEditor(tempFsPath);
        if (isLeft(tempEditorEither)) {
          const err = unwrapEither(tempEditorEither);
          logger.debugError(err.type, '\n', err.msg);
          return;
        }

        const tempEditor = unwrapEither(tempEditorEither);
        tempEditor.revealRange(range, vscode.TextEditorRevealType.AtTop);
        logger.debugSuccess('Navigate temp file!');
        return;
      }
    }
  );
};
