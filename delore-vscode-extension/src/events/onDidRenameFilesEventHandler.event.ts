import path from 'path';
import * as vscode from 'vscode';
import * as logger from '../utils/logger';

import { SUPPORTED_LANGUAGES } from '../constants/config';
import { InMemoryRepository } from '../repositories/inMemory.repository';
import { isLeft, unwrapEither } from '../utils/either';

export const onDidRenameFilesEventHandler = (): vscode.Disposable => {
  return vscode.workspace.onDidRenameFiles((event) => {
    event.files.forEach((file) => {
      const oldExt = path.extname(file.oldUri.fsPath);
      const newExt = path.extname(file.newUri.fsPath);

      if (
        SUPPORTED_LANGUAGES.includes(oldExt) &&
        SUPPORTED_LANGUAGES.includes(newExt)
      ) {
        // main logic
        const updateEditorFsPathEither =
          InMemoryRepository.getInstance().updateEditorFsPath(
            file.oldUri.fsPath,
            file.newUri.fsPath
          );

        if (isLeft(updateEditorFsPathEither)) {
          const err = unwrapEither(updateEditorFsPathEither);
          logger.debugError(err.type, '\n', err.msg);
          return;
        }

        return;
      }

      // PREPARE: what if a not .c/.cpp (supported language in general) file change to .c / .cpp file
      if (SUPPORTED_LANGUAGES.includes(newExt)) {
        return;
      }
    });
  });
};
