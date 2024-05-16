import * as vscode from 'vscode';
import * as logger from '../utils/logger';
import { sanitize } from '../utils/sanitize';

import { basename } from 'path';
import {
  EXTENSION_ID,
  MultiRoleModel,
  ModelType,
  ResourceManager
} from '../utils/data';
import { executePythonCommand } from '../utils/shell';

export const parseFunction1Command = (extensionPath: string) => {
  return async (): Promise<void> => {
    try {
      const currentEditor = vscode.window.activeTextEditor;

      // Check editor is opened
      if (!currentEditor) {
        throw new Error('There is no active text editor!');
      }

      // Extract highlighted part
      const selection = currentEditor.selection;
      if (!selection) {
        throw new Error('There is no selected code!');
      }

      const highlightedCode = currentEditor.document.getText(selection);

      // Sanitize input
      const sanitizedInput = sanitize(highlightedCode);

      // TODO: Using Document Symbol Provider is better, I discover this via Outline View in Explorer.
      // ! DONE: check testParse2.command.ts

      // Check if selected code is a function?
      const pathToHelperScriptCwd =
        ResourceManager.getInstance().getAbsToHelperDir(extensionPath);
      const pathToBinary =
        ResourceManager.getInstance().getPathToPythonBinary(extensionPath);
      const pathToIsFunctionScript =
        ResourceManager.getInstance().getAbsPathToIsFunctionScript(
          extensionPath
        );
      const param = sanitizedInput;

      const isFunction = await executePythonCommand<boolean>(
        pathToBinary,
        pathToIsFunctionScript,
        param,
        pathToHelperScriptCwd
      );
      if (!isFunction) {
        logger.notifyInfo('Selected code is not a function!');
        return;
      }
      logger.notifyInfo('Selection code is a function!');
    } catch (err) {
      logger.debugError(basename(module.filename), err);
      logger.notifyError(err);
      return;
    }
  };
};
