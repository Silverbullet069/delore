import * as vscode from 'vscode';
import { basename } from 'path';

import { EXTENSION_ID } from '../utils/data';
import * as logger from '../utils/logger';

// Business Logic + Func
import { ParseFuncService } from '../services/parseFunc.service';
import { BaseRepository } from '../../legacy/repositories/base/repo.base';
import { Func } from '../../legacy/model/func.model';

/**
 * Using VSCode Extension's API to write into custom outlines
 * @param repository
 * @param extensionPath
 * @returns
 */
export const parseFunction2Command = (
  repository: BaseRepository<Func>,
  extensionPath: string
) => {
  return async (): Promise<void> => {
    try {
      const currentEditor = vscode.window.activeTextEditor;

      // Check editor is opened
      if (!currentEditor) {
        throw new Error('There is no active text editor!');
      }

      // Extract function from DocumentSymbol
      const uri = currentEditor.document.uri;

      const symbols: vscode.DocumentSymbol[] =
        await vscode.commands.executeCommand(
          'vscode.executeDocumentSymbolProvider',
          uri
        );

      // Begin processing
      const processNodesService = new ParseFuncService(repository);
      await processNodesService.parseFunc(symbols, currentEditor.document);
    } catch (err) {
      logger.debugError(basename(module.filename), err);
      logger.notifyError(err);
      return;
    }
  };
};
