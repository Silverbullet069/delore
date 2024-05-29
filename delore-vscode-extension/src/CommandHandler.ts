import * as vscode from 'vscode';

import { OutlineTreeDataProvider } from './views/customTreeDataProvider';
import { activateDeloreCommandHandler } from './commands/activateDelore.command';
import { revealLineCommandHandler } from './commands/revealLine.command';

export class CommandHandler {
  /* ==================================================== */
  /* Design Pattern: Singleton                            */
  /* ==================================================== */

  private static _instance: CommandHandler;

  private constructor() {}

  public static get instance(): CommandHandler {
    if (!CommandHandler._instance) {
      CommandHandler._instance = new CommandHandler();
    }
    return CommandHandler._instance;
  }

  /* ==================================================== */
  /* List of registered commands                          */
  /* ==================================================== */

  public activateDeloreCommandWrapper(
    extensionPath: string,
    outlineTreeDataProvider: OutlineTreeDataProvider
  ): vscode.Disposable {
    return activateDeloreCommandHandler(extensionPath, outlineTreeDataProvider);
  }

  public revealLineCommandWrapper(): vscode.Disposable {
    return revealLineCommandHandler();
  }
}
