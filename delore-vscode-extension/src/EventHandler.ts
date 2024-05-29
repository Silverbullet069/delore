import * as vscode from 'vscode';

import { onDidRenameFilesEventHandler } from './events/onDidRenameFilesEventHandler.event';
import { syncRevealEventHandler } from './events/syncRevealEventHandler.event';

export class EventHandler {
  /* ==================================================== */
  /* Design Pattern: Singleton                            */
  /* ==================================================== */

  private static _instance: EventHandler;

  private constructor() {}

  public static get instance(): EventHandler {
    if (!EventHandler._instance) {
      EventHandler._instance = new EventHandler();
    }
    return EventHandler._instance;
  }

  /* ==================================================== */
  /* List of registered event handlers                    */
  /* ==================================================== */

  public onDidRenameFilesWrapper(): vscode.Disposable {
    return onDidRenameFilesEventHandler();
  }

  public syncRevealWrapper(): vscode.Disposable {
    return syncRevealEventHandler();
  }
}
