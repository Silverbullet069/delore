import * as vscode from 'vscode';
import { EXTENSION_NAME } from '../constants/config';

enum ColorCode {
  RED = 31,
  GREEN = 32,
  YELLOW = 33,
  BLUE = 34,
  PURPLE = 35,
  CYAN = 36
}

/* ============================================ */
/* DEBUG CONSOLE                                */
/* ============================================ */

export const debugInfo = (...msgs: any[]): void => {
  console.log(`[INFO]`, ...msgs); // default color: bright blue
};

export const debugWarn = (...msgs: any[]): void => {
  console.log(`\u001b[${ColorCode.YELLOW}m [WARN]`, ...msgs, 'm\u001b[0m:');
};

export const debugError = (...msgs: any[]): void => {
  console.log(`\u001b[${ColorCode.RED}m [ERRR]`, ...msgs, `\u001b[0m`);
};

export const debugSuccess = (...msgs: any[]): void => {
  console.log(`\u001b[${ColorCode.GREEN}m [SUCC]`, ...msgs, `\u001b[0m`);
};

/* ============================================ */
/* NOTIFICATION                                 */
/* ============================================ */

export const notifyInfo = (msg: any): void => {
  vscode.window.showInformationMessage(`${EXTENSION_NAME}: ${msg}`);
};

export const notifyWarn = (msg: any): void => {
  vscode.window.showWarningMessage(`${EXTENSION_NAME}: ${msg}`);
};

export const notifyError = (msg: any): void => {
  vscode.window.showErrorMessage(`${EXTENSION_NAME}: ${msg}`);
};
