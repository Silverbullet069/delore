import * as logger from '../shared/logger';
import * as vscode from 'vscode';
import { executePythonCommand } from '../shared/shell';

/**
 * This function is independent from model type (detect, locate, repair)
 *   and front-end code (VSCode Extension API)
 * @param pathToPythonBinary The absolute path to your Python binary
 * @param pathToModule The absolute path to your Python Module
 * @param param The parameter for Python Module, in this context it's usually a C/C++ function
 */
// export const executeModel = async (
//   pathToPythonBinary: string,
//   pathToModule: string,
//   param: string
// ): Promise<boolean | Error> => {
//   try {
//     const data = await executePythonCommand<boolean>(
//       pathToPythonBinary,
//       pathToModule,
//       param
//     );

//     // debug
//     logger.debugSuccess(__filename, data);
//     logger.debugSuccess(__filename, typeof data);

//     return data;
//   } catch (err) {
//     logger.debugError(__filename, err);

//     // Cre: https://stackoverflow.com/questions/37980559/is-it-better-to-return-undefined-or-null-from-a-javascript-function
//     return null;
//   }
// };

export const locate = (extensionPath: string, code: string) => {
  const editor = vscode.window.activeTextEditor;

  // Check editor is opened
  if (!editor) {
    logger.notifyError(`There is no active editor!`);
    return;
  }

  // Highlighting
  const highlightRed1 = vscode.window.createTextEditorDecorationType({
    backgroundColor: 'rgba(255, 0, 0, 0.7)' // Red background with 30% opacity
  });
  const highlightRed2 = vscode.window.createTextEditorDecorationType({
    backgroundColor: 'rgba(255, 0, 0, 0.2)' // Red background with 30% opacity
  });
  const highlightRed2Range: vscode.DecorationOptions[] = [
    { range: new vscode.Range(0, 0, 0, 500) } /* line 1 */,
    { range: new vscode.Range(2, 0, 2, 500) } /* line 3 */
  ];
  const highlightRed1Range: vscode.DecorationOptions[] = [
    { range: new vscode.Range(1, 0, 1, 500) } /* line 2 */
  ];

  editor.setDecorations(highlightRed1, highlightRed1Range);
  editor.setDecorations(highlightRed2, highlightRed2Range);

  // Split editor right
  vscode.commands.executeCommand('workbench.action.splitEditorRight');

  // Auto scroll
  let lineNum = editor.selection.active.line;
  let currentPositionInLine = editor.selection.active.character;

  console.log('a', editor.document.lineAt(editor.selection.active.line).text);

  const scrollBottomMargin = 4;
  const selectionTopMargin = 1;

  // Define the range you want to scroll to
  const range = editor.document.lineAt(lineNum + scrollBottomMargin).range;

  editor.selection = new vscode.Selection(
    new vscode.Position(lineNum - selectionTopMargin, 0),
    new vscode.Position(lineNum - selectionTopMargin, 0)
  );

  // Scroll to the defined range
  editor.revealRange(range);
};
