/* eslint-disable @typescript-eslint/space-before-function-paren */
/* eslint-disable @typescript-eslint/semi */
import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";

import { execSync } from "child_process";

function printInfo(msg: string): void {
  console.log(msg);
  vscode.window.showInformationMessage(msg);
}

function printError(msg: string) {
  console.error(msg);
  vscode.window.showErrorMessage(msg);
}

type Coordinate = {
  line: number;
  col: number;
};

type Repairation = {
  coor: Coordinate;
  newCode: string;
  reliability: number;
};

function isFileExisted(filePath: string): boolean {
  let flag = true;
  try {
    fs.accessSync(filePath, fs.constants.F_OK); // file is existed
  } catch (err) {
    flag = false;
  }

  return flag;
}

function isCppFile(filePath: string): boolean {
  if (!isFileExisted(filePath)) {
    return false;
  }

  const fileNameWithExtension = path.basename(filePath);
  return (
    fileNameWithExtension.endsWith(".cpp") ||
    fileNameWithExtension.endsWith(".c")
  );
}

function detect(content: string): boolean {
  let codeSnippet = content;

  if (isCppFile(content)) {
    codeSnippet = fs.readFileSync(content, { encoding: "utf-8" });
  }

  // debug
  console.log(codeSnippet);

  const detectModelPath = path.join(__dirname, "..", "..", "detection");
  process.chdir(detectModelPath);

  // printInfo(process.cwd());

  const pythonPath = path.resolve(
    detectModelPath,
    "py-detection",
    "bin",
    "python"
  );

  // const pipPath = path.resolve(detectModelPath, "py-detection" "bin", "pip");

  const detectModule = "detect";

  console.log("so far so good");

  // TODO: Sanitize input
  const output = execSync(`${pythonPath} -m ${detectModule} "${codeSnippet}"`)
    .toString()
    .trim();

  const isVulnerable = !(output[output.length - 1].length === 0);

  // debug
  printInfo(isVulnerable ? "true" : "false");

  return isVulnerable;
}

function locate(code: string): Coordinate[] {
  const test = code;

  // TODO: change Python script to run Localization model
  const output = execSync("python3 -c 'print(\"Run Python Code from VSCode\")'")
    .toString()
    .trim();

  // for now, let's keep it this way
  const arr: Coordinate[] = [];
  for (let i = 0; i < 10; ++i) {
    arr.push({ line: -1, col: -1 });
  }
  return arr;
}

function repair(locateRes: Coordinate[]): Repairation[] {
  // TODO: change Python script to run Repairation model
  const output = execSync("python3 -c 'print(\"Run Python Code from VSCode\")'")
    .toString()
    .trim();

  // for now, let's keep it this way
  const len = locateRes.length;
  const arr: Repairation[] = [];
  for (let i = 0; i < len; ++i) {
    arr.push({
      coor: locateRes[i],
      newCode: "New reliable code!",
      reliability: 0.0,
    });
  }
  return arr;
}

export async function activate(context: vscode.ExtensionContext) {
  let disposable = vscode.commands.registerCommand(
    "delore-vscode-extension.test-run-python-code",
    () => {
      const editor = vscode.window.activeTextEditor;

      if (editor) {
        /* ---------------------------------- */
        /*              Detection             */
        /* ---------------------------------- */

        // Method #1: Highlighting

        const selection = editor.selection;
        const highlightedText = editor.document.getText(selection);

        if (!highlightedText) {
          vscode.window.showErrorMessage("Please highlight your code!");
          return;
        }

        // Detection model used
        let isVulnerable = detect(highlightedText);

        if (!isVulnerable) {
          vscode.window.showInformationMessage("Your code MIGHT be clean!");
          return;
        } else {
          vscode.window.showInformationMessage("Vulnerability found!");
        }

        // Method #2: put cursor inside function
        // const positionCursor = editor.selection.active;
        // console.log(positionCursor.line, positionCursor.character);
        // TODO: find a C/C++ function parser to identify the beginning + end of a function

        // Method #3: Automatically run on change
        // TODO: find a C/C++ function parser

        /* ---------------------------------- */
        /*               Locate               */
        /* ---------------------------------- */
      }
    }
  );

  context.subscriptions.push(disposable);

  const highlightedDecorationType =
    vscode.window.createTextEditorDecorationType({
      backgroundColor: "rgba(255, 0, 0, 0.7)", // Yellow background with 30% opacity
    });

  // Register a command to trigger highlighting
  disposable = vscode.commands.registerCommand(
    "delore-vscode-extension.highlight-code",
    () => {
      const activeEditor = vscode.window.activeTextEditor;

      if (activeEditor) {
        const { document } = activeEditor;

        // Define ranges to highlight
        const rangesToHighlight: vscode.DecorationOptions[] = [
          { range: new vscode.Range(0, 0, 0, 5) }, // Highlight first 5 characters of the document
          { range: new vscode.Range(2, 0, 2, 10) }, // Highlight specific range in line 3
          // Add more ranges as needed
        ];

        // Apply the decoration
        activeEditor.setDecorations(
          highlightedDecorationType,
          rangesToHighlight
        );
      }
    }
  );

  context.subscriptions.push(disposable);

  // Test split editor right
  disposable = vscode.commands.registerCommand(
    "delore-vscode-extension.split-editor-right",
    () => {
      // Get the active editor
      const activeEditor = vscode.window.activeTextEditor;

      if (activeEditor) {
        // Execute the command to split editor right
        vscode.commands.executeCommand("workbench.action.splitEditorRight");
      } else {
        vscode.window.showErrorMessage("No active editor found.");
      }
    }
  );

  context.subscriptions.push(disposable);

  // Test auto scroll
  disposable = vscode.commands.registerCommand(
    "delore-vscode-extension.scroll-to-position",
    () => {
      // Get the active text editor
      const activeEditor = vscode.window.activeTextEditor;
      if (activeEditor) {
        const lineNum = 10; // change this value

        const scrollBottomMargin = 4;
        const selectionTopMargin = 1;

        // Define the range you want to scroll to
        const range = activeEditor.document.lineAt(
          lineNum + scrollBottomMargin
        ).range;

        activeEditor.selection = new vscode.Selection(
          new vscode.Position(lineNum - selectionTopMargin, 0),
          new vscode.Position(lineNum - selectionTopMargin, 0)
        );

        // Scroll to the defined range
        activeEditor.revealRange(range);
      } else {
        vscode.window.showErrorMessage("No active editor found.");
      }
    }
  );

  context.subscriptions.push(disposable);

  // Test get active window current scroll position
  disposable = vscode.commands.registerCommand(
    "delore-vscode-extension.get-scroll-position",
    () => {
      // Get the active text editor
      const activeEditor = vscode.window.activeTextEditor;
      if (activeEditor) {
        // Get the visible ranges (scroll position)
        const visibleRanges = activeEditor.visibleRanges;

        if (visibleRanges.length > 0) {
          const firstVisibleRange = visibleRanges[0];
          vscode.window.showInformationMessage(
            `Scroll position: Line ${
              firstVisibleRange.start.line + 1
            }, Column ${firstVisibleRange.start.character + 1}`
          );
        } else {
          vscode.window.showWarningMessage("No visible range found.");
        }
      } else {
        vscode.window.showErrorMessage("No active editor found.");
      }
    }
  );

  context.subscriptions.push(disposable);

  // Test auto split editor + copy content + scroll to position
  disposable = vscode.commands.registerCommand(
    "delore-vscode-extension.locate-vul",
    () => {
      // Get the active text editor
      const activeEditor = vscode.window.activeTextEditor;

      if (activeEditor) {
        const activeEditorFullText = activeEditor.document.getText();
        vscode.window.showInformationMessage(
          `Full text: ${activeEditorFullText}`
        );

        const selection = activeEditor.selection;
        if (selection) {
          const selectedStartPos = selection.start;
          const selectedEndPos = selection.end;

          const highlightedText = activeEditor.document.getText(selection);
          vscode.window.showInformationMessage(
            `Selected text: ${highlightedText}`
          );
          vscode.window.showInformationMessage(
            `Start position: Line ${selectedStartPos.line + 1}, Column ${
              selectedStartPos.character + 1
            } \n
            End position: Line ${selectedEndPos.line + 1}, Column ${
              selectedEndPos.character + 1
            }`
          );

          vscode.commands
            .executeCommand("workbench.action.splitEditorRight")
            .then(() => {
              const newEditor = vscode.window.activeTextEditor;
              if (newEditor) {
                // Start a new edit transaction in the new editor
                newEditor
                  .edit((editBuilder: vscode.TextEditorEdit) => {
                    // Insert text at the specified position
                    editBuilder.insert(selectedStartPos, highlightedText);
                  })
                  .then(
                    () => {
                      vscode.window.showInformationMessage(
                        "Text pasted at specified position in the new editor."
                      );
                    },
                    (error) => {
                      vscode.window.showErrorMessage(
                        "Failed to paste text in the new editor: " + error
                      );
                    }
                  );
              } else {
                vscode.window.showErrorMessage(
                  "Failed to get the newly created editor."
                );
              }
            });
        }
      } else {
        vscode.window.showErrorMessage("No active editor found.");
      }
    }
  );

  context.subscriptions.push(disposable);
}

// export async function activate(context: vscode.ExtensionContext) {
//   // Add a status bar item
//   let disposable = vscode.window.setStatusBarMessage("Code is clean. For now.");
//   context.subscriptions.push(disposable);

//   disposable = vscode.commands.registerCommand(
//     "delore-vscode-extension.test-extract-function",
//     () => {

// ... Detection has been taken

//         const { res: locateRes } = locate(highlightedText);

//         if (!isLocatable) {
//           vscode.window.showInformationMessage(
//             "This model can't locate vulnerabilities. Proceed to manual locate."
//           );
//           return;
//         }

//         vscode.window.showInformationMessage(
//           `${locateRes.line}, ${locateRes.col}`
//         );

//         // /* ---------------------------------- */
//         // /*               Repair               */
//         // /* ---------------------------------- */

//         const { res: repairRes, success: isRepairable } = repair(locateRes);

//         if (!isRepairable) {
//           vscode.window.showInformationMessage(
//             "This model can't repair vulnerabilities. Proceed to manual repair."
//           );
//           return;
//         }

//         vscode.window.showInformationMessage(repairRes.newCode);
//       }
//     }
//   );
//   context.subscriptions.push(disposable);

//   disposable = vscode.commands.registerCommand(
//     "delore-vscode-extension.test-run-python-command",
//     () => {}
//   );

//   context.subscriptions.push(disposable);
//}

// This method is called when your extension is deactivated
export function deactivate() {}
