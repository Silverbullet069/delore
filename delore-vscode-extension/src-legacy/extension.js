"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const child_process_1 = require("child_process");
function printInfo(msg) {
    console.log(msg);
    vscode.window.showInformationMessage(msg);
}
function printError(msg) {
    console.error(msg);
    vscode.window.showErrorMessage(msg);
}
function isFileExisted(filePath) {
    let flag = true;
    try {
        fs.accessSync(filePath, fs.constants.F_OK);
    }
    catch (err) {
        flag = false;
    }
    return flag;
}
function isCppFile(filePath) {
    if (!isFileExisted(filePath)) {
        return false;
    }
    const fileNameWithExtension = path.basename(filePath);
    return (fileNameWithExtension.endsWith(".cpp") ||
        fileNameWithExtension.endsWith(".c"));
}
function detect(content) {
    let codeSnippet = content;
    if (isCppFile(content)) {
        codeSnippet = fs.readFileSync(content, { encoding: "utf-8" });
    }
    console.log(codeSnippet);
    const detectModelPath = path.join(__dirname, "..", "..", "detection");
    process.chdir(detectModelPath);
    const pythonPath = path.resolve(detectModelPath, "py-detection", "bin", "python");
    const detectModule = "detect";
    console.log("so far so good");
    const output = (0, child_process_1.execSync)(`${pythonPath} -m ${detectModule} "${codeSnippet}"`)
        .toString()
        .trim();
    const isVulnerable = !(output[output.length - 1].length === 0);
    printInfo(isVulnerable ? "true" : "false");
    return isVulnerable;
}
function locate(code) {
    const test = code;
    const output = (0, child_process_1.execSync)("python3 -c 'print(\"Run Python Code from VSCode\")'")
        .toString()
        .trim();
    const arr = [];
    for (let i = 0; i < 10; ++i) {
        arr.push({ line: -1, col: -1 });
    }
    return arr;
}
function repair(locateRes) {
    const output = (0, child_process_1.execSync)("python3 -c 'print(\"Run Python Code from VSCode\")'")
        .toString()
        .trim();
    const len = locateRes.length;
    const arr = [];
    for (let i = 0; i < len; ++i) {
        arr.push({
            coor: locateRes[i],
            newCode: "New reliable code!",
            reliability: 0.0,
        });
    }
    return arr;
}
async function activate(context) {
    let disposable = vscode.commands.registerCommand("delore-vscode-extension.test-run-python-code", () => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const selection = editor.selection;
            const highlightedText = editor.document.getText(selection);
            if (!highlightedText) {
                vscode.window.showErrorMessage("Please highlight your code!");
                return;
            }
            let isVulnerable = detect(highlightedText);
            if (!isVulnerable) {
                vscode.window.showInformationMessage("Your code MIGHT be clean!");
                return;
            }
            else {
                vscode.window.showInformationMessage("Vulnerability found!");
            }
        }
    });
    context.subscriptions.push(disposable);
    const highlightedDecorationType = vscode.window.createTextEditorDecorationType({
        backgroundColor: "rgba(255, 0, 0, 0.7)",
    });
    disposable = vscode.commands.registerCommand("delore-vscode-extension.highlight-code", () => {
        const activeEditor = vscode.window.activeTextEditor;
        if (activeEditor) {
            const { document } = activeEditor;
            const rangesToHighlight = [
                { range: new vscode.Range(0, 0, 0, 5) },
                { range: new vscode.Range(2, 0, 2, 10) },
            ];
            activeEditor.setDecorations(highlightedDecorationType, rangesToHighlight);
        }
    });
    context.subscriptions.push(disposable);
    disposable = vscode.commands.registerCommand("delore-vscode-extension.split-editor-right", () => {
        const activeEditor = vscode.window.activeTextEditor;
        if (activeEditor) {
            vscode.commands.executeCommand("workbench.action.splitEditorRight");
        }
        else {
            vscode.window.showErrorMessage("No active editor found.");
        }
    });
    context.subscriptions.push(disposable);
    disposable = vscode.commands.registerCommand("delore-vscode-extension.scroll-to-position", () => {
        const activeEditor = vscode.window.activeTextEditor;
        if (activeEditor) {
            const lineNum = 10;
            const scrollBottomMargin = 4;
            const selectionTopMargin = 1;
            const range = activeEditor.document.lineAt(lineNum + scrollBottomMargin).range;
            activeEditor.selection = new vscode.Selection(new vscode.Position(lineNum - selectionTopMargin, 0), new vscode.Position(lineNum - selectionTopMargin, 0));
            activeEditor.revealRange(range);
        }
        else {
            vscode.window.showErrorMessage("No active editor found.");
        }
    });
    context.subscriptions.push(disposable);
    disposable = vscode.commands.registerCommand("delore-vscode-extension.get-scroll-position", () => {
        const activeEditor = vscode.window.activeTextEditor;
        if (activeEditor) {
            const visibleRanges = activeEditor.visibleRanges;
            if (visibleRanges.length > 0) {
                const firstVisibleRange = visibleRanges[0];
                vscode.window.showInformationMessage(`Scroll position: Line ${firstVisibleRange.start.line + 1}, Column ${firstVisibleRange.start.character + 1}`);
            }
            else {
                vscode.window.showWarningMessage("No visible range found.");
            }
        }
        else {
            vscode.window.showErrorMessage("No active editor found.");
        }
    });
    context.subscriptions.push(disposable);
    disposable = vscode.commands.registerCommand("delore-vscode-extension.locate-vul", () => {
        const activeEditor = vscode.window.activeTextEditor;
        if (activeEditor) {
            const activeEditorFullText = activeEditor.document.getText();
            vscode.window.showInformationMessage(`Full text: ${activeEditorFullText}`);
            const selection = activeEditor.selection;
            if (selection) {
                const selectedStartPos = selection.start;
                const selectedEndPos = selection.end;
                const highlightedText = activeEditor.document.getText(selection);
                vscode.window.showInformationMessage(`Selected text: ${highlightedText}`);
                vscode.window.showInformationMessage(`Start position: Line ${selectedStartPos.line + 1}, Column ${selectedStartPos.character + 1} \n
            End position: Line ${selectedEndPos.line + 1}, Column ${selectedEndPos.character + 1}`);
                vscode.commands
                    .executeCommand("workbench.action.splitEditorRight")
                    .then(() => {
                    const newEditor = vscode.window.activeTextEditor;
                    if (newEditor) {
                        newEditor
                            .edit((editBuilder) => {
                            editBuilder.insert(selectedStartPos, highlightedText);
                        })
                            .then(() => {
                            vscode.window.showInformationMessage("Text pasted at specified position in the new editor.");
                        }, (error) => {
                            vscode.window.showErrorMessage("Failed to paste text in the new editor: " + error);
                        });
                    }
                    else {
                        vscode.window.showErrorMessage("Failed to get the newly created editor.");
                    }
                });
            }
        }
        else {
            vscode.window.showErrorMessage("No active editor found.");
        }
    });
    context.subscriptions.push(disposable);
}
exports.activate = activate;
function deactivate() { }
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map