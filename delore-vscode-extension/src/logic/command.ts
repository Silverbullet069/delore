import {
  DecorationOptions,
  Position,
  Range,
  Selection,
  commands,
  window,
} from "vscode";

import { isCOrCPlusPlusFile, isFunctionInGeneral } from "./utils";

import { printErrorMsg, printInfoMsg } from "../ui/render";

export const test = (): void => {
  const editor = window.activeTextEditor;

  // Check editor is opened
  if (!editor) {
    printErrorMsg("There is no active editor!");
    return;
  }

  // Check file extension
  const fileUri = editor.document.uri;
  const fileExtension = fileUri.fsPath.split(".").pop();
  if (!isCOrCPlusPlusFile(fileExtension)) {
    printErrorMsg("Current open file in active editor is not a C/C++ file.");
  }

  // Method #1: Highlighting
  const selection = editor.selection;

  if (selection) {
    const highightedText = editor.document.getText(selection);
    printInfoMsg(highightedText);
    if (!isFunctionInGeneral(highightedText)) {
      printErrorMsg(
        "Current highlighted text in active editor is not a C/C++ function."
      );
    }
  }

  // Method #2: Extract all function
  // TODO: For now, lets assume that 1 file contains 1 CPP function
  const editorText = editor.document.getText();
  const sanitizeEditorText = editorText.replace("\n", " ");
  if (!isFunctionInGeneral(sanitizeEditorText)) {
    printErrorMsg("File content's is not a C/C++ function.");
  }

  // Highlighting
  const highlightRed1 = window.createTextEditorDecorationType({
    backgroundColor: "rgba(255, 0, 0, 0.7)", // Red background with 30% opacity
  });
  const highlightRed2 = window.createTextEditorDecorationType({
    backgroundColor: "rgba(255, 0, 0, 0.2)", // Red background with 30% opacity
  });
  const highlightRed2Range: DecorationOptions[] = [
    { range: new Range(0, 0, 0, 70) },
    { range: new Range(2, 0, 2, 70) },
  ];
  const highlightRed1Range: DecorationOptions[] = [
    { range: new Range(1, 0, 1, 70) },
  ];

  editor.setDecorations(highlightRed1, highlightRed1Range);
  editor.setDecorations(highlightRed2, highlightRed2Range);

  // Split editor right
  commands.executeCommand("workbench.action.splitEditorRight");

  // Auto scroll
  let lineNum = editor.selection.active.line;
  let currentPositionInLine = editor.selection.active.character;

  console.log("a", editor.document.lineAt(editor.selection.active.line).text);

  const scrollBottomMargin = 4;
  const selectionTopMargin = 1;

  // Define the range you want to scroll to
  const range = editor.document.lineAt(lineNum + scrollBottomMargin).range;

  editor.selection = new Selection(
    new Position(lineNum - selectionTopMargin, 0),
    new Position(lineNum - selectionTopMargin, 0)
  );

  // Scroll to the defined range
  editor.revealRange(range);
};

export const test2 = () => {
  isFunctionInGeneral("int add(int a, int b) { return a + b; }");
};

export const detect = () => {};

export const locate = () => {};

export const repair = () => {};

export const delore = () => {};
