import * as vscode from 'vscode';

/* ====================================================== */
/* Standardized Input (what TypeScript knows)             */
/* ====================================================== */

export type DetectionModelInputFunc = {
  unprocessedContent: string;
};

export type LocalizationModelInputFunc = {
  unprocessedContent: string;
};

export type RepairationModelInputFunc = {
  unprocessedContent: string;
  vulLines: {
    content: string;
    num: number;
  }[];
};

export type DetectionModelInput = {
  path: string;
  funcs: DetectionModelInputFunc[];
};

export type LocalizationModelInput = {
  path: string;
  funcs: LocalizationModelInputFunc[];
};

export type RepairationModelInput = {
  path: string;
  funcs: RepairationModelInputFunc[];
};

/* ====================================================== */
/* Standardized Output (what Python, Shell, ... knows)    */
/* ====================================================== */

export type DetectionModelOutput = {
  modelName: string;
  isVulnerable: boolean;
};

export type LocalizationModelOutput = {
  modelName: string;
  lines: {
    content: string;
    num: number;
    score: number;
    isVulnerable: boolean; // since the extension didn't know about the model logic so just the score isn't enough to determine whether or not a line is vulnerable, that's Python job
  }[];
};

export type RepairationModelOutput = {
  modelName: string;
  lines: {
    content: string;
    cwe: string;
    reason: string;
    fix: string;
  }[];
};

/* ====================================================== */
/* Application State                                      */
/* ====================================================== */

export type LineState = {
  numOnEditor: number;
  processedContent: string;
  startCharOnEditor: number;
  endCharOnEditor: number;
};

export type FuncState = {
  name: string;
  unprocessedContent: string;
  processedContent: string;
  processedContentHash: string; // the ID
  isRunDelore: boolean;

  lines: LineState[];

  detectionResults: DetectionModelOutput[];
  localizationResults: LocalizationModelOutput[];
  repairationResults: RepairationModelOutput[];

  mergeDetectionResult?: DetectionModelOutput;
  mergeLocalizationResult?: LocalizationModelOutput;
  mergeRepairationResult?: RepairationModelOutput;

  // more role can be added here in the future
  // ...
};

export type TempState = {
  fsPath: string;
  decoration: vscode.TextEditorDecorationType; // each temp file must maintain a decoration
};

export type EditorState = {
  editorFsPath: string;

  temp?: TempState;
  funcs: FuncState[];
};
