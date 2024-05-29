import * as vscode from 'vscode';

/* ====================================================== */
/* Standardized Input (what TypeScript knows)             */
/* ====================================================== */

type UnprocessedContentLine = string;

export type DetectionModelInput = {
  // unprocessedContentFunc: string; // using '\n' to cut lines in Python is risky, change to each lines
  modelName: string;
  lines: UnprocessedContentLine[];
};

export type LocalizationModelInput = {
  // unprocessedContentFunc: string;
  modelName: string;
  lines: UnprocessedContentLine[];
};

export type RepairationModelInput = {
  // unprocessedContentFunc: string;
  // possibleVulLines: number[];
  modelName: string;
  lines: UnprocessedContentLine[];
  vulLineNums: number[];
};

/* ====================================================== */
/* Standardized Output (what Python, Shell, ... knows)    */
/* ====================================================== */

/* ====================================================== */
/* Detection                                              */
/* ====================================================== */

export type DetectionModelOutput = {
  modelName: string;
  isVulnerable: boolean;
};

// used to check Python stdout
export const isDetectionModelOutput = (
  obj: any
): obj is DetectionModelOutput => {
  // logger.debugSuccess(typeof obj.modelName, typeof obj.isVulnerable);

  return (
    obj &&
    'modelName' in obj &&
    typeof obj.modelName === 'string' &&
    'isVulnerable' in obj &&
    typeof obj.isVulnerable === 'boolean'
  );
};

/* ====================================================== */
/* Localization                                           */
/* ====================================================== */

export type LocalizationModelLinesOutput = {
  num: number;
  score: number;
  content: string;
  isVulnerable: boolean; // since the extension didn't know about the model logic so just the score isn't enough to determine whether or not a line is vulnerable, that's Python job
};

export type LocalizationModelOutput = {
  modelName: string;
  lines: LocalizationModelLinesOutput[];
};

// used to check Python stdout
export const isLocalizationModelOutput = (
  obj: any
): obj is LocalizationModelOutput => {
  return (
    obj &&
    'modelName' in obj &&
    typeof obj.modelName === 'string' &&
    'lines' in obj &&
    Array.isArray(obj.lines) &&
    obj.lines.every(
      (line: any) =>
        'content' in line &&
        typeof line.content === 'string' &&
        'num' in line &&
        typeof line.num === 'number' &&
        'score' in line &&
        typeof line.score === 'number' &&
        'isVulnerable' in line &&
        typeof line.isVulnerable === 'boolean'
    )
  );
};

/* ====================================================== */
/* Repairation                                            */
/* ====================================================== */

export type RepairationModelLinesOutput = {
  content: string;
  num: number;
  cwe: string;
  reason: string;
  fix: string;
  isVulnerable: boolean;
};

export type RepairationModelOutput = {
  modelName: string;
  lines: RepairationModelLinesOutput[];
};

// used to check Python stdout
export const isRepairationModelOutput = (
  obj: any
): obj is RepairationModelOutput => {
  return (
    obj &&
    'modelName' in obj &&
    typeof obj.modelName === 'string' &&
    'lines' in obj &&
    Array.isArray(obj.lines) &&
    obj.lines.every(
      (line: any) =>
        'content' in line &&
        typeof line.content === 'string' &&
        'num' in line &&
        typeof line.num === 'number' &&
        'isVulnerable' in line &&
        (typeof line.isVulnerable === 'boolean' ||
          typeof line.isVulnerable === 'string') && // json parse don't know to handle true and false string
        'cwe' in line &&
        typeof line.cwe === 'string' &&
        'reason' in line &&
        typeof line.reason === 'string' &&
        'fix' in line &&
        typeof line.fix === 'string'
    )
  );
};

/* ====================================================== */
/* Application State                                      */
/* ====================================================== */

export type LineState = {
  unprocessedContent: string;
  processedContent: string;
  numOnEditor: number;
  startCharOnEditor: number;
  endCharOnEditor: number;
};

export type FuncState = {
  name: string;
  unprocessedContentFunc: string;
  processedContentFunc: string;
  processedContentFuncHash: string; // the ID
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
  vulDecoration: vscode.TextEditorDecorationType;
  repairDecoration: vscode.TextEditorDecorationType;
};

export type EditorState = {
  editorFsPath: string;

  temp?: TempState;
  funcs: FuncState[];
};
