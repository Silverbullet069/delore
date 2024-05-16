export interface DetectionResult {
  name: string;
  result: boolean;
}

export interface LocalizationResult {
  name: string;
  results: {
    line: number;
    score: number; // For now, my assumption
  }[];
}

export interface RepairationResult {
  name: string;
  results: {
    line: number;
    suggestion: string; // For now, assumption
  }[];
}

/* ============================================ */
/* FUNCTION                                     */
/* ============================================ */

export type ID = string | number;
export interface Func {
  id: ID;
  name: string;
  editorId: string;
  sanitizedContent: string;

  /* zero-based */
  lines: {
    num: number;
    content: string;
    startChar: number;
    endChar: number;
  }[];

  detectionResults: DetectionResult[];
  localizationResults: LocalizationResult[];
  repairationResults: RepairationResult[];
}
