import { window } from "vscode";
import { EXTENSION_NAME } from "../data/constant";

export const printInfoMsg = (msg: string): void => {
  console.log(`[INFO] ${EXTENSION_NAME}: ${msg}`);
  window.showInformationMessage(`${EXTENSION_NAME}: ${msg}`);
};

export const printWarningMsg = (msg: string): void => {
  console.warn(`[WARN] ${EXTENSION_NAME}: ${msg}`);
  window.showWarningMessage(`${EXTENSION_NAME}: ${msg}`);
};

export const printErrorMsg = (msg: string): void => {
  console.error(`[ERROR] ${EXTENSION_NAME}: ${msg}`);
  window.showErrorMessage(`${EXTENSION_NAME}: ${msg}`);
};
