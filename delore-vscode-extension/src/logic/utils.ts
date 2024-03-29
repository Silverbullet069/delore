import { window, Selection, TextLine } from "vscode";

import { spawn } from "child_process";
import { existsSync } from "fs";

import {
  EXTENSION_NAME,
  EXTENSION_ID,
  PYTHON_BIN_ABS_PATH,
  CHECK_FUNCTION_PYTHON_SCRIPT,
} from "../data/constant";
import { printInfoMsg } from "../ui/render";

/* ---------------------------------- */
/*             Run Python             */
/* ---------------------------------- */

const executePythonCommand = async (
  script: string,
  param: string | string[]
): Promise<string> => {
  return new Promise((resolve, reject) => {
    // python tmp.py param1 param2 ...

    let pythonProcess;
    if (typeof param === "string") {
      pythonProcess = spawn(PYTHON_BIN_ABS_PATH, [script, param]);
    } else {
      pythonProcess = spawn(PYTHON_BIN_ABS_PATH, [script, ...param]);
    }

    let output = "";

    pythonProcess.stdout.on("data", (data) => {
      output += data;
    });

    pythonProcess.stderr.on("data", (data) => {
      reject(new Error(data.toString()));
    });

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        resolve(output); // this is the output
      } else {
        reject(new Error(`Python process exited with code ${code}`));
      }
    });
  });
};

/* ---------------------------------- */
/*          C++ manipulation          */
/* ---------------------------------- */

export const isCOrCPlusPlusFile = (fileExt: string | undefined): boolean =>
  fileExt === "cpp" || fileExt === "c";

export const isFunctionInGeneral = async (code: string) => {
  const ans = await executePythonCommand(CHECK_FUNCTION_PYTHON_SCRIPT, code);

  printInfoMsg(ans);
  printInfoMsg(typeof ans);
};

// export const parseFunction = async (code: string):
