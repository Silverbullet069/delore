import * as vscode from 'vscode';
import { spawn, execFileSync } from 'child_process';
import * as fs from 'fs';

import * as logger from '../utils/logger';
import { Either, makeLeft, makeRight } from '../utils/either';
import { safeJsonParse } from '../utils/typeSafeJson';

type ExecutePythonErrorType =
  | 'PATH_TO_BINARY_IS_EMPTY'
  | 'PATH_TO_SCRIPT_IS_EMPTY'
  | 'PATH_TO_CWD_IS_EMPTY'
  | 'BINARY_NOT_FOUND'
  | 'SCRIPT_NOT_FOUND'
  | 'CWD_NOT_FOUND'
  | 'PARAM_IS_EMPTY'
  | 'INPUT_NOT_JSON'
  | 'OUTPUT_NOT_JSON'
  | 'PYTHON_PROCESS_TERMINATED'
  | 'USER_CANCELLED_PROCESS';

type ExecutePythonError = {
  type: ExecutePythonErrorType;
  msg: string;
};

/**
 * Run Python code from TypeScript.
 * Programmer should aware of its return type
 *
 * @param absPathToBinary The absolute path to python binary
 * @param absPathToScript The absolute path to python helper script
 * @paramInJSON paramInJSON The parameter that will be passed into python helper script, usually a code function
 * @returns using Either design pattern
 */
export const executePythonCommandSyncService = (
  absPathToBinary: string,
  absPathToScript: string,
  params: string[], // both default + user-defined + dynamic
  absPathToCwd: string
): Either<ExecutePythonError, string> => {
  // debug
  // logger.debugInfo(basename(module.filename), pathToCwd);
  // logger.debugInfo(basename(module.filename), pathToBinary);
  // logger.debugInfo(basename(module.filename), pathToScript);
  // logger.debugInfo(basename(module.filename), param);

  /* ==================================================== */
  /* Error Handling                                       */
  /* ==================================================== */

  if (!absPathToBinary) {
    return makeLeft({
      type: 'PATH_TO_BINARY_IS_EMPTY',
      msg: `Check again your path to binary.\n${new Error().stack}`
    });
  }

  if (!fs.existsSync(absPathToBinary)) {
    return makeLeft({
      type: 'BINARY_NOT_FOUND',
      msg: `Check again your path to binary: ${absPathToBinary}\n${new Error().stack}`
    });
  }

  if (!absPathToScript) {
    return makeLeft({
      type: 'PATH_TO_SCRIPT_IS_EMPTY',
      msg: `Check again your path to script.\n${new Error().stack}`
    });
  }

  if (!fs.existsSync(absPathToScript)) {
    return makeLeft({
      type: 'SCRIPT_NOT_FOUND',
      msg: `Check again your path to script: ${absPathToScript}\n${new Error().stack}`
    });
  }

  if (!absPathToCwd) {
    return makeLeft({
      type: 'PATH_TO_CWD_IS_EMPTY',
      msg: `Check again your path to cwd.\n${new Error().stack}`
    });
  }

  if (!fs.existsSync(absPathToCwd)) {
    return makeLeft({
      type: 'CWD_NOT_FOUND',
      msg: `Check again your path to cwd: ${absPathToCwd}\n${new Error().stack}`
    });
  }

  if (!params || params.length === 0) {
    return makeLeft({
      type: 'PARAM_IS_EMPTY',
      msg: `Check again your parameter.\n${new Error().stack}`
    });
  }

  try {
    const lastParam = params[params.length - 1];
    const lastParamObj = safeJsonParse(params[params.length - 1]);
  } catch (err) {
    return makeLeft({
      type: 'INPUT_NOT_JSON',
      msg: `Check again your params: ${params}\n${new Error().stack}`
    });
  }

  /* ==================================================== */
  /* Main Logic                                           */
  /* ==================================================== */

  // Python command structure: binary + script + static param + dynamic param (json)

  try {
    // debug
    // logger.debugSuccess('absPathToBinary: ' + absPathToBinary);
    // logger.debugSuccess('absPathToScript: ' + absPathToScript);
    // logger.debugSuccess('absPathToCwd: ' + absPathToCwd);
    // logger.debugSuccess([...params]);

    const stdout = execFileSync(absPathToBinary, [absPathToScript, ...params], {
      cwd: absPathToCwd,
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'pipe'] // stdin and stdout are piped, stderr is piped as well
    });

    // debug
    logger.debugSuccess(stdout);

    const lines = stdout.trim().split('\n');
    const numOfLines = lines.length;
    const lastLine = lines[numOfLines - 1];

    // By design, Python output consists of log and json string. The last line is the json string.

    try {
      const lastLineObj = safeJsonParse(lastLine);
      // logger.debugInfo([...lines]); // also include last line, that's also good

      return makeRight(lastLine);
    } catch (err) {
      return makeLeft({
        type: 'OUTPUT_NOT_JSON',
        msg: `Error: ${err}\nCheck again your lastLine: ${lastLine}\n${new Error().stack}`
      });
    }
  } catch (stderr) {
    return makeLeft({
      type: 'PYTHON_PROCESS_TERMINATED',
      msg: `${stderr}\n${new Error().stack}`
    });
  }
};

/**
 * Async version of `executePythonCommandSync()`
 *
 * @param absPathToBinary The absolute path to python binary
 * @param absPathToScript The absolute path to python helper script
 * @paramInJSON paramInJSON The parameter that will be passed into python helper script, usually a code function
 * @returns an Entity wrapped inside Promise
 */
export async function executePythonCommandAsyncService(
  absPathToBinary: string,
  absPathToScript: string,
  params: string[], // both default + user-defined + dynamic
  absPathToCwd: string,
  token: vscode.CancellationToken
): Promise<Either<ExecutePythonError, string>> {
  // debug
  // logger.debugInfo(basename(module.filename), pathToCwd);
  // logger.debugInfo(basename(module.filename), pathToBinary);
  // logger.debugInfo(basename(module.filename), pathToScript);
  // logger.debugInfo(basename(module.filename), param);

  /* ==================================================== */
  /* Error Handling                                       */
  /* ==================================================== */

  if (!absPathToBinary) {
    return makeLeft({
      type: 'PATH_TO_BINARY_IS_EMPTY',
      msg: `Check again your path to binary.\n${new Error().stack}`
    });
  }

  if (!fs.existsSync(absPathToBinary)) {
    return makeLeft({
      type: 'BINARY_NOT_FOUND',
      msg: `Check again your path to binary: ${absPathToBinary}\n${new Error().stack}`
    });
  }

  if (!absPathToScript) {
    return makeLeft({
      type: 'PATH_TO_SCRIPT_IS_EMPTY',
      msg: `Check again your path to script.\n${new Error().stack}`
    });
  }

  if (!fs.existsSync(absPathToScript)) {
    return makeLeft({
      type: 'SCRIPT_NOT_FOUND',
      msg: `Check again your path to script: ${absPathToScript}\n${new Error().stack}`
    });
  }

  if (!absPathToCwd) {
    return makeLeft({
      type: 'PATH_TO_CWD_IS_EMPTY',
      msg: `Check again your path to cwd.\n${new Error().stack}`
    });
  }

  if (!fs.existsSync(absPathToCwd)) {
    return makeLeft({
      type: 'CWD_NOT_FOUND',
      msg: `Check again your path to cwd: ${absPathToCwd}\n${new Error().stack}`
    });
  }

  if (!params || params.length === 0) {
    return makeLeft({
      type: 'PARAM_IS_EMPTY',
      msg: `Check again your parameter.\n${new Error().stack}`
    });
  }

  try {
    const lastParam = params[params.length - 1];
    const lastParamObj = await safeJsonParse(params[params.length - 1]);
  } catch (err) {
    return makeLeft({
      type: 'INPUT_NOT_JSON',
      msg: `Check again your params: ${params}\n${new Error().stack}`
    });
  }

  /* ==================================================== */
  /* Main Logic                                           */
  /* ==================================================== */

  return new Promise((resolve, reject) => {
    // Python command structure: binary + script + static param + dynamic param (json)
    const pythonProcess = spawn(absPathToBinary, [absPathToScript, ...params], {
      cwd: absPathToCwd
    });

    // only works inside async function
    token.onCancellationRequested(() => {
      logger.notifyError('User cancelled process!');
      pythonProcess.kill(); // send 'SIGTERM' signal
    });

    let stdOut = '';
    pythonProcess.stdout.on('data', (data) => {
      stdOut += data;
    });

    let stdErr = '';
    pythonProcess.stderr.on('data', (data) => {
      stdErr += data;
    });

    pythonProcess.on('close', async (code, signal) => {
      if (signal === 'SIGTERM') {
        resolve(
          makeLeft({
            type: 'USER_CANCELLED_PROCESS',
            msg: `User cancelled progress! DeLoRe stopped!`
          })
        );
      } else if (code === 0) {
        // Process exited normally

        const lines = stdOut.trim().split('\n');
        const numOfLines = lines.length;
        const lastLine = lines[numOfLines - 1];

        // By design, Python output consists of log and json string. The last line is the json string.

        try {
          const lastLineObj = await safeJsonParse(lastLine);

          resolve(makeRight(lastLine)); // json string
        } catch (err) {
          resolve(
            makeLeft({
              type: 'OUTPUT_NOT_JSON',
              msg: `Check again your lastLine: ${lastLine}
                Stack: ${new Error().stack}`
            })
          );
        }
      } else {
        // Process existed with an error
        resolve(
          makeLeft({
            type: 'PYTHON_PROCESS_TERMINATED',
            msg: `Debug: ${stdOut}
              Python process exited with code ${code}: ${stdErr}
              Stack: ${new Error().stack}`
          })
        );
      }
    });
  });
}

// Run Shell code from TypeScript, ...
