import { basename } from 'path';
import { spawn } from 'child_process';
import * as logger from './logger';

type PythonStandarizedInput = {
  json: string;
};

type PythonStandarizedOutput = {
  json: string;
};

/**
 * Run Python code from TypeScript.
 * Programmer should aware of its return type
 *
 * @param absPathToBinary The absolute path to python binary
 * @param absPathToScript The absolute path to python helper script
 * @param param The parameter that will be passed into python helper script, usually a code function
 * @returns A promise that resolve to a value with the same return type that is pre-specified
 */
export async function executePythonCommand<T>(
  absPathToBinary: string,
  absPathToScript: string,
  param: string,
  absPathToCwd?: string
): Promise<T> {
  // debug
  // logger.debugInfo(basename(module.filename), pathToCwd);
  // logger.debugInfo(basename(module.filename), pathToBinary);
  // logger.debugInfo(basename(module.filename), pathToScript);
  // logger.debugInfo(basename(module.filename), param);

  return new Promise((resolve, reject) => {
    const pythonProcess = spawn(absPathToBinary, [absPathToScript, param], {
      cwd: absPathToCwd
    });

    let stdOut = '';
    pythonProcess.stdout.on('data', (data) => {
      stdOut += data;
    });

    let stdErr = '';
    pythonProcess.stderr.on('data', (data) => {
      stdErr += data;
      // reject(new Error(data.toString()));
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        const lines: string[] = stdOut.trimEnd().split('\n');
        const lastLine: string = lines[lines.length - 1];

        logger.debugInfo(basename(module.filename), lastLine);

        // By design, all Python script print
        // - verbose log
        // - one JSON string at the last line whose object is structured by PythonOutput

        const res: PythonOutput<T> = JSON.parse(lastLine);
        const msg: string = res.msg;
        const data: T = res.data;

        resolve(data);
      } else {
        reject(new Error(`Python process exited with code ${code}: ${stdErr}`));
      }
    });
  });
}

// Run Shell code from TypeScript, ...
