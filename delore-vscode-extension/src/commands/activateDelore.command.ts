import * as vscode from 'vscode';
import * as logger from '../utils/logger';

import { InMemoryRepository } from '../repositories/inMemory.repository';
import { isLeft, unwrapEither } from '../utils/either';

import { OutlineTreeDataProvider } from '../views/customTreeDataProvider';
import { displayResultService } from '../services/displayResult.service';
import { modelService } from '../services/model.service';
import { SUPPORTED_LANGUAGES } from '../constants/config';
import path from 'path';

export const activateDeloreCommand = (
  extensionPath: string,
  outlineTreeDataProvider: OutlineTreeDataProvider
) => {
  // for registerTextEditorCommand()
  return async (editor: vscode.TextEditor): Promise<void> => {
    let isModelRun = false;

    // actually no need to check, but just in case not use registerTextEditorCommand() anymore
    if (!editor) {
      logger.debugInfo(`Delore not supported empty editor!`);
      return;
    }

    if (
      !SUPPORTED_LANGUAGES.includes(path.extname(editor.document.uri.fsPath))
    ) {
      logger.debugInfo(`Delore not supported this file extension!`);
      return;
    }

    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Window,
        title: 'Running DeLoRe'
        // only Notification can be cancelled
      },
      async (progress, token) => {
        token.onCancellationRequested(() => {
          logger.notifyInfo('User cancelled DeLoRe Extension!');
        });

        /* ================================================== */
        /* Detection                                          */
        /* ================================================== */

        progress.report({ message: 'Detection...' });

        await vscode.window.withProgress(
          {
            location: {
              viewId: 'detectionModelView'
            },
            title: 'Running Detection...'
          },
          async (progress, token) => {
            token.onCancellationRequested(() => {
              logger.notifyInfo('User cancelled Detection service!');
            });

            // detectionService(extensionPath, editorFsPath, funcs);
            const detectionServiceEither = await modelService(
              extensionPath,
              'detection',
              editor
            );

            if (isLeft(detectionServiceEither)) {
              const err = unwrapEither(detectionServiceEither);
              logger.debugError(err.type, '\n', err.msg);
            }

            const success = unwrapEither(detectionServiceEither);
            if (success === 'RUN') {
              isModelRun = true;
            }
          }
        );

        if (!isModelRun) {
          return;
        }

        /* ================================================== */
        /* Localization                                       */
        /* ================================================== */

        progress.report({ message: 'Localization...' });

        await vscode.window.withProgress(
          {
            location: {
              viewId: 'localizationModelView'
            },
            title: 'Running Localization...'
          },
          async (progress, token) => {
            token.onCancellationRequested(() => {
              logger.notifyInfo('User cancelled Localization service!');
            });

            // NOTE: this service has a little different logic, since it uses detection result
            const localizationServiceEither = await modelService(
              extensionPath,
              'localization',
              editor
            );

            if (isLeft(localizationServiceEither)) {
              const err = unwrapEither(localizationServiceEither);
              logger.debugError(err.type, '\n', err.msg);
              return;
            }

            const success = unwrapEither(localizationServiceEither);

            if (success === 'RUN') {
              isModelRun = true;
            }
          }
        );

        if (!isModelRun) {
          return;
        }

        /* ============================================== */
        /* Repairation                                    */
        /* ============================================== */

        progress.report({ message: 'Repairation...' });

        await vscode.window.withProgress(
          {
            location: {
              viewId: 'repairationModelView'
            },
            title: 'Running Localization...'
          },
          async (progress, token) => {
            token.onCancellationRequested(() => {
              logger.notifyInfo('User cancelled Repairation service!');
            });

            // NOTE: for now,  repairation use VSCode's languageModels API
            const repairationServiceEither = await modelService(
              extensionPath,
              'repairation',
              editor
            );

            if (isLeft(repairationServiceEither)) {
              const err = unwrapEither(repairationServiceEither);
              logger.debugError(err.type, '\n', err.msg);
              return;
            }
          }
        );
      }
    );

    // only called when all 3 model has been run through
    const displayResultServiceEither = await displayResultService(
      extensionPath,
      editor
    );

    // handle error
    if (isLeft(displayResultServiceEither)) {
      const err = unwrapEither(displayResultServiceEither);
      logger.debugError(err.type, '\n', err.msg);
    }

    // update custom outline
    outlineTreeDataProvider.refresh();
  };
};
