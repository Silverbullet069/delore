import * as vscode from 'vscode';
import * as logger from '../utils/logger';

import { isLeft, unwrapEither } from '../utils/either';

import { OutlineTreeDataProvider } from '../views/customTreeDataProvider';
import { displayResultService } from '../services/displayResult.service';
import { runModelService } from '../services/runModel.service';
import { EXTENSION_ID, SUPPORTED_LANGUAGES } from '../constants/config';
import path from 'path';
import { ServiceHandler } from '../ServiceHandler';

export const activateDeloreCommandHandler = (
  extensionPath: string,
  outlineTreeDataProvider: OutlineTreeDataProvider
): vscode.Disposable => {
  //
  return vscode.commands.registerTextEditorCommand(
    `${EXTENSION_ID}.activateDelore`,
    async (editor: vscode.TextEditor): Promise<void> => {
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
        async (statusBarProgress, statusBarToken) => {
          // idk if this is needed?
          statusBarToken.onCancellationRequested(() => {
            logger.notifyInfo('User cancelled DeLoRe Extension!');
            return;
          });

          let isModelRun = false;

          /* ============================================ */

          // Detection
          statusBarProgress.report({ message: 'Detection...' });

          await vscode.window.withProgress(
            {
              location: vscode.ProgressLocation.Notification,
              title: 'Detection',
              cancellable: true // Notification can be cancelled
            },
            async (detectionProgress, detectionToken) => {
              // start view progress
              await vscode.window.withProgress(
                {
                  location: {
                    viewId: 'detectionModelView'
                  },
                  title: 'Detection Model View Progress',
                  cancellable: true
                },
                async (viewProgress, viewToken) => {
                  viewProgress.report({ message: 'Running...' });

                  const detectionRunModelServiceEither =
                    await ServiceHandler.instance.runModelServiceWrapper(
                      extensionPath,
                      'detection',
                      editor,
                      detectionProgress,
                      detectionToken
                    );

                  if (isLeft(detectionRunModelServiceEither)) {
                    const err = unwrapEither(detectionRunModelServiceEither);
                    logger.debugError(err.type, '\n', err.msg);
                    isModelRun = false;
                    return;
                  }

                  isModelRun = unwrapEither(detectionRunModelServiceEither);
                  return;
                }
              ); // end of view progress
            }
          ); // end of notification progress

          // stop from going further
          if (!isModelRun) {
            return;
          }

          /* ================================================== */
          /* Localization                                       */
          /* ================================================== */

          statusBarProgress.report({ message: 'Localization...' });
          await vscode.window.withProgress(
            {
              location: vscode.ProgressLocation.Notification,
              title: 'Localization',
              cancellable: true
            },
            async (localizationProgress, localizationToken) => {
              localizationToken.onCancellationRequested(() => {
                logger.notifyInfo('User cancelled Localization service!');
                isModelRun = false;
                return;
              });

              await vscode.window.withProgress(
                {
                  location: {
                    viewId: 'localizationModelView'
                  },
                  title: 'Localization Model View Progress',
                  cancellable: true
                },
                async (viewProgress, viewToken) => {
                  viewProgress.report({ message: 'Running...' });

                  // NOTE: this service has a little different logic, since it uses detection result
                  const localizationRunModelServiceEither =
                    await ServiceHandler.instance.runModelServiceWrapper(
                      extensionPath,
                      'localization',
                      editor,
                      localizationProgress,
                      localizationToken
                    );

                  if (isLeft(localizationRunModelServiceEither)) {
                    const err = unwrapEither(localizationRunModelServiceEither);
                    logger.debugError(err.type, '\n', err.msg);
                    isModelRun = false;
                    return;
                  }

                  isModelRun = unwrapEither(localizationRunModelServiceEither);
                  return;
                }
              );
            }
          );

          // stop from going further
          if (!isModelRun) {
            return;
          }

          /* ============================================== */
          /* Repairation                                    */
          /* ============================================== */

          statusBarProgress.report({ message: 'Repairation...' });

          await vscode.window.withProgress(
            {
              location: vscode.ProgressLocation.Notification,
              title: 'Repairation',
              cancellable: true
            },
            async (repairationProgress, repairationToken) => {
              await vscode.window.withProgress(
                {
                  location: {
                    viewId: 'repairationModelView'
                  },
                  title: 'Repairation Model View Progress',
                  cancellable: true
                },
                async (viewProgress, viewToken) => {
                  viewProgress.report({ message: 'Running...' });

                  // NOTE: for now, repairation use VSCode's languageModels API
                  const repairationRunModelServiceEither =
                    await ServiceHandler.instance.runModelServiceWrapper(
                      extensionPath,
                      'repairation',
                      editor,
                      repairationProgress,
                      repairationToken
                    );

                  if (isLeft(repairationRunModelServiceEither)) {
                    const err = unwrapEither(repairationRunModelServiceEither);
                    logger.debugError(err.type, '\n', err.msg);
                    isModelRun = false;
                    return;
                  }

                  isModelRun = unwrapEither(repairationRunModelServiceEither);
                  return;
                }
              );
            }
          );
        }
      );

      // NOTE: you can still check isModelRun here if you have other models in the future

      // only called when all 3 model has been run through
      // TODO: might not the best UI decision, user wants to see results asap
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
    }
  );
};
