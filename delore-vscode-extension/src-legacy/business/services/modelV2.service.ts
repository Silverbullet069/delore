import * as vscode from 'vscode';
import * as logger from '../../../src/views/logger';

import {
  EXTENSION_ID,
  ModelRole,
  modelRoles,
  resourceManager
} from '../../../src/constants/config';
import {
  Either,
  isLeft,
  isStrictNever,
  makeLeft,
  makeRight,
  unwrapEither
} from '../../../src/utils/either';
import { ActiveModelSetting } from '../../../src/model/settings.model';
import {
  DetectionModelInput,
  DetectionModelInputFunc,
  DetectionModelOutput,
  FuncState,
  LocalizationModelInput,
  LocalizationModelInputFunc,
  LocalizationModelOutput,
  RepairationModelInput,
  RepairationModelInputFunc,
  RepairationModelOutput
} from '../../../src/model/state.model';
import {
  capitalize,
  isDetectionModelOutput,
  isLocalizationModelOutput,
  isRepairationModelOutput
} from '../../../src/utils/comparison';
import {
  safeJsonParse,
  safeJsonStringify
} from '../../../src/utils/typeSafeJson';
import { executePythonCommandSync } from '../../../src/utils/shell';
import { InMemoryRepository } from '../../../src/repositories/inMemory.repository';
import { basename } from 'path';

/* ====================================================== */
/* Model Service                                          */
/* ====================================================== */

export type ModelServiceErrorType =
  | 'EXTENSION_PATH_NOT_FOUND'
  | 'ROLE_NOT_FOUND'
  | 'EDITOR_NOT_FOUND'
  | 'GET_MODEL_FROM_RESOURCE_MANAGER_ERR'
  | 'ACTIVE_MODEL_SETTING_NOT_EXIST_ERR'
  | 'UPDATE_MERGE_RESULT_IN_ONE_EDITOR_ERR'
  | 'GET_FUNCS_IN_ONE_EDITOR_ERR';

// When I design Error like this, the design must give the caller context about all the outcomes, without resolves to Context Switching
export type ModelServiceError = {
  type: ModelServiceErrorType;
  msg: string;
};

export type ModelServiceSuccessType = 'NOT_RUN' | 'RUN';

export const modelService = async (
  extensionPath: string,
  modelRole: ModelRole,
  editor: vscode.TextEditor
): Promise<Either<ModelServiceError, ModelServiceSuccessType>> => {
  /* ===================================================== */
  /* Runtime Parameter Check                               */
  /* ===================================================== */

  if (!extensionPath) {
    return makeLeft({
      type: 'EXTENSION_PATH_NOT_FOUND',
      msg: `Check again your extension path.\n${new Error().stack}`
    });
  }

  if (!modelRoles.includes(modelRole)) {
    return makeLeft({
      type: 'ROLE_NOT_FOUND',
      msg: `Check again your model role: ${modelRole}.\n${new Error().stack}`
    });
  }

  if (!editor) {
    return makeLeft({
      type: 'EDITOR_NOT_FOUND',
      msg: `Check again your editor.\n${new Error().stack}`
    });
  }

  // flag
  let isModelRun = false;
  const editorFsPath = editor.document.uri.fsPath;

  /* ==================================================== */
  /* Extract Funcs                                        */
  /* ==================================================== */

  const funcsEither =
    InMemoryRepository.getInstance().getFuncsInOneEditor(editorFsPath);

  // err handle
  if (isLeft(funcsEither)) {
    const err = unwrapEither(funcsEither);
    return makeLeft({
      type: 'GET_FUNCS_IN_ONE_EDITOR_ERR',
      msg: `Model role: ${modelRole}\n${err.type}\n${err.msg}` // no stack
    });
  }

  // extract funcs
  const funcs = unwrapEither(funcsEither);

  /* ==================================================== */
  /* Get All Models                                       */
  /* ==================================================== */

  const allModelsEither = resourceManager.getModelsByRole(modelRole);

  // err handle
  if (isLeft(allModelsEither)) {
    const err = unwrapEither(allModelsEither);
    return makeLeft({
      type: 'GET_MODEL_FROM_RESOURCE_MANAGER_ERR',
      msg: `Model role: ${modelRole}\n${err.type}\n${err.msg}` // no stack
    });
  }

  // extract all models
  const allModels = unwrapEither(allModelsEither);

  /* ==================================================== */
  /* VSCode Setting                                       */
  /* ==================================================== */

  const config = vscode.workspace.getConfiguration(EXTENSION_ID);
  const allModelSettings = config.get<ActiveModelSetting[]>(
    `${modelRole}.active`
  );

  if (!allModelSettings) {
    return makeLeft({
      type: 'ACTIVE_MODEL_SETTING_NOT_EXIST_ERR',
      msg: `${EXTENSION_ID}.${modelRole}.active setting not existed!\n${new Error().stack}`
    });
  }

  const activeModelSettings = allModelSettings.filter(
    (model) => model.isActive
  );

  const activeModelNames = activeModelSettings.map((setting) => setting.name);

  const activeModels = allModels.filter((model) =>
    // case-insensitive
    activeModelNames
      .map((name) => name.toLowerCase())
      .includes(model.name.toLowerCase())
  );

  // debug
  logger.debugSuccess(
    `File: ${basename(module.filename)}
    Function: ${modelService.name}
    Active Models: ${activeModels.map((model) => model.name)}`
  );

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: `${capitalize(modelRole)} model`,
      cancellable: true
    },

    async (progress, token) => {
      token.onCancellationRequested(() => {
        logger.notifyInfo(`User cancelled ${capitalize(modelRole)} service!`);
      });

      // 1 active model for 1 specific role
      // Ad-hoc integration with VSCode's Language Model API
      if (
        modelRole === 'repairation' &&
        activeModels.length === 1 &&
        activeModels[0].name === 'github-copilot-gpt4'
      ) {
        // TODO: call prompt here

        return; // go straight to merge result, which doesn't do anything since there aren't logic for repairation
      }

      // iterate active models in 1 role
      activeModels.forEach(async (model) => {
        /* ============================================== */

        progress.report({ message: `${model.name} condition checking...` });

        /* ============================================== */

        const pathEither = resourceManager.getPathToPythonBinary(extensionPath);

        // can't return either since we're inside a forEach(), log is good enough.
        if (isLeft(pathEither)) {
          const err = unwrapEither(pathEither);
          logger.debugError(err.type, '\n', err.msg);
          return; // nuke the model
        }
        const absPathToBinary = unwrapEither(pathEither);

        /* ============================================== */

        if (model.relPathToScript === '') {
          logger.debugError(
            `Model role: ${modelRole}\nModel ${model.name} don't have relPathToScript. Check constants/config.ts!`
          );
          return; // nuke the model
        }

        const absPathToScript = extensionPath + model.relPathToScript;

        /* ============================================== */

        if (model.relPathToCWD === '') {
          logger.debugError(
            `Model role: ${modelRole}\nModel ${model.name} don't have relPathToCWD. Check constants/config.ts!`
          );
          return; // nuke the model
        }

        const absPathToCwd = extensionPath + model.relPathToCWD;

        /* ============================================== */

        // construct an array of standardized input models
        const inputFuncs = funcs
          .filter((func) => {
            // only the start of chain - detection needed this
            if (modelRole === 'detection' && func.isRunDelore) {
              logger.debugSuccess(
                `Function: ${func.name} has run through Delore before!`
              );
              return false;
            }

            // skip func if it had result before with the same model
            if (
              func[`${modelRole}Results`].find(
                (modelResult) => modelResult.modelName === model.name
              )
            ) {
              logger.debugSuccess(
                `In ${modelRole} service, function has used model ${model.name}.`
              );
              return false;
            }

            // skip func if locate, repair but func not run through detection service
            if (
              (modelRole === 'localization' || modelRole === 'repairation') &&
              (func.detectionResults.length === 0 || !func.mergeDetectionResult)
            ) {
              logger.debugSuccess(`Haven't run through detection service.`);
              return false;
            }

            // skip func if locate, repair but func run through detection service but predicted as non-vul
            if (
              (modelRole === 'localization' || modelRole === 'repairation') &&
              func.mergeDetectionResult &&
              !func.mergeDetectionResult.isVulnerable
            ) {
              logger.debugSuccess(
                `Detection service predicted this function as non-vul.`
              );
              return false;
            }

            // skip func if repair but func not run through localization service
            if (
              modelRole === 'repairation' &&
              (func.localizationResults.length === 0 ||
                !func.mergeLocalizationResult)
            ) {
              logger.debugSuccess(`Haven't run through localization service.`);
              return false;
            }

            return true;
          })
          .map((func) => {
            // If a func get here, it means it passes all conditions can be run
            // PREPARE: this should be Repository's responsibility
            func.isRunDelore = true;

            if (modelRole === 'detection') {
              return {
                unprocessedContent: func.unprocessedContent
              } satisfies DetectionModelInputFunc;
            }

            if (modelRole === 'localization') {
              return {
                unprocessedContent: func.unprocessedContent
              } satisfies LocalizationModelInputFunc;
            }

            if (modelRole === 'repairation') {
              return {
                unprocessedContent: func.unprocessedContent,
                vulLines:
                  func.mergeLocalizationResult?.lines
                    .filter((line) => line.isVulnerable)
                    .map((line) => {
                      return { content: line.content, num: line.num };
                    }) || []
              } satisfies RepairationModelInputFunc;
            }

            isStrictNever(modelRole);
            return null; // it will never get here
          })
          .filter(
            (
              inputFunc
            ): inputFunc is
              | DetectionModelInputFunc
              | LocalizationModelInputFunc
              | RepairationModelInputFunc => inputFunc !== null
          );

        /* ============================================== */
        /* Input                                          */
        /* ============================================== */

        const defaultParams = model.args; // this can be empty
        const settingParams: string[] = []; // PREPARE: this can be implemented in the future
        const paramJSON = safeJsonStringify({
          path: editor.document.uri.fsPath,
          funcs: inputFuncs
        } satisfies
          | DetectionModelInput
          | LocalizationModelInput
          | RepairationModelInput);

        // NOTE: settingParams MUST COME AFTER defaultParams
        const params = [...defaultParams, ...settingParams, paramJSON];

        // Set flag
        isModelRun = true;

        // Run only once
        // Show notification on VSCode
        progress.report({
          message: `${model.name} - ${editor.document.fileName}`
        });

        const modelOutputEither = executePythonCommandSync(
          absPathToBinary,
          absPathToScript,
          params,
          absPathToCwd
        );

        /* ================================================ */
        /* Output                                           */
        /* ================================================ */

        if (isLeft(modelOutputEither)) {
          const err = unwrapEither(modelOutputEither);
          logger.debugError(err.type, '\n', err.msg);
          return; // nuke the func
        }

        const modelOutputJSON = unwrapEither(modelOutputEither);
        const modelOutput = safeJsonParse(modelOutputJSON);

        /* ================================================ */
        /* Runtime Check                                    */
        /* ================================================ */

        if (
          (modelRole === 'detection' && !isDetectionModelOutput(modelOutput)) ||
          (modelRole === 'localization' &&
            !isLocalizationModelOutput(modelOutput)) ||
          (modelRole === 'repairation' &&
            !isRepairationModelOutput(modelOutput))
        ) {
          logger.debugError(
            `Output JSON: ${modelOutputJSON} does not follows the ${modelRole} standard!`
          );
          return; // nuke the func
        }

        // iterate funcs
        funcs.forEach(async (func) => {
          if (
            modelRole === 'localization' &&
            isLocalizationModelOutput(modelOutput) &&
            modelOutput.lines.length !== func.lines.length
          ) {
            logger.debugError(
              `Output JSON: ${modelOutputJSON} return different lines length compare to the length in editor!`
            );
            return; // nuke the func
          }

          const updateEither =
            InMemoryRepository.getInstance().updateModelResultInOneFunc(
              modelRole,
              editorFsPath,
              func.processedContentHash,
              modelOutput as
                | DetectionModelOutput
                | LocalizationModelOutput
                | RepairationModelOutput
            );

          if (isLeft(updateEither)) {
            const err = unwrapEither(updateEither);
            logger.debugError(err.type, '\n', err.msg);
            return; // nuke the func
          }

          // debug
          // logger.debugSuccess(modelOutput);
        });
        // end vscode notification here
      });

      // NOTE: since some models use intermediary files, running 1 model in multiple process at the same time is impossible
      // const detectionOutputEithers = await Promise.all(promises);
    }
  );

  // merge result
  const updateMergeResultEither =
    InMemoryRepository.getInstance().updateMergeResultInOneEditor(
      modelRole,
      editorFsPath
    );

  if (isLeft(updateMergeResultEither)) {
    const err = unwrapEither(updateMergeResultEither);
    return makeLeft({
      type: 'UPDATE_MERGE_RESULT_IN_ONE_EDITOR_ERR',
      msg: `${err.type}\n${err.msg}` // no need stack if return another Either Left
    });
  }

  if (isModelRun) {
    return makeRight('RUN');
  }

  return makeRight('NOT_RUN');
};
