import * as vscode from 'vscode';
import * as logger from '../views/logger';

import { EXTENSION_ID, ModelRole, resourceManager } from '../constants/config';
import {
  Either,
  isLeft,
  isStrictNever,
  makeLeft,
  makeRight,
  unwrapEither
} from '../utils/either';
import { ActiveModelSetting } from '../model/settings.model';
import {
  DetectionModelInput,
  DetectionModelOutput,
  FuncState,
  LocalizationModelInput,
  LocalizationModelOutput,
  RepairationModelInput,
  RepairationModelOutput
} from '../model/state.model';
import {
  capitalize,
  isDetectionModelOutput,
  isLocalizationModelOutput,
  isRepairationModelOutput
} from '../utils/comparison';
import { safeJsonParse, safeJsonStringify } from '../utils/typeSafeJson';
import { executePythonCommandSync } from '../utils/shell';
import { InMemoryRepository } from '../repositories/inMemory.repository';
import { basename } from 'path';

/* ====================================================== */
/* Model Service                                          */
/* ====================================================== */

export type ModelServiceErrorType =
  | 'MODEL_FROM_RESOURCE_ERR'
  | 'ACTIVE_MODEL_SETTING_NOT_EXIST'
  | 'REPOSITORY_UPDATE_ERR';

// When I design Error like this, the design must give the caller context about all the outcomes, without resolves to Context Switching
export type ModelServiceError = {
  type: ModelServiceErrorType;
  msg: string;
};

export type ModelServiceSuccessType = 'NOT_RUN' | 'RUN';

const handleModelInput = (
  modelRole: ModelRole,
  func: FuncState
): Either<
  'ROLE_NOT_FOUND',
  DetectionModelInput | LocalizationModelInput | RepairationModelInput
> => {
  switch (modelRole) {
    case 'detection':
      return makeRight({
        unprocessedContent: func.unprocessedContent
      });
    case 'localization':
      return makeRight({
        unprocessedContent: func.unprocessedContent
      });
    case 'repairation':
      return makeRight({
        unprocessedContent: func.unprocessedContent,
        possibleVulLines: func.mergeLocalizationResult?.lines
          .filter((line) => line.isVulnerable)
          .map((line) => line.num)
      });
    default:
      return makeLeft('ROLE_NOT_FOUND');
  }
};

export const modelService = async (
  extensionPath: string,
  modelRole: ModelRole,
  editorFsPath: string,
  funcs: FuncState[]
): Promise<Either<ModelServiceError, ModelServiceSuccessType>> => {
  // flag
  let isModelRun = false;

  const allModelsEither = resourceManager.getModelsByRole(modelRole);

  if (isLeft(allModelsEither)) {
    const err = unwrapEither(allModelsEither);
    return makeLeft({
      type: 'MODEL_FROM_RESOURCE_ERR',
      msg: `Model role: ${modelRole}\n${err.type}\n${err.msg}` // no need stack trace since every err.msg got it.
    });
  }

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
      type: 'ACTIVE_MODEL_SETTING_NOT_EXIST',
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
    `
    File: ${basename(module.filename)}
    Function: ${modelService.name}
    Active Models:`,
    activeModels
  );

  // relocate into deep here to prevent display notification when model not running since content func not changed
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

      // Ad-hoc integration with VSCode's Language Model API
      // Its official LLM is GitHub Copilot
      // Since it's built-in, not an outside model
      if (
        modelRole === 'repairation' &&
        activeModels.length === 1 &&
        activeModels[0].name === 'github-copilot-gpt4'
      ) {
        return; // go straight to merge result, which doesn't do anything since there aren't logic for repairation
      }

      // iterate models (detection, localization, repairation)
      activeModels.forEach(async (model) => {
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

        // iterate funcs
        funcs.forEach(async (func) => {
          // only the start of chain - detection needed this
          if (modelRole === 'detection' && func.isRunDelore) {
            logger.debugSuccess(
              `Function: ${func.name} has run through Delore before!`
            );
            return;
          }

          // PREPARE: this should be Repository's responsibility
          func.isRunDelore = true;

          // skip func if it had result before with the same model
          if (
            func[`${modelRole}Results`].find(
              (modelResult) => modelResult.modelName === model.name
            )
          ) {
            logger.debugSuccess(
              `In ${modelRole} service, function has used model ${model.name}.`
            );
            return;
          }

          // skip func if locate, repair but func not run through detection service
          if (
            (modelRole === 'localization' || modelRole === 'repairation') &&
            (func.detectionResults.length === 0 || !func.mergeDetectionResult)
          ) {
            logger.debugSuccess(`Haven't run through detection service.`);
            return;
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
            return;
          }

          // skip func if repair but func not run through localization service
          if (
            modelRole === 'repairation' &&
            (func.localizationResults.length === 0 ||
              !func.mergeLocalizationResult)
          ) {
            logger.debugSuccess(`Haven't run through localization service.`);
            return;
          }

          // skip func if repair, run through localization service but not a single line is vul?
          // later
          // for now, lets just assume that every localization service result in at least one line is vul.

          /* ============================================== */
          /* Input                                          */
          /* ============================================== */

          const defaultParams = model.args; // this can be empty
          const settingParams: string[] = []; // PREPARE: this can be implemented in the future

          const paramObjEither = handleModelInput(modelRole, func);

          if (isLeft(paramObjEither)) {
            const err = unwrapEither(paramObjEither);
            logger.debugError(err);
            return; // nuke the func
          }

          const paramObj = unwrapEither(paramObjEither);
          const paramJSON = safeJsonStringify(paramObj);

          // NOTE: settingParams MUST COME AFTER defaultParams
          const params = [...defaultParams, ...settingParams, paramJSON];

          /* ============================================== */
          /* Execute                                        */
          /* ============================================== */

          // Show notification on VSCode
          progress.report({
            message: `${model.name} - ${func.name}`
          });

          // Set flag
          isModelRun = true;

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
            (modelRole === 'detection' &&
              !isDetectionModelOutput(modelOutput)) ||
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
      type: 'REPOSITORY_UPDATE_ERR',
      msg: `${err.type}\n${err.msg}` // no need stack if return another Either Left
    });
  }

  if (isModelRun) {
    return makeRight('RUN');
  }

  return makeRight('NOT_RUN');
};
