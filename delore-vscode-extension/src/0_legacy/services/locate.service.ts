import * as vscode from 'vscode';
import * as logger from '../../utils/logger';

import {
  EXTENSION_ID,
  resourceManager,
  ActiveModelSetting
} from '../../constants/config';
import {
  Either,
  isLeft,
  makeLeft,
  makeRight,
  unwrapEither
} from '../../utils/either';
import {
  FuncState,
  LocalizationModelInput,
  LocalizationModelOutput
} from '../../type/state.type';
import { isLocalizationModelOutput } from '../../type/state.type';
import { safeJsonParse, safeJsonStringify } from '../../utils/typeSafeJson';
import { executePythonCommandSyncService } from '../../services/shell.service';
import { InMemoryRepository } from '../../repositories/inMemory.repository';
import { ModelServiceError } from '../../services/runModel.service';

// 99% resembles detectionResult
// 1% different: it uses detectionResult
// and it's easier to read, can't believe I forget about separation of concerns

export const localizationService = (
  extensionPath: string,
  editorFsPath: string,
  funcs: FuncState[]
): Either<ModelServiceError, 'SUCCESS'> => {
  const allLocalizationModelsEither =
    resourceManager.getModelsByRole('localization');

  if (isLeft(allLocalizationModelsEither)) {
    const err = unwrapEither(allLocalizationModelsEither);
    return makeLeft({
      type: 'GET_MODEL_FROM_RESOURCE_MANAGER_ERR',
      msg: `Localization \n${err.type}\n${err.msg}` // no need stack trace since every err.msg got it.
    });
  }

  const allLocalizationModels = unwrapEither(allLocalizationModelsEither);

  /* ==================================================== */
  /* VSCode Setting                                       */
  /* ==================================================== */

  const config = vscode.workspace.getConfiguration(EXTENSION_ID);
  const allLocalizationModelSettings =
    config.get<ActiveModelSetting[]>(`localization.active`);

  if (!allLocalizationModelSettings) {
    return makeLeft({
      type: 'ACTIVE_MODEL_SETTING_NOT_EXIST',
      msg: `${EXTENSION_ID}.localization.active setting not existed!\n${new Error().stack}`
    });
  }

  const activeLocalizationModelSettings = allLocalizationModelSettings.filter(
    (model) => model.isActive
  );

  const activeLocalizationModelNames = activeLocalizationModelSettings.map(
    (setting) => setting.name
  );

  const activeLocalizationModels = allLocalizationModels.filter((model) =>
    // case-insensitive
    activeLocalizationModelNames
      .map((name) => name.toLowerCase())
      .includes(model.name.toLowerCase())
  );

  // iterate localization models
  activeLocalizationModels.forEach((localizationModel) => {
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

    if (localizationModel.relPathToScript === '') {
      logger.debugError(
        `Model ${localizationModel.name} don't have relPathToScript. Check constants/config.ts!`
      );
      return; // nuke the model
    }

    const absPathToScript = extensionPath + localizationModel.relPathToScript;

    /* ============================================== */

    if (localizationModel.relPathToCWD === '') {
      logger.debugError(
        `Model ${localizationModel.name} don't have relPathToCWD. Check constants/config.ts!`
      );
      return; // nuke the model
    }

    const absPathToCwd = extensionPath + localizationModel.relPathToCWD;

    /* ============================================== */

    // iterate funcs
    funcs.forEach(async (func) => {
      // skip func if it had result before with the same model
      // if (
      //   func.localizationResults.find(
      //     (localizationResult) =>
      //       localizationResult.modelName === localizationModel.name
      //   )
      // ) {
      //   return;
      // }

      // skip func if this func hasn't been run through detection service
      if (func.detectionResults.length === 0 || !func.mergeDetectionResult) {
        return;
      }

      // skip func if this func has run through detection service but predicted as non-vul
      if (
        func.mergeDetectionResult &&
        !func.mergeDetectionResult.isVulnerable
      ) {
        return;
      }

      /* ============================================== */
      /* Input                                          */
      /* ============================================== */

      // PREPARE: this should be Repository's responsibility
      func.isRunDelore = true;

      const defaultParams = localizationModel.args; // this can be empty
      const settingParams: string[] = []; // PREPARE: this can be implemented in the future

      const paramObj: LocalizationModelInput = {
        modelName: localizationModel.name,
        lines: func.lines.map((line) => line.unprocessedContent)
      };
      const paramJSON = safeJsonStringify(paramObj);

      // NOTE: settingParams MUST COME AFTER defaultParams
      const params = [...defaultParams, ...settingParams, paramJSON];

      /* ============================================== */

      const localizationOutputEither = executePythonCommandSyncService(
        absPathToBinary,
        absPathToScript,
        params,
        absPathToCwd
      );

      /* ================================================ */
      /* Output                                           */
      /* ================================================ */

      if (isLeft(localizationOutputEither)) {
        const err = unwrapEither(localizationOutputEither);
        logger.debugError(err.type, '\n', err.msg);
        return; // nuke the func
      }

      const localizationOutputJSON = unwrapEither(localizationOutputEither);
      const localizationOutput = (await safeJsonParse(
        localizationOutputJSON
      )) as LocalizationModelOutput;

      // Check if localizationOutput structure is LocalizationModelOutput
      if (!isLocalizationModelOutput(localizationOutput)) {
        logger.debugError(
          `The output json: ${localizationOutputJSON} does not follows the localization model output standard! Check again.`
        );
        return; // nuke the func
      }

      const updateEither =
        InMemoryRepository.getInstance().updateModelResultInOneFunc(
          'localization',
          editorFsPath,
          func.processedContentFuncHash,
          localizationOutput
        );

      if (isLeft(updateEither)) {
        const err = unwrapEither(updateEither);
        logger.debugError(err.type, '\n', err.msg);
        return; // nuke the func
      }

      // debug
      logger.debugSuccess(localizationOutput);
    });

    // NOTE: since some models use intermediary files, running 1 model in multiple process at the same time is impossible
    // const detectionOutputEithers = await Promise.all(promises);
  });

  const updateMergeLocalizationEither =
    InMemoryRepository.getInstance().updateMergeResultInOneEditor(
      'localization',
      editorFsPath
    );

  if (isLeft(updateMergeLocalizationEither)) {
    const err = unwrapEither(updateMergeLocalizationEither);
    return makeLeft({
      type: 'UPDATE_MERGE_RESULT_IN_ONE_EDITOR_ERR',
      msg: `Localization\n${err.type}\n${err.msg}` // no need stack
    });
  }

  return makeRight('SUCCESS');
};
