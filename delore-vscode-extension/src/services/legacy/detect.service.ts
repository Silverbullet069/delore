import * as vscode from 'vscode';
import * as logger from '../../utils/logger';

import { executePythonCommandSync } from '../../utils/shell';
import {
  DetectionModelInput,
  DetectionModelOutput,
  FuncState
} from '../../model/state.model';
import { EXTENSION_ID, resourceManager } from '../../constants/config';
import { ActiveModelSetting } from '../../model/settings.model';
import { safeJsonParse, safeJsonStringify } from '../../utils/typeSafeJson';
import { isDetectionModelOutput } from '../../model/state.model';
import {
  Either,
  isLeft,
  makeLeft,
  makeRight,
  unwrapEither
} from '../../utils/either';
import { InMemoryRepository } from '../../repositories/inMemory.repository';
import { ModelServiceError } from '../model.service';

export const detectionService = (
  extensionPath: string,
  editorFsPath: string,
  funcs: FuncState[]
): Either<ModelServiceError, 'SUCCESS'> => {
  const allDetectionModelsEither = resourceManager.getModelsByRole('detection');

  // err handle
  if (isLeft(allDetectionModelsEither)) {
    const err = unwrapEither(allDetectionModelsEither);
    return makeLeft({
      type: 'GET_MODEL_FROM_RESOURCE_MANAGER_ERR',
      msg: `Detection. ${err.type}\n${err.msg}` // no need stack trace since it's in every err.msg from a Left Either.
    });
  }

  const allDetectionModels = unwrapEither(allDetectionModelsEither);

  /* ==================================================== */
  /* VSCode Setting                                       */
  /* ==================================================== */

  const config = vscode.workspace.getConfiguration(EXTENSION_ID);
  const allDetectionModelSettings =
    config.get<ActiveModelSetting[]>('detection.active');

  if (!allDetectionModelSettings) {
    return makeLeft({
      type: 'ACTIVE_MODEL_SETTING_NOT_EXIST',
      msg: `${EXTENSION_ID}.detection.active setting not existed!\n${new Error().stack}`
    });
  }

  const activeDetectionModelSettings = allDetectionModelSettings.filter(
    (model) => model.isActive
  );

  const activeDetectionModelNames = activeDetectionModelSettings.map(
    (setting) => setting.name
  );
  const activeDetectionModels = allDetectionModels.filter((model) =>
    // case-insensitive
    activeDetectionModelNames
      .map((name) => name.toLowerCase())
      .includes(model.name.toLowerCase())
  );

  // iterate detection models
  activeDetectionModels.forEach((detectionModel) => {
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

    if (detectionModel.relPathToScript === '') {
      logger.debugError(
        `Model ${detectionModel.name} don't have relPathToScript. Check constants/config.ts!`
      );
      return; // nuke the model
    }

    const absPathToScript = extensionPath + detectionModel.relPathToScript;

    /* ============================================== */

    if (detectionModel.relPathToCWD === '') {
      logger.debugError(
        `Model ${detectionModel.name} don't have relPathToCWD. Check constants/config.ts!`
      );
      return; // nuke the model
    }

    const absPathToCwd = extensionPath + detectionModel.relPathToCWD;

    /* ============================================== */

    // iterate funcs
    funcs.forEach(async (func) => {
      // skip func if it had result before with the same model
      // if (
      //   func.detectionResults.find(
      //     (detectionResult) => detectionResult.modelName === detectionModel.name
      //   )
      // ) {
      //   return;
      // }

      // skip func if had run Delore before
      if (func.isRunDelore) {
        logger.debugSuccess(
          `${func.name} has ran DeLoRe before! Skipped running detection model ${detectionModel.name} on it!`
        );
        return;
      }

      /* ============================================== */
      /* Input                                          */
      /* ============================================== */

      // PREPARE: this should be Repository's responsibility
      func.isRunDelore = true;

      const defaultParams = detectionModel.args; // this can be empty
      const settingParams: string[] = []; // PREPARE: this can be implemented in the future

      const paramObj: DetectionModelInput = {
        modelName: detectionModel.name,
        lines: func.lines.map((line) => line.unprocessedContent)
      };
      const paramJSON = safeJsonStringify(paramObj);

      // NOTE: settingParams MUST COME AFTER defaultParams
      const params = [...defaultParams, ...settingParams, paramJSON];

      const detectionOutputEither = executePythonCommandSync(
        absPathToBinary,
        absPathToScript,
        params,
        absPathToCwd
      );

      /* ================================================ */
      /* Output                                           */
      /* ================================================ */

      if (isLeft(detectionOutputEither)) {
        const err = unwrapEither(detectionOutputEither);
        logger.debugError(err.type, '\n', err.msg);
        return; // nuke the func
      }

      const detectionOutputJSON = unwrapEither(detectionOutputEither);
      const detectionOutput = (await safeJsonParse(
        detectionOutputJSON
      )) as DetectionModelOutput;

      // Check if detectionOutput structure is DetectionModelOutput
      if (!isDetectionModelOutput(detectionOutput)) {
        logger.debugError(
          `The output json: ${detectionOutputJSON} does not follows the detection model output standard! Check again.`
        );
        return; // nuke the func
      }

      const updateEither =
        InMemoryRepository.getInstance().updateModelResultInOneFunc(
          'detection',
          editorFsPath,
          func.processedContentFuncHash,
          detectionOutput
        );

      if (isLeft(updateEither)) {
        const err = unwrapEither(updateEither);
        logger.debugError(err.type, '\n', err.msg);
        return; // nuke the func
      }

      // debug
      logger.debugSuccess(
        `Detection ${unwrapEither(updateEither)}! Model: ${detectionModel.name} - Function: ${func.name} - Output: ${detectionOutput}`
      );
    });

    // NOTE: since some models use intermediary files, running 1 model in multiple process at the same time is impossible
    // PREPARE: can works with other models that used non-intermediary file.
    // const detectionOutputEithers = await Promise.all(promises);
  });

  const updateMergeDetectionEither =
    InMemoryRepository.getInstance().updateMergeResultInOneEditor(
      'detection',
      editorFsPath
    );

  if (isLeft(updateMergeDetectionEither)) {
    const err = unwrapEither(updateMergeDetectionEither);
    return makeLeft({
      type: 'UPDATE_MERGE_RESULT_IN_ONE_EDITOR_ERR',
      msg: `${err.type}\n${err.msg}` // no need stack
    });
  }

  return makeRight('SUCCESS');
};
