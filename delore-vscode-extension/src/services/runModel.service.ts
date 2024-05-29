import * as vscode from 'vscode';

import * as logger from '../utils/logger';
import {
  EXTENSION_ID,
  ModelName,
  ModelRole,
  modelRoles,
  resourceManager,
  ActiveModelSetting,
  instructionsV2
} from '../constants/config';
import {
  Either,
  isLeft,
  isStrictNever,
  makeLeft,
  makeRight,
  unwrapEither
} from '../utils/either';
import {
  DetectionModelInput,
  DetectionModelOutput,
  FuncState,
  LocalizationModelInput,
  LocalizationModelOutput,
  RepairationModelInput,
  RepairationModelLinesOutput,
  RepairationModelOutput
} from '../type/state.type';
import {
  isDetectionModelOutput,
  isLocalizationModelOutput,
  isRepairationModelOutput
} from '../type/state.type';
import { safeJsonParse, safeJsonStringify } from '../utils/typeSafeJson';
import { InMemoryRepository } from '../repositories/inMemory.repository';
import { capitalize } from '../utils/misc';
import { parseJsonOrDefault } from '../utils/parseJsonOrDefault';
import { ServiceHandler } from '../ServiceHandler';

/* ====================================================== */
/* Type                                                   */
/* ====================================================== */

export type ModelServiceErrorType =
  | 'EXTENSION_PATH_NOT_FOUND'
  | 'ROLE_NOT_FOUND'
  | 'EDITOR_NOT_FOUND'
  | 'GET_MODEL_FROM_RESOURCE_MANAGER_ERR'
  | 'ACTIVE_MODEL_SETTING_NOT_EXIST'
  | 'UPDATE_MERGE_RESULT_IN_ONE_EDITOR_ERR'
  | 'GET_FUNCS_IN_ONE_EDITOR_ERR'
  | 'GENERATIVE_MODEL_ERR';

// When I design Error like this, the design must give the caller context about all the outcomes, without resolves to Context Switching
export type ModelServiceError = {
  type: ModelServiceErrorType;
  msg: string;
};

export type ModelServiceSuccessType = boolean;

export type CustomProgress = Partial<{
  message: string;
  increment: number;
}>;

const createModelInput = (
  modelName: ModelName,
  modelRole: ModelRole,
  func: FuncState
): Either<
  'ROLE_NOT_FOUND',
  DetectionModelInput | LocalizationModelInput | RepairationModelInput
> => {
  switch (modelRole) {
    case 'detection':
      return makeRight({
        modelName,
        lines: func.lines.map((line) => line.unprocessedContent)
      });
    case 'localization':
      return makeRight({
        modelName,
        lines: func.lines.map((line) => line.unprocessedContent)
      });
    case 'repairation':
      return makeRight({
        modelName,
        lines: func.lines.map((line) => line.unprocessedContent),
        vulLineNums: func.mergeLocalizationResult?.lines
          .filter((line) => line.isVulnerable)
          .map((line) => line.num)
      });
    default:
      return makeLeft('ROLE_NOT_FOUND');
  }
};

export const runModelService = async (
  extensionPath: string,
  modelRole: ModelRole,
  editor: vscode.TextEditor,
  progress: vscode.Progress<CustomProgress>,
  token: vscode.CancellationToken
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
  let isAtLeastOneModelRun = false;
  const editorFsPath = editor.document.uri.fsPath;

  /* ==================================================== */
  /* Extract Funcs                                        */
  /* ==================================================== */
  makeRight;
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

  const allModelsEither = resourceManager.getModelsByRole(modelRole);

  if (isLeft(allModelsEither)) {
    const err = unwrapEither(allModelsEither);
    return makeLeft({
      type: 'GET_MODEL_FROM_RESOURCE_MANAGER_ERR',
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
    `List active ${modelRole} models: `,
    activeModels.map((model) => `${model.name} `)
  );

  // iterate models (detection, localization, repairation)
  // using for ... of to achieve outer async/await

  for (const [index, model] of activeModels.entries()) {
    // ! Ad-hoc integration with VSCode's Language Model API and OpenAI Node API Library
    // VSCode's Language Model API integrate with GitHub Copilot
    if (
      modelRole === 'repairation' &&
      (model.name === 'github-copilot-gpt4' || model.name === 'gpt-4o')
    ) {
      logger.debugSuccess(`I'm ${model.name}!`);

      // using for ... of to achieve outer async/await
      for (const func of funcs) {
        if (!func.mergeLocalizationResult) {
          logger.debugError(
            `Skipped! There is no merge localization result in func: ${func.name}`
          );
          continue; // do not use 'return'
        }

        if (func.mergeRepairationResult) {
          logger.debugSuccess(
            `Skipped! There is a merge repairation result in func :${func.name} already!`
          );
          continue;
        }

        const funcWithDelimiter = `<function>${func.unprocessedContentFunc}</function>`;

        logger.debugSuccess('func with delim: ', '\n', funcWithDelimiter);

        const linesWithDelimiter = func.mergeLocalizationResult.lines
          .filter((line) => line.isVulnerable)
          .reduce(
            (prev, current) =>
              prev + `<line num="${current.num}">${current.content}</line>\n`,
            ''
          );

        logger.debugSuccess('lines with delim', '\n', linesWithDelimiter);

        const systemMsg = instructionsV2;
        const userMsg = funcWithDelimiter + '\n' + linesWithDelimiter;

        progress.report({
          message: `${model.name} - ${func.name}`
        });

        // set flag
        isAtLeastOneModelRun = true;

        const outputEither =
          model.name === 'github-copilot-gpt4'
            ? await ServiceHandler.instance.promptGitHubCopilotWrapper(
                systemMsg,
                userMsg
              )
            : model.name === 'gpt-4o'
              ? await ServiceHandler.instance.promptOpenAIWrapper(
                  process.env['OPENAI_API_KEY'] || '',
                  systemMsg,
                  userMsg,
                  model.name
                )
              : isStrictNever(model.name as never);

        if (isLeft(outputEither)) {
          const err = unwrapEither(outputEither);
          logger.debugError(
            `Error when prompted in function: ${func.name}\n${err.type}\n${err.msg}` // no stack
          );
          continue;
        }

        const modelLinesOutputJSON = unwrapEither(outputEither);
        const defaultOutput = func.mergeLocalizationResult.lines
          .filter((line) => line.isVulnerable)
          .map(
            (line) =>
              ({
                content: line.content,
                num: line.num,
                isVulnerable: line.isVulnerable,
                cwe: '',
                reason: '',
                fix: ''
              }) satisfies RepairationModelLinesOutput
          );

        const modelLinesOutput = await parseJsonOrDefault(
          modelLinesOutputJSON,
          defaultOutput
        );

        const modelOutput = {
          modelName: model.name,
          lines: modelLinesOutput
        };

        logger.debugSuccess(modelOutput);

        // TODO: fix this later
        // if (!isRepairationModelOutput(modelOutput)) {
        //   logger.debugError(
        //     `Model ${model.name} not give repairation standardized output!${new Error().stack}
        //     Model name: ${model.name}
        //     Model lines: ${modelLinesOutputJSON}`
        //   );
        //   return;
        // }

        const updateEither =
          InMemoryRepository.getInstance().updateModelResultInOneFunc(
            'repairation',
            editorFsPath,
            func.processedContentFuncHash,
            modelOutput as RepairationModelOutput
          );

        if (isLeft(updateEither)) {
          const err = unwrapEither(updateEither);
          logger.debugError(err.type, '\n', err.msg);
          continue;
        }
      }

      continue; // next model
    }

    /* ============================================== */

    const pathEither = resourceManager.getPathToPythonBinary(extensionPath);

    // can't return either since we're inside a forEach(), log is good enough.
    if (isLeft(pathEither)) {
      const err = unwrapEither(pathEither);
      logger.debugError(err.type, '\n', err.msg);
      continue; // next model
    }
    const absPathToBinary = unwrapEither(pathEither);

    /* ============================================== */

    if (model.relPathToScript === '') {
      logger.debugError(
        `Model role: ${modelRole}\nModel ${model.name} don't have relPathToScript. Check constants/config.ts!`
      );
      continue; // next model
    }

    const absPathToScript = extensionPath + model.relPathToScript;

    /* ============================================== */

    if (model.relPathToCWD === '') {
      logger.debugError(
        `Model role: ${modelRole}\nModel ${model.name} don't have relPathToCWD. Check constants/config.ts!`
      );
      continue; // next model
    }

    const absPathToCwd = extensionPath + model.relPathToCWD;

    /* ============================================== */

    // iterate funcs
    for (const func of funcs) {
      //
      if (modelRole === 'detection' && func.isRunDelore) {
        logger.debugSuccess(
          `Function: ${func.name} has run through Delore before!`
        );

        continue; // next func
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
        continue; // next func
      }

      // skip func if locate, repair but func not run through detection service
      if (
        (modelRole === 'localization' || modelRole === 'repairation') &&
        (func.detectionResults.length === 0 || !func.mergeDetectionResult)
      ) {
        logger.debugSuccess(`Haven't run through detection service.`);
        continue; // next func
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
        continue; // next func
      }

      // skip func if repair but func not run through localization service
      if (
        modelRole === 'repairation' &&
        (func.localizationResults.length === 0 || !func.mergeLocalizationResult)
      ) {
        logger.debugSuccess(`Haven't run through localization service.`);
        continue; // next func
      }

      // skip func if repair, run through localization service but not a single line is vul?
      // later
      // for now, lets just assume that every localization service result in at least one line is vul.

      /* ============================================== */
      /* Input                                          */
      /* ============================================== */

      const defaultParams = model.args; // this can be empty
      const settingParams: string[] = []; // PREPARE: this can be implemented in the future

      const paramObjEither = createModelInput(model.name, modelRole, func);

      if (isLeft(paramObjEither)) {
        const err = unwrapEither(paramObjEither);
        logger.debugError(err);
        continue; // next func
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
        message: `${capitalize(model.name)} - ${func.name}`
      });

      const modelOutputEither =
        await ServiceHandler.instance.executePythonCommandAsyncWrapper(
          absPathToBinary,
          absPathToScript,
          params,
          absPathToCwd,
          token
        );

      // const modelOutputEither =
      //   ServiceHandler.instance.executePythonCommandSyncWrapper(
      //     absPathToBinary,
      //     absPathToScript,
      //     params,
      //     absPathToCwd
      //   );

      /* ================================================ */
      /* Output                                           */
      /* ================================================ */

      if (isLeft(modelOutputEither)) {
        const err = unwrapEither(modelOutputEither);
        logger.debugError(err.type, '\n', err.msg);
        continue; // next func
      }

      const modelOutputJSON = unwrapEither(modelOutputEither);
      const modelOutput = await safeJsonParse(modelOutputJSON);

      // debug
      modelRole === 'localization' && logger.debugSuccess(modelOutput);

      /* ================================================ */
      /* Runtime Check                                    */
      /* ================================================ */

      if (
        (modelRole === 'detection' && !isDetectionModelOutput(modelOutput)) ||
        (modelRole === 'localization' &&
          !isLocalizationModelOutput(modelOutput)) ||
        (modelRole === 'repairation' && !isRepairationModelOutput(modelOutput))
      ) {
        logger.debugError(
          `Output JSON: ${modelOutputJSON} does not follows the ${modelRole} standard!`
        );
        continue; // next func
      }

      if (
        modelRole === 'localization' &&
        isLocalizationModelOutput(modelOutput) &&
        modelOutput.lines.length !== func.lines.length
      ) {
        logger.debugError(
          `Output JSON: ${modelOutputJSON} return different lines length compare to the length in editor!`
        );
        continue; // next func
      }

      const updateEither =
        InMemoryRepository.getInstance().updateModelResultInOneFunc(
          modelRole,
          editorFsPath,
          func.processedContentFuncHash,
          modelOutput as
            | DetectionModelOutput
            | LocalizationModelOutput
            | RepairationModelOutput
        );

      if (isLeft(updateEither)) {
        const err = unwrapEither(updateEither);
        logger.debugError(err.type, '\n', err.msg);
        continue; // next func
      }

      // logger.debugSuccess(modelOutput);
      // Set flag
      isAtLeastOneModelRun = true;
      func.isRunDelore = true;
    } // end iterate funcs
  } // end iterate models

  if (!isAtLeastOneModelRun) {
    return makeRight(isAtLeastOneModelRun);
  }

  // merge result
  logger.debugSuccess(`Prepare to merge ${modelRole} results.`);

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

  return makeRight(isAtLeastOneModelRun);
};
