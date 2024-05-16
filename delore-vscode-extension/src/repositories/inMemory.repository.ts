import * as logger from '../utils/logger';

import {
  Either,
  isStrictNever,
  makeLeft,
  makeRight,
  unwrapEither
} from '../utils/either';
import {
  DetectionModelOutput,
  EditorState,
  FuncState,
  LocalizationModelOutput,
  RepairationModelOutput,
  TempState
} from '../model/state.model';
import { ModelRole, modelRoles } from '../constants/config';
import {
  mergeDetectionModelOutputs,
  mergeLocalizationModelOutputs,
  mergeRepairationModelOutputs
} from '../services/mergeResult.service';

type RepositoryErrorType =
  | 'EDITOR_FS_PATH_EMPTY'
  | 'TEMP_FS_PATH_EMPTY'
  | 'TEMP_NOT_FOUND'
  | 'EDITOR_NOT_FOUND'
  | 'FUNC_NOT_FOUND'
  | 'HASH_IS_EMPTY'
  | 'MODEL_NOT_FOUND'
  | 'RESULT_EXISTED'
  | 'ROLE_NOT_FOUND'
  | 'DECORATION_EMPTY';

type RepositoryError = {
  type: RepositoryErrorType;
  msg: string;
};

const editorFsPathEmptyErrorTemplate = (): RepositoryError => {
  return {
    type: 'EDITOR_FS_PATH_EMPTY',
    msg: `Check again your editor fs path.\n${new Error().stack}`
  };
};

const editorNotFoundErrorTemplate = (editorFsPath: string): RepositoryError => {
  return {
    type: 'EDITOR_NOT_FOUND',
    msg: `Check again your editor fs path: ${editorFsPath}.\n${new Error().stack}`
  };
};

const tempNotFoundErrorTemplate = (editorFsPath: string): RepositoryError => {
  return {
    type: 'TEMP_NOT_FOUND',
    msg: `Check again editor fs path: ${editorFsPath}.\n${new Error().stack}`
  };
};

const tempFsPathEmptyErrorTemplate = (
  editorFsPath: string
): RepositoryError => {
  return {
    type: 'TEMP_FS_PATH_EMPTY',
    msg: `Check again editor fs path: ${editorFsPath}.\n${new Error().stack}`
  };
};

const decorationEmptyErrorTemplate = (
  editorFsPath: string,
  tempFsPath: string
): RepositoryError => {
  return {
    type: 'DECORATION_EMPTY',
    msg: `Check again your editor fs path: ${editorFsPath} and temp fs path: ${tempFsPath}.\n${new Error().stack}`
  };
};

const roleNotFoundErrorTemplate = (modelRole: ModelRole): RepositoryError => {
  return {
    type: 'ROLE_NOT_FOUND',
    msg: `Check model role: ${modelRole}.\n${new Error().stack}`
  };
};

const funcNotFoundErrorTemplate = (funcHash: string): RepositoryError => {
  return {
    type: 'FUNC_NOT_FOUND',
    msg: `Check func hash: ${funcHash}.\n${new Error().stack}`
  };
};

const resultExistedErrorTemplate = (
  modelRole: ModelRole,
  modelOutput:
    | DetectionModelOutput
    | LocalizationModelOutput
    | RepairationModelOutput
): RepositoryError => {
  return {
    type: 'RESULT_EXISTED',
    msg: `Model role: ${modelRole}\nResult: ${modelOutput.modelName}.\n${new Error().stack}`
  };
};

export class InMemoryRepository {
  private static _instance: InMemoryRepository;
  private _appState: EditorState[];

  private constructor() {
    this._appState = [];
  }

  public static getInstance(): InMemoryRepository {
    if (!InMemoryRepository._instance) {
      InMemoryRepository._instance = new InMemoryRepository();
    }
    return InMemoryRepository._instance;
  }

  public getFuncsInOneEditor(
    editorFsPath: string
  ): Either<RepositoryError, FuncState[]> {
    if (!editorFsPath) {
      return makeLeft(editorFsPathEmptyErrorTemplate());
    }

    const editorState = this._appState.find(
      (editorState) => editorState.editorFsPath === editorFsPath
    );

    if (!editorState) {
      return makeRight([]);
    }

    return makeRight(editorState.funcs);
  }

  public getTempInOneEditor(
    editorFsPath: string
  ): Either<RepositoryError, TempState | null> {
    if (!editorFsPath) {
      return makeLeft(editorFsPathEmptyErrorTemplate());
    }

    const editorState = this._appState.find(
      (editorState) => editorState.editorFsPath === editorFsPath
    );

    if (!editorState) {
      return makeRight(null); // outline tree data provider access this and might not have value
    }

    return makeRight(editorState.temp || null);
  }

  public updateTempInOneEditor(
    editorFsPath: string,
    newTempState: TempState
  ): Either<RepositoryError, 'SUCCESS'> {
    if (!editorFsPath) {
      return makeLeft(editorFsPathEmptyErrorTemplate());
    }

    if (!newTempState) {
      return makeLeft(tempNotFoundErrorTemplate(editorFsPath));
    }

    if (!newTempState.fsPath) {
      return makeLeft(tempFsPathEmptyErrorTemplate(editorFsPath));
    }

    if (!newTempState.vulDecoration) {
      return makeLeft(
        decorationEmptyErrorTemplate(editorFsPath, newTempState.fsPath)
      );
    }

    const editorState = this._appState.find(
      (editorState) => editorState.editorFsPath === editorFsPath
    );

    if (!editorState) {
      return makeLeft(editorNotFoundErrorTemplate(editorFsPath));
    }

    const tempState = editorState.temp;

    // new temp state
    if (!tempState) {
      editorState.temp = newTempState;
      return makeRight('SUCCESS');
    }

    // modified temp state
    tempState.fsPath = newTempState.fsPath;
    tempState.vulDecoration.dispose();
    tempState.vulDecoration = newTempState.vulDecoration;

    return makeRight('SUCCESS');
  }

  public updateEditorFsPath(
    oldEditorFsPath: string,
    newEditorFsPath: string
  ): Either<RepositoryError, 'SUCCESS'> {
    if (!oldEditorFsPath) {
      return makeLeft(editorFsPathEmptyErrorTemplate());
    }

    if (!newEditorFsPath) {
      return makeLeft(editorFsPathEmptyErrorTemplate());
    }

    const editorState = this._appState.find(
      (editorState) => editorState.editorFsPath === oldEditorFsPath
    );

    if (!editorState) {
      return makeLeft(editorNotFoundErrorTemplate(oldEditorFsPath));
    }

    editorState.editorFsPath = newEditorFsPath;
    return makeRight('SUCCESS');
  }

  public updateModelResultInOneFunc(
    modelRole: ModelRole,
    editorFsPath: string,
    processedContentHash: string,
    modelOutput:
      | DetectionModelOutput
      | LocalizationModelOutput
      | RepairationModelOutput
  ): Either<RepositoryError, 'SUCCESS'> {
    if (!modelRoles.includes(modelRole)) {
      return makeLeft(roleNotFoundErrorTemplate(modelRole));
    }

    if (!editorFsPath) {
      return makeLeft(editorFsPathEmptyErrorTemplate());
    }

    const editorState = this._appState.find(
      (editorState) => editorState.editorFsPath === editorFsPath
    );

    if (!editorState) {
      return makeLeft(editorNotFoundErrorTemplate(editorFsPath));
    }

    // extract func via hash
    const func = editorState.funcs.find(
      (func) => func.processedContentHash === processedContentHash
    );

    if (!func) {
      return makeLeft(funcNotFoundErrorTemplate(processedContentHash));
    }

    const modelResult = func[`${modelRole}Results`]?.find(
      (modelResult) => modelResult.modelName === modelOutput.modelName
    );

    // first time
    if (!modelResult) {
      if (modelRole === 'detection') {
        func.detectionResults.push(modelOutput as DetectionModelOutput);
        return makeRight('SUCCESS');
      }

      if (modelRole === 'localization') {
        func.localizationResults.push(modelOutput as LocalizationModelOutput);
        return makeRight('SUCCESS');
      }

      if (modelRole === 'repairation') {
        func.repairationResults.push(modelOutput as RepairationModelOutput);
        logger.debugSuccess('push repairation result success');
        return makeRight('SUCCESS');
      }
    }

    // replace? impossible
    return makeLeft(resultExistedErrorTemplate(modelRole, modelOutput));
  }

  public updateMergeResultInOneEditor(
    modelRole: ModelRole,
    editorFsPath: string
  ): Either<RepositoryError, 'SUCCESS'> {
    /* ================================================== */
    /* Parameters Validation                              */
    /* ================================================== */

    if (!modelRoles.includes(modelRole)) {
      return makeLeft(roleNotFoundErrorTemplate(modelRole));
    }

    if (!editorFsPath) {
      return makeLeft(editorFsPathEmptyErrorTemplate());
    }

    /* ================================================== */
    /* Main Logic                                         */
    /* ================================================== */

    const editorState = this._appState.find(
      (editorState) => editorState.editorFsPath === editorFsPath
    );

    if (!editorState) {
      return makeLeft(editorNotFoundErrorTemplate(editorFsPath));
    }

    const funcs = editorState.funcs;

    funcs.forEach((func) => {
      // skip
      if (func[`${modelRole}Results`]?.length === 0) {
        return;
      }

      switch (modelRole) {
        case 'detection':
          func.mergeDetectionResult = mergeDetectionModelOutputs(
            func.detectionResults
          );

          // debug
          logger.debugSuccess(
            'Merge Detection Result',
            '\n',
            func.mergeDetectionResult
          );

          break;
        case 'localization':
          func.mergeLocalizationResult = mergeLocalizationModelOutputs(
            func.localizationResults
          );

          // debug
          logger.debugSuccess(
            'Merge Localization Result',
            '\n',
            func.mergeLocalizationResult
          );

          break;
        case 'repairation':
          func.mergeRepairationResult = mergeRepairationModelOutputs(
            func.repairationResults
          );

          // debug
          logger.debugSuccess(
            'Merge Repairation Result',
            '\n',
            func.mergeRepairationResult
          );

          break;
        default:
          isStrictNever(modelRole);
      }
    });

    return makeRight('SUCCESS');
  }

  public updateAllFuncsInOneEditor(
    editorFsPath: string,
    newFuncs: FuncState[]
  ): Either<RepositoryError, 'SUCCESS'> {
    if (!editorFsPath) {
      return makeLeft(editorFsPathEmptyErrorTemplate());
    }

    const editorState = this._appState.find(
      (editorState) => editorState.editorFsPath === editorFsPath
    );

    // new editor
    if (!editorState) {
      this._appState.push({
        editorFsPath: editorFsPath,
        funcs: newFuncs
        // don't have tempState at the beginning
      });
      return makeRight('SUCCESS');
    }

    // optimization + preserve model results for existing funcs
    editorState.funcs = newFuncs.map((newFunc) => {
      const similarFunc = editorState.funcs.find(
        (oldFunc) =>
          oldFunc.processedContentHash === newFunc.processedContentHash
      );

      // new or modified funcs
      if (!similarFunc) {
        return newFunc;
      }

      // existing funcs, range might change
      return {
        ...newFunc,

        // Persists properties
        isRunDelore: similarFunc.isRunDelore, // persists this since content not changed

        detectionResults: similarFunc.detectionResults,
        localizationResults: similarFunc.localizationResults,
        repairationResults: similarFunc.repairationResults,

        mergeDetectionResult: similarFunc.mergeDetectionResult,
        mergeLocalizationResult: similarFunc.mergeLocalizationResult,
        mergeRepairationResult: similarFunc.mergeRepairationResult
      };
    });

    return makeRight('SUCCESS');
  }

  // updateOneDetectionResultInOneFunc: (
  //   state: EditorState[],
  //   editorUrl: string,
  //   funcHash: string,
  //   result: DetectionResult
  // ) => EditorState[];

  // updateAllDetectionResultsInOneFunc: (
  //   state: EditorState[],
  //   editorUrl: string,
  //   funcHash: string,
  //   results: DetectionResult[]
  // ) => EditorState[];

  // updateAllDetectionResultsInAllFunc: (
  //   state: EditorState[],
  //   editorUrl: string,
  //   funcHashes: string[],
  //   results: DetectionResult[]
  // ) => EditorState[];
}
