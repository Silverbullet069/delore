import * as vscode from 'vscode';

import { ModelRole } from './constants/config';
import { CustomProgress, runModelService } from './services/runModel.service';
import {
  mergeDetectionModelOutputs,
  mergeLocalizationModelOutputs,
  mergeRepairationModelOutputs
} from './services/mergeResult.service';
import {
  DetectionModelOutput,
  LocalizationModelOutput,
  RepairationModelOutput
} from './type/state.type';
import {
  promptOpenAIService,
  promptGithubCopilotService
} from './services/prompt.service';
import {
  executePythonCommandAsyncService,
  executePythonCommandSyncService
} from './services/shell.service';

export class ServiceHandler {
  /* ==================================================== */
  /* Design Pattern: Singleton                            */
  /* ==================================================== */

  private static _instance: ServiceHandler;

  private constructor() {}

  public static get instance(): ServiceHandler {
    if (!ServiceHandler._instance) {
      ServiceHandler._instance = new ServiceHandler();
    }
    return ServiceHandler._instance;
  }

  /* ==================================================== */
  /* List of services                                     */
  /* ==================================================== */

  public runModelServiceWrapper(
    extensionPath: string,
    modelRole: ModelRole,
    editor: vscode.TextEditor,
    progress: vscode.Progress<CustomProgress>,
    token: vscode.CancellationToken
  ) {
    return runModelService(extensionPath, modelRole, editor, progress, token);
  }

  public mergeDetectionModelOutputsWrapper(
    detectionResults: DetectionModelOutput[]
  ): DetectionModelOutput {
    return mergeDetectionModelOutputs(detectionResults);
  }

  public mergeLocalizationModelOutputsWrapper(
    localizationResults: LocalizationModelOutput[]
  ): LocalizationModelOutput {
    return mergeLocalizationModelOutputs(localizationResults);
  }

  public mergeRepairationModelOutputsWrapper(
    repairationResults: RepairationModelOutput[]
  ): RepairationModelOutput {
    return mergeRepairationModelOutputs(repairationResults);
  }

  public executePythonCommandSyncWrapper(
    absPathToBinary: string,
    absPathToScript: string,
    params: string[],
    absPathToCwd: string
  ) {
    return executePythonCommandSyncService(
      absPathToBinary,
      absPathToScript,
      params,
      absPathToCwd
    );
  }

  public executePythonCommandAsyncWrapper(
    absPathToBinary: string,
    absPathToScript: string,
    params: string[],
    absPathToCwd: string,
    token: vscode.CancellationToken
  ) {
    return executePythonCommandAsyncService(
      absPathToBinary,
      absPathToScript,
      params,
      absPathToCwd,
      token
    );
  }

  public promptOpenAIWrapper(
    apiKey: string,
    systemMsg: string,
    userMsg: string,
    model: string
  ) {
    return promptOpenAIService(apiKey, systemMsg, userMsg, model);
  }

  public promptGitHubCopilotWrapper(systemMsg: string, userMsg: string) {
    return promptGithubCopilotService(systemMsg, userMsg);
  }
}
