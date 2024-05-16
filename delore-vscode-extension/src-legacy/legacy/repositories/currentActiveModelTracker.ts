import * as vscode from 'vscode';
import * as logger from '../../../src/utils/logger';

import { writeFile as writeFileAsync } from 'fs/promises';
import { readFile as readFileAsync } from 'fs/promises';
import { EXTENSION_ID, ModelName, ModelRole } from '../../shared/data';

export class CurrentActiveModelTracker {
  private readonly _currentActiveModelMap: Map<ModelRole, ModelName>;
  private static _instance: CurrentActiveModelTracker;

  public static getInstance(): CurrentActiveModelTracker {
    if (!CurrentActiveModelTracker._instance) {
      CurrentActiveModelTracker._instance = new CurrentActiveModelTracker();
    }
    return CurrentActiveModelTracker._instance;
  }

  private constructor() {
    this._currentActiveModelMap = new Map<ModelRole, ModelName>();

    this._currentActiveModelMap.set(
      'detection',
      <ModelName>(
        vscode.workspace
          .getConfiguration(EXTENSION_ID)
          .get('preference.detection')
      )
    );
    this._currentActiveModelMap.set(
      'localization',
      <ModelName>(
        vscode.workspace
          .getConfiguration(EXTENSION_ID)
          .get('preference.localization')
      )
    );
    this._currentActiveModelMap.set(
      'repairation',
      <ModelName>(
        vscode.workspace
          .getConfiguration(EXTENSION_ID)
          .get('preference.repairation')
      )
    );
  }

  public update(modelType: ModelRole, modelName: ModelName) {
    this._currentActiveModelMap.set(modelType, modelName);
  }

  public get(modelType: ModelRole): ModelName {
    const modelName = this._currentActiveModelMap.get(modelType);

    if (!modelName) {
      throw new Error(
        `ModelName is undefined! Check modelType: ${modelType} argument!`
      );
    }

    return modelName;
  }
}
