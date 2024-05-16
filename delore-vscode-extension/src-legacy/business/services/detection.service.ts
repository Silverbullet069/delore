import * as vscode from 'vscode';
import * as logger from '../../shared/logger';
import { basename } from 'path';

import { DetectionResult, Func } from '../../legacy/model/func.model';
import { BaseRepository } from '../../legacy/repositories/base/repo.base';
import { ModelType, ResourceManager } from '../../shared/data';
import { CurrentActiveModelTracker } from '../../legacy/repositories/currentActiveModelTracker';

export interface IDetectionService {
  detect(...args: any): any;
}

export class DetectionService implements IDetectionService {
  // Dependency Inversion, IoC
  private readonly _repository: BaseRepository<Func>;

  // Dependency Injection
  public constructor(repository: BaseRepository<Func>) {
    this._repository = repository;
  }

  public async detect(extensionPath: string): Promise<DetectionResult> {
    try {
      const funcs: Func[] = await this._repository.findAll();
      const currentActiveDetectionModel =
        CurrentActiveModelTracker.getInstance().get(ModelType.DETECTION);

      if (!preferenceDetectionModelName) {
        throw new Error(
          "VSCode Setting Detection Model's preference is missing!"
        );
      }

      const preferenceDetectionModel = ResourceManager.getInstance()
        .getModels(ModelType.DETECTION)
        .find((model) => model.name === preferenceDetectionModelName);

      if (!preferenceDetectionModel) {
        throw new Error(
          "Something's wrong here! We should be able to retrieve Preference Detection Model here!"
        );
      }

      const absPathToBinary =
        ResourceManager.getInstance().getPathToPythonBinary(extensionPath);
      const absPathToScript =
        extensionPath + preferenceDetectionModel.relPathToModule;
      const absPathToCWD =
        extensionPath + preferenceDetectionModel.relPathToCWD;
      const param = prefer;

      const isVulnerablePromises = funcs.map((func) =>
        executePythonCommand(
          absPathToBinary,
          absPathToScript,
          func.sanitizedContent,
          absPathToCWD
        )
      );

      const isVulnerables = await Promise.all(isVulnerablePromises);

      logger.debugInfo(basename(module.filename), isVulnerables);
    } catch (err) {
      logger.debugError(
        `File: ${basename(module.filename)}\n`,
        `Class: ${DetectionService.name}\n`,
        `Function: detect()\n`,
        err
      );
    }
  }
}
