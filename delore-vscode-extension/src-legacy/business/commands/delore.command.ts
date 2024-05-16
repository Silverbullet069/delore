import * as vscode from 'vscode';

import { Func } from '../../legacy/model/func.model';
import { BaseRepository } from '../../legacy/repositories/base/repo.base';
import * as logger from '../utils/logger';
import { EXTENSION_ID, ModelType, ResourceManager } from '../utils/data';

import { executePythonCommand } from '../utils/shell';
import { basename } from 'path';

export const deloreCommand = (
  repository: BaseRepository<Func>,
  extensionPath: string
) => {
  return async () => {
    try {
    } catch (err) {
      logger.debugError(basename(module.filename), err);
    }
  };
};
