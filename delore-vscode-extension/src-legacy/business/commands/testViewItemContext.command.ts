import { basename } from 'path';
import * as logger from '../utils/logger';

export const testViewItemContextCommand = (extensionPath: string) => {
  return async (): Promise<void> => {
    try {
      logger.debugSuccess(
        basename(module.filename),
        'Hello from Test View Item Context Command'
      );
    } catch (err) {
      logger.debugError(basename(module.filename), err);
    }
  };
};
