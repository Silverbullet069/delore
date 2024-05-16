import { basename } from 'path';
import * as logger from '../utils/logger';

export const testViewItemContextInlineCommand = (extensionPath: string) => {
  return async (): Promise<void> => {
    try {
      logger.debugSuccess(
        basename(module.filename),
        'Hello from Test View Item Context Inline Command'
      );
    } catch (err) {
      logger.debugError(basename(module.filename), err);
    }
  };
};
