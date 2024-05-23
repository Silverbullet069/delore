import * as logger from '../utils/logger';
import { safeJsonParse } from './typeSafeJson';

export const parseJsonOrDefault = async (json: string, defaultValue: any) => {
  try {
    return await safeJsonParse(json);
  } catch (err) {
    logger.debugError(err);
    return defaultValue;
  }
};
