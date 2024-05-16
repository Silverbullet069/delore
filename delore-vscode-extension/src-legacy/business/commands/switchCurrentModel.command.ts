import { ModelName, ModelType } from '../utils/data';
import { CurrentActiveModelTracker } from '../../legacy/repositories/currentActiveModelTracker';

export const switchCurrentActiveModelCommand = (
  modelType: ModelType,
  modelName: ModelName
) => {
  CurrentActiveModelTracker.getInstance().update(modelType, modelName);

  // Specify UI change logic here...
};
