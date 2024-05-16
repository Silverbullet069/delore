import { readFile as readFileAsync } from 'fs/promises';
import { writeFile as writeFileAsync } from 'fs/promises';

import { AppState } from '../../../src/model/state.model';
import { safeJsonParse, safeJsonStringify } from '../../shared/typeSafeJson';
import { Either, makeLeft, makeRight } from '../../shared/error';

type LoadError = 'FILE_NOT_FOUND' | 'FILE_CORRUPTED';
type SaveError = '';
const persistState = {
  load: async (pathToJson: string): Promise<Either<LoadError, AppState>> => {
    try {
      const stateJson = await readFileAsync(pathToJson, {
        encoding: 'utf8'
      });
    } catch (err) {
      return makeLeft('FILE_NOT_FOUND');
    }

    const state = <AppState>safeJsonParse(stateJson);

    if (!state) {
      return makeLeft('FILE_NOT_FOUND');
    }

    return makeRight(state);
  },

  save: async (
    pathToJson: string,
    state: AppState
  ): Promise<Either<SaveError, void>> => {
    const stateJson = safeJsonStringify(state);
    await writeFileAsync(pathToJson, stateJson);
  }
};
