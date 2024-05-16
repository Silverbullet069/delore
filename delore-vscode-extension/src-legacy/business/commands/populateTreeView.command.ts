import { basename } from 'path';

import { AppState } from '../../../src/model/state.model';

export const populdateTreeViewCommand = (
  extensionPath: string,
  state: AppState
) => {
  return () => {
    const editorId = vscode.window.activeTextEditor?.document.uri;
  };
};
