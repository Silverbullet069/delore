import * as vscode from 'vscode';
import { exit } from 'process';
import path from 'path';

import { InMemoryRepository } from '../repositories/inMemory.repository';
import { FuncState } from '../type/state.type';
import {
  EXTENSION_ID,
  ModelRole,
  SUPPORTED_LANGUAGES,
  ActiveModelSetting
} from '../constants/config';

import * as logger from '../utils/logger';
import { resourceManager } from '../constants/config';
import { isLeft, unwrapEither } from '../utils/either';

/**
 * Cre: https://github.com/microsoft/vscode-extension-samples/blob/main/tree-view-sample/src/nodeDependencies.ts
 */
class CustomTreeItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly contextValue: string,
    public readonly iconPath: vscode.Uri | vscode.ThemeIcon | string,
    public readonly command?: vscode.Command
  ) {
    super(label, collapsibleState);
    this.tooltip = this.label;
  }
}

/**
 * Cre: https://github.com/microsoft/vscode-extension-samples/blob/main/tree-view-sample/src/nodeDependencies.ts
 */
abstract class GeneralTreeDataProvider
  implements vscode.TreeDataProvider<CustomTreeItem>
{
  private _onDidChangeTreeData: vscode.EventEmitter<
    CustomTreeItem | undefined | void
  > = new vscode.EventEmitter<CustomTreeItem | undefined | void>();
  readonly onDidChangeTreeData:
    | vscode.Event<CustomTreeItem | undefined | void>
    | undefined = this._onDidChangeTreeData.event;

  refresh(): void {
    this._onDidChangeTreeData.fire(); // that's why you need 'void' upthere
  }

  getTreeItem(
    element: CustomTreeItem
  ): vscode.TreeItem | Thenable<vscode.TreeItem> {
    return element;
  }

  getChildren(
    element?: CustomTreeItem
  ): vscode.ProviderResult<CustomTreeItem[]> {
    throw new Error('getChildren() not implemented!'); // Nuke the app
  }
}

/* ====================================================== */
/* ModelView                                              */
/* ====================================================== */

export class ModelTreeDataProvider extends GeneralTreeDataProvider {
  private _modelRole: ModelRole;
  private _extensionPath: string;

  constructor(modelRole: ModelRole, extensionPath: string) {
    super();
    this._modelRole = modelRole;
    this._extensionPath = extensionPath;
  }

  getChildren(
    element?: CustomTreeItem | undefined
  ): vscode.ProviderResult<CustomTreeItem[]> {
    // no nested info
    if (element) {
      return [];
    }

    const config = vscode.workspace.getConfiguration(EXTENSION_ID);
    const activeOrNotModels = config.get<ActiveModelSetting[]>(
      `${this._modelRole}.active`
    );

    if (!activeOrNotModels) {
      logger.debugError(`${this._modelRole}.active setting not existed`);
      return [];
    }

    const activeModels = activeOrNotModels.filter((model) => model.isActive);
    return activeModels.map((model) => {
      const relPathToIconEither = resourceManager.getRelPathToModelIcon(
        model.name
      );

      // error handling
      if (isLeft(relPathToIconEither)) {
        const errType = unwrapEither(relPathToIconEither);
        logger.debugError(errType);

        return new CustomTreeItem(
          'Model not found',
          vscode.TreeItemCollapsibleState.None,
          'Model not found',
          new vscode.ThemeIcon('error'),
          {
            command: '',
            title: '',
            arguments: []
          }
        );
      }

      const relPathToIcon = unwrapEither(relPathToIconEither);
      const iconPath =
        relPathToIcon === ''
          ? new vscode.ThemeIcon('gear')
          : vscode.Uri.file(path.join(this._extensionPath, relPathToIcon));

      logger.debugInfo(iconPath);

      return new CustomTreeItem(
        model.name,
        vscode.TreeItemCollapsibleState.None,
        model.name,
        iconPath,
        {
          command: '',
          title: '',
          arguments: []
        }
      );
    });
  }
}

/* ====================================================== */
/* Custom Outline View                                    */
/* ====================================================== */

export class OutlineTreeDataProvider extends GeneralTreeDataProvider {
  private _extensionPath: string;

  constructor(extensionPath: string) {
    super();
    this._extensionPath = extensionPath;
  }

  getChildren(
    element?: CustomTreeItem | undefined
  ): vscode.ProviderResult<CustomTreeItem[]> {
    if (element) {
      return []; // no nested info
    }

    // NOTE: This is only for initialization
    // NOTE: To update children data, use onDidChangeTreeData
    const editor = vscode.window.activeTextEditor;
    if (
      !editor ||
      (editor &&
        !SUPPORTED_LANGUAGES.includes(path.extname(editor.document.uri.fsPath)))
    ) {
      return []; // if not editor or editor but not supported languages, do not populate data
    }

    const editorFsPath = editor.document.uri.fsPath;

    const funcsEither =
      InMemoryRepository.getInstance().getFuncsInOneEditor(editorFsPath);

    const tempEither =
      InMemoryRepository.getInstance().getTempInOneEditor(editorFsPath);

    if (isLeft(funcsEither)) {
      const err = unwrapEither(funcsEither);
      logger.debugError(err.type, '\n', err.msg);
      return;
    }

    if (isLeft(tempEither)) {
      const err = unwrapEither(tempEither);
      logger.debugError(err.type, '\n', err.msg);
      return;
    }

    const funcs = unwrapEither(funcsEither);
    const temp = unwrapEither(tempEither);

    return funcs.map((func) => {
      const start = new vscode.Position(
        func.lines[0].numOnEditor,
        func.lines[0].startCharOnEditor
      );
      const end = new vscode.Position(
        func.lines[0].numOnEditor,
        func.lines[0].endCharOnEditor
      );
      const range = new vscode.Range(start, end);

      return new CustomTreeItem(
        func.name,
        vscode.TreeItemCollapsibleState.None,
        'function',
        !func.isRunDelore
          ? new vscode.ThemeIcon('question')
          : func.mergeDetectionResult && func.mergeDetectionResult.isVulnerable
            ? new vscode.ThemeIcon('issue-opened')
            : new vscode.ThemeIcon('issue-closed'),
        {
          command: 'delore.revealLine',
          title: 'Scroll to line',
          arguments: [editor, range, temp?.fsPath]
        }
      );
    });
  }
}
