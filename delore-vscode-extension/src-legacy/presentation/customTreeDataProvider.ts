import * as vscode from 'vscode';
import { basename } from 'path';

import * as logger from '../shared/logger';
import {
  EXTENSION_ID,
  MultiRoleModel,
  ModelType,
  ResourceManager,
  SingleRoleModel
} from '../shared/data';

/**
 * Cre: https://github.com/microsoft/vscode-extension-samples/blob/main/tree-view-sample/src/nodeDependencies.ts
 */
export class ModelViewItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly description: string,
    public readonly contextValue: string,
    public readonly command?: vscode.Command
  ) {
    super(label, collapsibleState);
    this.tooltip = `${this.label}`;
  }

  iconPath = '$(library)';
}

/**
 * Cre: https://github.com/microsoft/vscode-extension-samples/blob/main/tree-view-sample/src/nodeDependencies.ts
 */
export abstract class BaseTreeDataProvider<T extends vscode.TreeItem>
  implements vscode.TreeDataProvider<T>
{
  private _onDidChangeTreeData: vscode.EventEmitter<T | undefined | void> =
    new vscode.EventEmitter<T | undefined | void>();
  readonly onDidChangeTreeData: vscode.Event<T | undefined | void> | undefined =
    this._onDidChangeTreeData.event;

  constructor(private readonly _workspaceRoot: string | undefined) {}

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getChildren(element?: T | undefined): vscode.ProviderResult<T[]> {
    throw new Error('getChildren() not implemented!');
  }

  getTreeItem(element: T): vscode.TreeItem | Thenable<vscode.TreeItem> {
    return element;
  }
}

export class ModelProvider extends BaseTreeDataProvider<ModelViewItem> {
  constructor(
    _workspaceRoot: string | undefined,
    private readonly _modelType: ModelType
  ) {
    super(_workspaceRoot);
  }

  getChildren(
    element?: ModelViewItem | undefined
  ): vscode.ProviderResult<ModelViewItem[]> {
    try {
      if (element) {
        // for now, focus on root ViewItem first
        // TODO: we can specify score on each line in a function
        return [];
      } else {
        const models: SingleRoleModel[] =
          ResourceManager.getInstance().getModels(this._modelType);

        return models.map((model) => {
          const preferenceModel = <string>(
            vscode.workspace
              .getConfiguration(EXTENSION_ID)
              .get(`preference.${this._modelType}`) // ON CHANGE: change this line if package.json is changed
          );

          // Debug
          logger.debugInfo(
            basename(module.filename),
            ModelProvider.name,
            'getChildren()',
            '\n',
            this._modelType,
            preferenceModel
          );

          return new ModelViewItem(
            model.name,
            vscode.TreeItemCollapsibleState.None,
            model.name.toLowerCase().includes(preferenceModel.toLowerCase())
              ? 'preference'
              : '',
            this._modelType,
            // TODO: add switch current model logic here
            {
              command: 'delore.switchCurrentModel',
              title: '',
              arguments: [this._modelType, model.name]
            }
          );
        });
      }
    } catch (err) {
      logger.debugError(
        `File: ${basename(module.filename)}\n`,
        `Class: ${ModelProvider.name}\n`,
        `Function: getChildren()\n`,
        err
      );
    }

    return []; // return empty array when there is an Exception
  }
}
