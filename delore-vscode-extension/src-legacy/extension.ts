import * as vscode from 'vscode';

import { EXTENSION_ID } from './shared/data';
import { deloreCommand } from './commands/delore.command';

import { CCppConfigDocumentSymbolProvider } from './presentation/customDocumentSymbolProvider';
import { ModelProvider } from './presentation/customTreeDataProvider';

export function activate(context: vscode.ExtensionContext) {
  // Debug
  // console.log(context.extensionPath);
  // console.log(context.extensionUri.fsPath); // same as above

  const funcJsonRepository: BaseRepository<Func> = new FuncJsonRepository(
    ResourceManager.getInstance().getAbsPathToCacheFile(context.extensionPath)
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      `${EXTENSION_ID}.test1`,
      parseFunction1Command(context.extensionPath)
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      `${EXTENSION_ID}.test2`,
      parseFunction2Command(funcJsonRepository, context.extensionPath)
    )
  );

  // Add custom outline inside original Outline
  // It's not the behavior that we desire
  // context.subscriptions.push(
  //   vscode.languages.registerDocumentSymbolProvider(
  //     { scheme: 'file', language: 'cpp' },
  //     new CCppConfigDocumentSymbolProvider()
  //   )
  // );
  // context.subscriptions.push(
  //   vscode.languages.registerDocumentSymbolProvider(
  //     { scheme: 'file', language: 'c' },
  //     new CCppConfigDocumentSymbolProvider()
  //   )
  // );

  // Test Tree View
  context.subscriptions.push(
    vscode.commands.registerCommand(
      `${EXTENSION_ID}.testViewTitle`,
      testViewItemContextCommand(context.extensionPath)
    )
  );
  context.subscriptions.push(
    vscode.commands.registerCommand(
      `${EXTENSION_ID}.testViewTitleNavigation`,
      testViewTitleNavigationCommand(context.extensionPath)
    )
  );
  context.subscriptions.push(
    vscode.commands.registerCommand(
      `${EXTENSION_ID}.testViewItemContext`,
      testViewItemContextCommand(context.extensionPath)
    )
  );
  context.subscriptions.push(
    vscode.commands.registerCommand(
      `${EXTENSION_ID}.testViewItemContextInline`,
      testViewItemContextInlineCommand(context.extensionPath)
    )
  );

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(
      'detection-models-view',
      new ModelProvider(context.extensionPath, ModelType.DETECTION)
    )
  );

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(
      'localization-models-view',
      new ModelProvider(context.extensionPath, ModelType.LOCALIZATION)
    )
  );

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(
      'repairation-models-view',
      new ModelProvider(context.extensionPath, ModelType.REPAIRATION)
    )
  );

  // const selector: DocumentFilter[] = [];
  // for (const language of ["c", "cpp"]) {
  //   selector.push({ language, scheme: "file" });
  //   selector.push({ language, scheme: "untitled" });
  // }

  // context.subscriptions.push(
  //   languages.registerCodeActionsProvider(
  //     selector,
  //     new DetectVulnerabilityProvider(),
  //     DetectVulnerabilityProvider.metadata
  //   )
  // );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      `${EXTENSION_ID}.activateDelore`,
      deloreCommand(funcJsonRepository, context.extensionPath)
    )
  );
}

// This method is called when your extension is deactivated
export function deactivate() {}
