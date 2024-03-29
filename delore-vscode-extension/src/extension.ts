import { commands, ExtensionContext, window } from "vscode";
import { EXTENSION_ID } from "./data/constant";
import { test, detect, locate, repair, delore, test2 } from "./logic/command";

export function activate(context: ExtensionContext) {
  context.subscriptions.push(
    commands.registerCommand(`${EXTENSION_ID}.hello-world`, () => {
      window.showInformationMessage("Hello World");
    })
  );

  context.subscriptions.push(
    commands.registerCommand(`${EXTENSION_ID}.test`, test)
  );

  context.subscriptions.push(
    commands.registerCommand(`${EXTENSION_ID}.test2`, test2)
  );

  context.subscriptions.push(
    commands.registerCommand(`${EXTENSION_ID}.detect`, detect)
  );

  context.subscriptions.push(
    commands.registerCommand(`${EXTENSION_ID}.locate`, locate)
  );

  context.subscriptions.push(
    commands.registerCommand(`${EXTENSION_ID}.repair`, repair)
  );

  context.subscriptions.push(
    commands.registerCommand(`${EXTENSION_ID}.delore`, delore)
  );
}

// This method is called when your extension is deactivated
export function deactivate() {}
