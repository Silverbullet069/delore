import { EXTENSION_ID } from '../utils/data';

import {
  CodeAction,
  CodeActionContext,
  CodeActionKind,
  CodeActionProvider,
  CodeActionProviderMetadata,
  Range,
  Selection,
  TextDocument,
  workspace
} from 'vscode';

// import { detectCommandHandler } from "./commandHandlers";
// import { PackageConfigurationProperties } from "./config";

// export class DetectVulnerabilityProvider implements CodeActionProvider {
//   public static readonly metadata: CodeActionProviderMetadata = {
//     providedCodeActionKinds: [CodeActionKind.QuickFix],
//   };

//   public async provideCodeActions(
//     document: TextDocument,
//     range: Range | Selection,
//     context: CodeActionContext
//   ): Promise<CodeAction[]> {
//     const config = workspace.getConfiguration(
//       PackageConfigurationProperties.Id
//     );
//     const isPreferable = config.get<boolean>(
//       PackageConfigurationProperties.Preferable
//     );

//     const action = new CodeAction(
//       "Generate Detection Result",
//       CodeActionKind.QuickFix
//     );
//     const args: Parameters<typeof detectCommandHandler> = [isVulnerable];
//     action.command = {
//       title: "Generate Detection Result",
//       command: `${EXTENSION_ID}.experiment-detect`,
//       arguments: args,
//     };
//     action.isPreferred = isPreferable;

//     return [action];
//   }
// }
