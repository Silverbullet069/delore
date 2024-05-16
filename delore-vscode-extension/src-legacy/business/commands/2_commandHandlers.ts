import {
  window,
  languages,
  commands,
  TextDocument,
  Range,
  Position,
  Selection,
  Diagnostic,
  DiagnosticSeverity,
  DecorationOptions,
  workspace,
} from "vscode";

/* ------------------------------------------------------ */
/*               1 - Test adding Quick Fix                */
/* ------------------------------------------------------ */

const diagnosticCollection = languages.createDiagnosticCollection(
  "delore-quick-fix-ui"
);

export const testProvideDiagnostics = (document: TextDocument): void => {
  const diagnostics = [];

  // Perform diagnostics logic
  // Assume detect an issue at line 2 (also where function names)
  const diagnosticRange = new Range(
    new Position(1, 0),
    new Position(1, Number.MAX_VALUE)
  );
  const diagnosticMessage = "Possible consists of vulnerabilities";
  const diagnosticSeverity = DiagnosticSeverity.Hint;

  const diagnostic = new Diagnostic(
    diagnosticRange,
    diagnosticMessage,
    diagnosticSeverity
  );

  diagnostics.push(diagnostic);
  diagnosticCollection.set(document.uri, diagnostics);
};

// Provide quick fix
// export const testProvideQuickFixes(document: vscode.TextDocument, range: vscode.Range): vscode.CodeAction[] => {

//   const codeAction2 = new vscode.CodeAction('abc')
//   codeAction2.

//   const codeActionTitle = 'Detect Vulnerabilities'
//   const codeActionKind = vscode.CodeActionKind.QuickFix
//   const codeAction = () => {
//     const edit = new vscode.WorkspaceEdit();
//     edit.insert(document.uri, new vscode.Position(range.end.line, range.end.character), ';')
//     return vscode.workspace.applyEdit(edit);
//   }
// }

/* ------------------------------------------------------ */
