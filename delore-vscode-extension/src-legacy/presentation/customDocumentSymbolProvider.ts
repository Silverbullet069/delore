import * as vscode from 'vscode';

export class CCppConfigDocumentSymbolProvider
  implements vscode.DocumentSymbolProvider
{
  provideDocumentSymbols(
    document: vscode.TextDocument,
    token: vscode.CancellationToken
  ): vscode.ProviderResult<
    vscode.SymbolInformation[] | vscode.DocumentSymbol[]
  > {
    let symbols: vscode.DocumentSymbol[] = [];
    for (let i = 0; i < document.lineCount; ++i) {
      const line = document.lineAt(i);
      const lineStr = line.text;

      // TODO: integrate with Python code
      // for now, lets just test with this two return value first
      if (lineStr.startsWith('int') || lineStr.startsWith('char')) {
        let symbol = new vscode.DocumentSymbol(
          `func${i + 1}`,
          'Function',
          vscode.SymbolKind.Function,
          line.range,
          line.range
        );

        symbols.push(symbol);
      }
    }

    return symbols;
  }
}
