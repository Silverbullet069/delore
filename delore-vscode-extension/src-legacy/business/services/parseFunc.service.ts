import * as vscode from 'vscode';
import * as logger from '../../shared/logger';
import { basename } from 'path';

import { ResourceManager } from '../../shared/data';
import { sanitize } from '../../shared/sanitize';

import {
  DetectionResult,
  Func,
  LocalizationResult,
  RepairationResult
} from '../../legacy/model/func.model';
import { FuncJsonRepository } from '../../legacy/repositories/repo.impl';
import { BaseRepository } from '../../legacy/repositories/base/repo.base';
import * as crypto from 'crypto';

/**
 * Cre: https://github.com/jmbeach/vscode-list-symbols/blob/master/src/SymbolKinds.ts
 */
const symbolKinds = [
  'file',
  'module',
  'namespace',
  'package',
  'class',
  'method',
  'property',
  'field',
  'constructor',
  'enum',
  'interface',
  'function',
  'variable',
  'constant',
  'string',
  'number',
  'boolean',
  'array',
  'object',
  'key',
  'null',
  'enumMember',
  'struct',
  'event',
  'operator',
  'type parameter'
];

export interface IParseFuncService {
  parseFunc(...args: any): any;
}

/**
 * Parse a C/CPP codes into in-memory and cache file
 * Only capture functions
 *
 * Cre: https://github.com/jmbeach/vscode-list-symbols/blob/master/src/extension.ts
 */
export class ParseFuncService implements IParseFuncService {
  // Dependency Inversion, IoC
  private readonly _repository: BaseRepository<Func>;

  // Dependency Injection
  public constructor(repository: BaseRepository<Func>) {
    this._repository = repository;
  }

  public async parseFunc(
    symbols: vscode.DocumentSymbol[],
    document: vscode.TextDocument
  ): Promise<void> {
    for (const symbol of symbols) {
      const symbolKind = symbolKinds[symbol.kind];
      if (symbolKind !== 'function') {
        continue;
      }

      const name = symbol.name;
      const unsanitizedContent = document.getText(symbol.range);
      const sanitizedContent = sanitize(unsanitizedContent);
      const id = crypto
        .createHash('md5')
        .update(sanitizedContent)
        .digest('hex');
      const editorId = document.uri.fsPath;

      const lines = [];
      const startLineNum = symbol.range.start.line;
      const endLineNum = symbol.range.end.line;
      for (let i = startLineNum; i <= endLineNum; ++i) {
        const textLine = document.lineAt(i);

        const num = textLine.lineNumber;
        const content = textLine.text.trim();
        const startChar = textLine.range.start.character;
        const endChar = textLine.range.end.character;

        lines.push({ num, content, startChar, endChar });
      }

      // It's empty, for now
      const detectionResults: DetectionResult[] = [];
      const localizationResults: LocalizationResult[] = [];
      const repairationResults: RepairationResult[] = [];

      const func: Func = {
        id,
        name,
        editorId,
        sanitizedContent,
        lines,
        detectionResults,
        localizationResults,
        repairationResults
      };

      logger.debugSuccess(`ID: ${id}\n`, `Name: ${name}\n`);

      await this._repository.create(func); // ! most important!

      // Next level
      const symbolChildren = symbol.children;
      if (symbolChildren) {
        await this.parseFunc(symbolChildren, document);
      }
    }

    await this._repository.save(); // ! add persistance
  }
}
