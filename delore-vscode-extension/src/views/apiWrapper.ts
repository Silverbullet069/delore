import * as vscode from 'vscode';
import { Either, makeLeft, makeRight } from '../utils/either';

type EditorRetrievalErrorType =
  | 'FILE_NOT_VISIBLE_ERROR'
  | 'FILE_NOT_OPENED'
  | 'NOT_MAIN_EDITOR'
  | 'VSCODE_API_ERROR';

type EditorRetrievalError = {
  type: EditorRetrievalErrorType;
  msg: string;
};

export const isFileOpen = (fileFsPath: string): boolean => {
  return vscode.workspace.textDocuments.some(
    (document) => document.uri.fsPath === fileFsPath
  );
};

export const isFileOpenAndVisible = (fileFsPath: string): boolean => {
  return vscode.window.visibleTextEditors.some(
    (editor) => editor.document.uri.fsPath === fileFsPath
  );
};

// failed
const openFileBeside = async (
  fileFsPath: string,
  activeFsPath: string,
  activeViewColumn: vscode.ViewColumn | undefined
): Promise<Either<EditorRetrievalError, 'SUCCESS'>> => {
  if (!activeViewColumn) {
    return makeLeft({
      type: 'NOT_MAIN_EDITOR',
      msg: `Not main editor, check your editor view column.\n${new Error().stack}`
    });
  }

  try {
    // Show Text
    await vscode.window.showTextDocument(vscode.Uri.file(fileFsPath), {
      viewColumn:
        activeViewColumn === vscode.ViewColumn.Two
          ? vscode.ViewColumn.One
          : activeViewColumn === vscode.ViewColumn.One
            ? vscode.ViewColumn.Two
            : vscode.ViewColumn.Beside, // if neither, open on side
      preview: false,
      preserveFocus: false // false if you want to mark editor read-only
    });

    // mark editor read-only
    await vscode.commands.executeCommand(
      `workbench.action.files.setActiveEditorReadonlyInSession`
    );

    // change active editor back to source code editor
    await vscode.window.showTextDocument(vscode.Uri.file(activeFsPath), {
      viewColumn: activeViewColumn,
      preview: false,
      preserveFocus: false
    });

    return makeRight('SUCCESS');
  } catch (err) {
    return makeLeft({
      type: 'VSCODE_API_ERROR',
      msg: `${err}`
    });
  }
};

// failed attempt
const resetTempVisibility = async (
  fileFsPath: string,
  activeFsPath: string,
  activeViewColumn: vscode.ViewColumn | undefined
): Promise<Either<EditorRetrievalError, 'SUCCESS'>> => {
  if (!activeViewColumn) {
    return makeLeft({
      type: 'NOT_MAIN_EDITOR',
      msg: `Not main editor, check your editor view column.\n${new Error().stack}`
    });
  }

  try {
    // close all file with the same name in all editors
    for await (let editor of vscode.window.visibleTextEditors) {
      if (editor.document.uri.fsPath === fileFsPath) {
        await vscode.window.showTextDocument(editor.document.uri, {
          viewColumn: editor.viewColumn,
          preview: false,
          preserveFocus: false
        });
        await vscode.commands.executeCommand(
          'workbench.action.closeActiveEditor'
        );
      }
    }

    await vscode.window.showTextDocument(vscode.Uri.file(fileFsPath), {
      viewColumn:
        activeViewColumn === vscode.ViewColumn.Two
          ? vscode.ViewColumn.One
          : activeViewColumn === vscode.ViewColumn.One
            ? vscode.ViewColumn.Two
            : vscode.ViewColumn.Beside, // if neither, open on side
      preview: false,
      preserveFocus: false // false if you want to mark editor read-only
    });

    // mark editor read-only
    await vscode.commands.executeCommand(
      `workbench.action.files.setActiveEditorReadonlyInSession`
    );

    // change active editor back to source code editor
    // not possible

    return makeRight('SUCCESS');
  } catch (err) {
    return makeLeft({
      type: 'VSCODE_API_ERROR',
      msg: `${err}`
    });
  }
};

export const getVisibleTextEditor = (
  fileFsPath: string
): Either<EditorRetrievalError, vscode.TextEditor> => {
  const editor = vscode.window.visibleTextEditors.find(
    (editor) => editor.document.uri.fsPath === fileFsPath
  );

  if (!editor) {
    return makeLeft({
      type: 'FILE_NOT_VISIBLE_ERROR',
      msg: `Check again your file fs path: ${fileFsPath}.\n${new Error().stack}`
    });
  }

  return makeRight(editor);
};
