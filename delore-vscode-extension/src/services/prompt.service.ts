import * as vscode from 'vscode';
import OpenAI from 'openai';
import { Either, makeLeft, makeRight } from '../utils/either';

import * as logger from '../utils/logger';
export const LLM_COMMAND_ID = 'delore.testGithubCopilotIntegration';
const LANGUAGE_MODEL_ID = 'copilot-gpt-4'; // slower, but stronger

type PromptErrorType =
  | 'API_KEY_NOT_FOUND'
  | 'SYSTEM_MSG_NOT_FOUND'
  | 'USER_MSG_NOT_FOUND'
  | 'OPENAI_API_ERROR'
  | 'VSCODE_LANGUAGE_MODELS_API_SEND_CHAT_REQ_ERR'
  | 'VSCODE_LANGUAGE_MODEL_RESPONSE_STREAM_ERR';

type PromptError = {
  type: PromptErrorType;
  msg: string;
};

export const promptOpenAIService = async (
  apiKey: string,
  systemMsg: string,
  userMsg: string,
  model: string
): Promise<Either<PromptError, string>> => {
  // Validation

  if (!apiKey) {
    return makeLeft({
      type: 'API_KEY_NOT_FOUND',
      msg: `Check again your api key in your .env file.\n${new Error().stack}`
    });
  }

  if (!systemMsg) {
    return makeLeft({
      type: 'SYSTEM_MSG_NOT_FOUND',
      msg: `Check again your editor.\n${new Error().stack}`
    });
  }

  if (!userMsg) {
    return makeLeft({
      type: 'USER_MSG_NOT_FOUND',
      msg: `Check again your user messages.\n${new Error().stack}`
    });
  }

  // PREPARE:
  const openai = new OpenAI({
    apiKey: apiKey
  });

  try {
    const stream = await openai.chat.completions.create({
      model: model,
      messages: [
        {
          role: 'system',
          content: systemMsg
        },
        {
          role: 'user',
          content: userMsg
        }
      ],
      stream: true
    });

    let output = '';
    for await (const chunk of stream) {
      output += chunk.choices[0]?.delta?.content || '';
    }

    logger.debugSuccess('ChatGPT response', '\n', output);
    return makeRight(output);
  } catch (err) {
    const error = <Error>err;

    return makeLeft({
      type: 'OPENAI_API_ERROR',
      msg: `${error.name}\n${error.message}\n${error.stack}`
    });
  }
};

export const promptGithubCopilotService = async (
  systemMsg: string,
  userMsg: string
): Promise<Either<PromptError, string>> => {
  /* ==================================================== */
  /* Validate Parameter                                   */
  /* ==================================================== */
  if (!systemMsg) {
    return makeLeft({
      type: 'SYSTEM_MSG_NOT_FOUND',
      msg: `Check again your editor.\n${new Error().stack}`
    });
  }

  if (!userMsg) {
    return makeLeft({
      type: 'USER_MSG_NOT_FOUND',
      msg: `Check again your user messages.\n${new Error().stack}`
    });
  }

  const messages = [
    new vscode.LanguageModelChatSystemMessage(systemMsg),
    new vscode.LanguageModelChatUserMessage(userMsg)
  ];

  let chatResponse: vscode.LanguageModelChatResponse | undefined;
  try {
    chatResponse = await vscode.lm.sendChatRequest(
      LANGUAGE_MODEL_ID,
      messages,
      {},
      new vscode.CancellationTokenSource().token
    );
  } catch (err) {
    // making the chat request might fail because
    // - user consent not given
    // - model does not exist
    // - quote limits exceeded
    // - other issues

    if (err instanceof vscode.LanguageModelError) {
      return makeLeft({
        type: 'VSCODE_LANGUAGE_MODELS_API_SEND_CHAT_REQ_ERR',
        msg: `Code: ${err.code}
        Name: ${err.name}
        Msg: ${err.message}
        ${err.stack}`
      });
    }

    return makeLeft({
      type: 'VSCODE_LANGUAGE_MODELS_API_SEND_CHAT_REQ_ERR',
      msg: `${err}` // more general error
    });
  }

  // Stream the code into the output as it is coming in from the Language Model
  let output = '';
  try {
    for await (const fragment of chatResponse.stream) {
      output += fragment;
    }
  } catch (err) {
    // async response stream may fail, e.g network interruption or server side error
    const error = <Error>err;
    return makeLeft({
      type: 'VSCODE_LANGUAGE_MODEL_RESPONSE_STREAM_ERR',
      msg: `${error.name}\n${error.message}\n${error.stack}`
    });
  }

  logger.debugInfo('GitHub Copilot response', '\n', output);
  return makeRight(output);
};
