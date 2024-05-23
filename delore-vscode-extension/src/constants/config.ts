import * as fs from 'fs';
import { Either, isStrictNever, makeLeft, makeRight } from '../utils/either';
import { instructionsV2 } from './systemMessage';

// Data as Code, no external .env, .json or Database
// NOTE: 'package.json' and 'data.ts' reflects each other

export const EXTENSION_NAME = 'DeLoRe';
export const EXTENSION_ID = 'delore';
export const SUPPORTED_LANGUAGES = ['.c', '.cpp'];

type ResourceManagerErrorType = 'FILE_NOT_FOUND' | 'DEFAULT_MODEL_NOT_FOUND';

type ResourceManagerError = {
  type: ResourceManagerErrorType;
  msg: string;
};

/* ====================================================== */
/* Role                                                   */
/* ====================================================== */

export type ModelRole = 'detection' | 'localization' | 'repairation';

export const modelRoles: ModelRole[] = [
  'detection',
  'localization',
  'repairation'
];

/* ====================================================== */
/* Model                                                  */
/* ====================================================== */

// Convention: lowercase + hyphen
export type ModelName =
  | 'devign'
  | 'linevd'
  | 'linevul'
  | 'github-copilot-gpt4'
  | 'gpt-4o';

// Convention: everything must have a value
// Empty string has their meaning
type MultiRoleModel = {
  name: ModelName;
  desc: string;
  isGPTModel: boolean;
  relPathToIcon: string;

  // if not gpt model
  relPathToCWD: string;

  // if gpt model
  apiKey: string;

  roles: {
    role: ModelRole; // 'role' rather than 'type', since 1 model can have multiple 'role'

    // if not gpt model (python)
    relPathToScript: string;
    args: string[];

    // if gpt model
  }[];
};

// prettier-ignore
type SingleRoleModel = {
  name: ModelName;
  desc: string;
  role: ModelRole;
  isGPTModel: boolean;
  relPathToIcon: string;

  // if not gpt model (python)
  relPathToCWD: string;
  relPathToScript: string;
  args: string[];

  // if gpt model
  apiKey: string;
};

// convention: every model with their role must match the one in VSCode's settings
const defaultModels: MultiRoleModel[] = [
  {
    name: 'devign',
    desc: 'https://github.com/saikat107/Devign',
    isGPTModel: false,
    apiKey: '',
    relPathToIcon: '', // use default icon
    relPathToCWD: '/python/ai_models/devign',
    roles: [
      {
        role: 'detection',
        relPathToScript: '/python/ai_models/devign/detect.py',
        args: []
      }
    ]
  } satisfies MultiRoleModel,
  {
    name: 'linevul',
    desc: 'https://github.com/awsm-research/LineVul',
    isGPTModel: false,
    apiKey: '',
    relPathToIcon: '/asset/linevul_logo.png',
    relPathToCWD: '/python/ai_models/linevul/linevul',
    roles: [
      {
        role: 'detection',
        relPathToScript: '/python/ai_models/linevul/linevul/linevul_main.py',
        // prettier-ignore
        args: [
          // ! NOTE: rel paths only valid if relPathToScript is specified
          // default arguments
          "--model_name", "12heads_linevul_model.bin",
          "--output_dir", "./saved_models",
          "--model_type", "roberta",
          "--tokenizer_name", "microsoft/codebert-base",
          "--model_name_or_path", "microsoft/codebert-base",
          "--block_size", "512",
          "--eval_batch_size", "512",

          // additional arguments
          "--do_use",
          "--function_level", 
          "--input_json"
        ]
      },
      {
        role: 'localization',
        relPathToScript: '/python/ai_models/linevul/linevul/linevul_main.py',
        // prettier-ignore
        args: [
          // default arguments
          "--model_name", "12heads_linevul_model.bin",
          "--output_dir", "./saved_models",
          "--model_type", "roberta",
          "--tokenizer_name", "microsoft/codebert-base",
          "--model_name_or_path", "microsoft/codebert-base",
          "--block_size", "512",
          "--eval_batch_size", "512",

          // additional arguments
          "--do_use",
          "--line_level", 
          "--input_json"
        ]
      }
    ]
  } satisfies MultiRoleModel,
  {
    name: 'linevd',
    desc: 'https://github.com/davidhin/linevd',
    isGPTModel: false,
    apiKey: '',
    relPathToIcon: '', // default icon
    relPathToCWD: '/python/ai_models/linevd',
    roles: [
      {
        role: 'detection',
        relPathToScript: '/python/ai_models/linevd/main.py',
        args: ['--function_level', '--input_json']
      },
      {
        role: 'localization',
        relPathToScript: '/python/ai_models/linevd/main.py',
        args: ['--line_level', '--input_json']
      }
    ]
  } satisfies MultiRoleModel,
  {
    name: 'github-copilot-gpt4',
    desc: 'https://github.com/features/copilot',
    isGPTModel: true,
    apiKey: '',
    relPathToIcon: '/asset/github_copilot_logo.png',
    relPathToCWD: '',
    roles: [
      {
        role: 'repairation',
        relPathToScript: '',
        args: []
      }
    ]
  } satisfies MultiRoleModel,
  {
    name: 'gpt-4o',
    desc: 'https://openai.com/index/hello-gpt-4o/',
    isGPTModel: true,
    apiKey: process.env['OPENAI_API_KEY'] || '',
    relPathToIcon: '/asset/ChatGPT_logo.svg',
    relPathToCWD: '',
    roles: [
      {
        role: 'repairation',
        relPathToScript: '',
        args: []
      }
    ]
  } satisfies MultiRoleModel
] as const; // immutable object
[] as const;

export const resourceManager = {
  /* ========================================== */
  /* ENVIRONMENT                                */
  /* ========================================== */

  getPathToPipBinary(rootDir: string): Either<ResourceManagerError, string> {
    const path = rootDir + '/python/virtual_envs/py-delore/bin/pip';

    if (!fs.existsSync(path)) {
      return makeLeft({
        type: 'FILE_NOT_FOUND',
        msg: `Check again ${path} in ${this.getPathToPipBinary.name} function.\n${new Error().stack}`
      });
    }

    return makeRight(path);
  },

  getPathToPythonBinary(rootDir: string): Either<ResourceManagerError, string> {
    const path = rootDir + '/python/virtual_envs/py-delore/bin/python';

    if (!fs.existsSync(path)) {
      return makeLeft({
        type: 'FILE_NOT_FOUND',
        msg: `Check again ${path} in ${this.getPathToPythonBinary.name} function.\n${new Error().stack}`
      });
    }

    return makeRight(path);
  },

  /* ========================================== */
  /* HELPER SCRIPT SECTION                      */
  /* ========================================== */

  getAbsToHelperDir(rootDir: string): Either<ResourceManagerError, string> {
    const path = rootDir + '/python/helper_scripts';

    if (!fs.existsSync(path)) {
      return makeLeft({
        type: 'FILE_NOT_FOUND',
        msg: `Check again ${path} in ${this.getAbsToHelperDir.name} function.\n${new Error().stack}`
      });
    }

    return makeRight(path);
  },

  getAbsPathToIsFunctionScript(
    rootDir: string
  ): Either<ResourceManagerError, string> {
    const path = this.getAbsToHelperDir(rootDir) + '/is_function.py';

    if (!fs.existsSync(path)) {
      return makeLeft({
        type: 'FILE_NOT_FOUND',
        msg: `Check again ${path} in ${this.getAbsPathToIsFunctionScript.name} function.\n${new Error().stack}`
      });
    }

    return makeRight(path);
  },

  getAbsPathToParseFunctionScript(
    rootDir: string
  ): Either<ResourceManagerError, string> {
    const path = this.getAbsToHelperDir(rootDir) + '/parse_function.py';

    if (!fs.existsSync(path)) {
      return makeLeft({
        type: 'FILE_NOT_FOUND',
        msg: `Check again ${path} in ${this.getAbsPathToParseFunctionScript.name} function.\n${new Error().stack}`
      });
    }

    return makeRight(path);
  },

  /* ========================================== */
  /* MODEL DESTRUCTURIZED                       */
  /* ========================================== */

  getRelPathToModelIcon(
    modelName: string
  ): Either<ResourceManagerError, string> {
    const extractedModel = defaultModels.find(
      (model) => model.name === modelName
    );

    if (!extractedModel) {
      return makeLeft({
        type: 'DEFAULT_MODEL_NOT_FOUND',
        msg: `Check again model name: ${modelName}.\n${new Error().stack}`
      });
    }

    return makeRight(extractedModel.relPathToIcon);
  },

  getModelsByRole(
    modelRole: ModelRole
  ): Either<ResourceManagerError, SingleRoleModel[]> {
    const extractedModels = defaultModels
      .map((model) => {
        const role = model.roles.find((role) => role.role === modelRole);
        if (!role) {
          return null;
        }

        return {
          name: model.name,
          desc: model.desc,
          relPathToCWD: model.relPathToCWD,
          relPathToIcon: model.relPathToIcon,
          apiKey: model.apiKey,
          isGPTModel: model.isGPTModel,

          // add property here
          ...role
        } satisfies SingleRoleModel;
      })
      .filter((model) => model !== null) as SingleRoleModel[];

    return makeRight(extractedModels);
  },

  /* ========================================== */
  /* PERSISTENCE                                */
  /* ========================================== */

  getAbsPathToJSON(rootDir: string): Either<ResourceManagerError, string> {
    const path = rootDir + '/persistance/1.json';

    if (!fs.existsSync(path)) {
      return makeLeft({
        type: 'FILE_NOT_FOUND',
        msg: `Check again ${path} in ${this.getAbsPathToJSON.name} function.\n${new Error().stack}`
      });
    }

    return makeRight(path);
  },

  /* ==================================================== */
  /* Security                                             */
  /* ==================================================== */
  getAbsPathToLocalEnv(rootDir: string): Either<ResourceManagerError, string> {
    const path = rootDir + '/env/local.extension.env';

    if (!fs.existsSync(path)) {
      return makeLeft({
        type: 'FILE_NOT_FOUND',
        msg: `Check again ${path} in ${this.getAbsPathToLocalEnv.name} function.\n${new Error().stack}`
      });
    }

    return makeRight(path);
  }
} as const; // immutable
