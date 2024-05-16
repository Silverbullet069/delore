import { Either, makeLeft, makeRight } from './error';
import { existsSync } from 'fs';

// Data as Code, no external .env, .json or Database
// NOTE: 'package.json' and 'data.ts' reflects each other

export const EXTENSION_NAME = 'DeLoRe';
export const EXTENSION_ID = 'delore';

export type ModelRole = 'detection' | 'localization' | 'repairation';
export type ModelName = 'Devign' | 'LineVD' | 'LineVul' | 'GPT4';

type SingleRoleModelData = {
  name: ModelName;
  role: ModelRole;
  relPathToCWD: string;
  relPathToModule: string;
  desc: string;
  args: string[];
};

type MultiRoleModelData = {
  name: ModelName;
  relPathToCWD: string;
  roles: {
    role: ModelRole; // name 'role' rather than 'type', since 1 model can have multiple 'role'
    desc: string;
    relPathToModule: string;
    args: string[];
  }[];
};

const globalModels: MultiRoleModelData[] = [
  {
    name: 'Devign',
    relPathToCWD: '/python/ai_models/devign',
    roles: [
      {
        role: 'detection',
        desc: '',
        relPathToModule: '/python/ai_models/devign/detect.py',
        args: []
      }
    ]
  },
  {
    name: 'LineVul',
    relPathToCWD: '/python/ai_models/linevul',
    roles: [
      {
        role: 'detection',
        desc: '',
        relPathToModule: '/python/ai_models/linevul/linevul/linevul_main.py',
        // prettier-ignore
        args: [
          // ! REMEMBER: these relative paths are only valid if you specified relPathToModule
          "--model_name", "12heads_linevul_model.bin",
          "--output_dir", "./saved_models",
          "--model_type", "roberta",
          "--tokenizer_name", "microsoft/codebert-base",
          "--model_name_or_path", "microsoft/codebert-base",
          "--block_size", "1024",
          "--eval_batch_size", "1024",
          "--function_level", // addition argument
        ]
      },
      {
        role: 'localization',
        desc: '',
        relPathToModule: '/python/ai_models/linevul/linevul/linevul_main.py',
        // prettier-ignore
        args: [
          "--model_name", "12heads_linevul_model.bin",
          "--output_dir", "./saved_models",
          "--model_type", "roberta",
          "--tokenizer_name", "microsoft/codebert-base",
          "--model_name_or_path", "microsoft/codebert-base",
          "--block_size", "1024",
          "--eval_batch_size", "1024",
          "--line_level", // addition argument
        ]
      }
    ]
  },
  // TODO
  {
    name: 'LineVD',
    relPathToCWD: '/python/ai_models/linevd',
    roles: [
      {
        role: 'detection',
        desc: '',
        relPathToModule: '',
        args: []
      },
      {
        role: 'localization',
        desc: '',
        relPathToModule: '',
        args: []
      }
    ]
  },
  // TODO
  {
    name: 'GPT4',
    relPathToCWD: '/python/ai_models/gpt4',
    roles: [
      {
        role: 'detection',
        desc: '',
        relPathToModule: '',
        args: []
      },
      {
        role: 'localization',
        desc: '',
        relPathToModule: '',
        args: []
      },
      {
        role: 'repairation',
        desc: '',
        relPathToModule: '',
        args: []
      }
    ]
  }
] as const; // immutable object

/**
 * Some note:
 *
 * - Change from Singleton to const Object since I prefer FP over OOP in TypeScript
 * - Use suffix "-Manager" to separate it from VSCode Extension API
 */
export const resourceManager = {
  checkIsPathExisted(path: string) {
    if (!existsSync(path))
      throw new Error(`${path} not existed. Please check again!`);
  },

  /* ========================================== */
  /* ENVIRONMENT                                */
  /* ========================================== */

  getPathToPipBinary(rootDir: string): string {
    const path = rootDir + '/python/virtual_envs/py-delore/bin/pip';
    this.checkIsPathExisted(path);
    return path;
  },

  getPathToPythonBinary(rootDir: string): string {
    const path = rootDir + '/python/virtual_envs/py-delore/bin/python';
    this.checkIsPathExisted(path);
    return path;
  },

  /* ========================================== */
  /* HELPER SCRIPT SECTION                      */
  /* ========================================== */

  getAbsToHelperDir(rootDir: string): string {
    const path = rootDir + '/python/helper_scripts';
    this.checkIsPathExisted(path);
    return path;
  },
  getAbsPathToIsFunctionScript(rootDir: string): string {
    const path = this.getAbsToHelperDir(rootDir) + '/is_function.py';
    this.checkIsPathExisted(path);
    return path;
  },
  getAbsPathToParseFunctionScript(rootDir: string): string {
    const path = this.getAbsToHelperDir(rootDir) + '/parse_function.py';
    this.checkIsPathExisted(path);
    return path;
  },

  /* ========================================== */
  /* FLATTEN MODEL                              */
  /* ========================================== */

  getModelsByRole(
    modelRole: ModelRole
  ): Either<'MODEL_NOT_FOUND', SingleRoleModelData[]> {
    const extractedModels: SingleRoleModelData[] = [];

    globalModels.forEach((model) => {
      const role = model.roles.find((role) => role.role === modelRole);

      if (role) {
        extractedModels.push({
          name: model.name,
          relPathToCWD: model.relPathToCWD,
          ...role
        });
      }
    });

    if (extractedModels.length === 0) {
      return makeLeft('MODEL_NOT_FOUND');
    }

    return makeRight(extractedModels);
  },

  /* ========================================== */
  /* PERSISTENCE                                */
  /* ========================================== */

  getAbsPathToJSON(rootDir: string): string {
    const path = rootDir + '/persistance/1.json';
    this.checkIsPathExisted(path);
    return path;
  }
} as const; // immutable
