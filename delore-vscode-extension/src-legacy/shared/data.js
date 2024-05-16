"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.resourceManager = exports.EXTENSION_ID = exports.EXTENSION_NAME = void 0;
const error_1 = require("./error");
const fs_1 = require("fs");
exports.EXTENSION_NAME = 'DeLoRe';
exports.EXTENSION_ID = 'delore';
const globalModels = [
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
                args: [
                    "--model_name", "12heads_linevul_model.bin",
                    "--output_dir", "./saved_models",
                    "--model_type", "roberta",
                    "--tokenizer_name", "microsoft/codebert-base",
                    "--model_name_or_path", "microsoft/codebert-base",
                    "--block_size", "1024",
                    "--eval_batch_size", "1024",
                    "--function_level",
                ]
            },
            {
                role: 'localization',
                desc: '',
                relPathToModule: '/python/ai_models/linevul/linevul/linevul_main.py',
                args: [
                    "--model_name", "12heads_linevul_model.bin",
                    "--output_dir", "./saved_models",
                    "--model_type", "roberta",
                    "--tokenizer_name", "microsoft/codebert-base",
                    "--model_name_or_path", "microsoft/codebert-base",
                    "--block_size", "1024",
                    "--eval_batch_size", "1024",
                    "--line_level",
                ]
            }
        ]
    },
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
];
exports.resourceManager = {
    checkIsPathExisted(path) {
        if (!(0, fs_1.existsSync)(path))
            throw new Error(`${path} not existed. Please check again!`);
    },
    getPathToPipBinary(rootDir) {
        const path = rootDir + '/python/virtual_envs/py-delore/bin/pip';
        this.checkIsPathExisted(path);
        return path;
    },
    getPathToPythonBinary(rootDir) {
        const path = rootDir + '/python/virtual_envs/py-delore/bin/python';
        this.checkIsPathExisted(path);
        return path;
    },
    getAbsToHelperDir(rootDir) {
        const path = rootDir + '/python/helper_scripts';
        this.checkIsPathExisted(path);
        return path;
    },
    getAbsPathToIsFunctionScript(rootDir) {
        const path = this.getAbsToHelperDir(rootDir) + '/is_function.py';
        this.checkIsPathExisted(path);
        return path;
    },
    getAbsPathToParseFunctionScript(rootDir) {
        const path = this.getAbsToHelperDir(rootDir) + '/parse_function.py';
        this.checkIsPathExisted(path);
        return path;
    },
    getModelsByRole(modelRole) {
        const extractedModels = [];
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
            return (0, error_1.makeLeft)('MODEL_NOT_FOUND');
        }
        return (0, error_1.makeRight)(extractedModels);
    },
    getAbsPathToJSON(rootDir) {
        const path = rootDir + '/persistance/1.json';
        this.checkIsPathExisted(path);
        return path;
    }
};
