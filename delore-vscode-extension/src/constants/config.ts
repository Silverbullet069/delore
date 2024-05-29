import * as fs from 'fs';
import { Either, isStrictNever, makeLeft, makeRight } from '../utils/either';

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

// package.json "delore.[detection|localization|repairation].active"
export type ActiveModelSetting = {
  name: string;
  isActive: boolean;
};

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

/* ====================================================== */
/* Github Copilot Instruction                             */
/* ====================================================== */

// https://towardsdatascience.com/detecting-insecure-code-with-llms-8b8ad923dd98
// Zero-shot template: role + code delimiter + output json format: https://arxiv.org/abs/2308.14434
// Think step-by-step : https://arxiv.org/abs/2205.11916
// Upgrade with few-shots: include a few successful code-answer examples before asking LLM https://arxiv.org/abs/2005.14165
// Upgrade with KNN Few-shots with Code fix: Include request for a fixed version if a CWE is found. Prompting both CWE detection + fix together bring "virtuous cycle" and force the LLM to "self-audit", think more deeply about the steps to accurately identify vulnerabilities (chain-of-thought prompting) https://arxiv.org/abs/2308.10345

export const instructionsV2: string = `
You are a brilliant software security expert.
Your input: a vulnerable C/C++ function delimited by XML tag <function></function> and a list of lines whose vulnerability potential in the function are the highest. Each line delimited by XML tag <line num=></line>, "num" is the zero-based line number of that line in function.

Your response: JSON format, with the following data
{
  "content": string (you write the line content specified in input)
  "num": number (you write the "num" attribute in <line></line> XML tag here)
  "isVulnerable": boolean (you determine the vulnerability status. If the line really contains any CWE security vulnerailities, you write "True". If the line does not contain any vulnerabilities, you write "False".)
  "cwe": string (you determine the vulnerability number found)
  "reason": string (you write the name of that vulnerability)
  "fix": string (you rewrite the line with same functionality and vulnerability-free)
}
NOTE: "cwe", "reason" and "fix" is an empty string if "isVulnerable" is "false".

Here are 2 examples that are out of context with future prompt:

Example 1 input: <function>
void nothing() {
  return;
}
</function>

<line num="0">void nothing() {</line>
<line num="1">return;</line>
<line num="2">}</line>

Example 1 response: [
  {
    "content": "void nothing() {",
    "num": 0,
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "return;",
    "num": 1,
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "}",
    "num": 2,
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  }
]

Example 2 Input: <function>
  PHP_FUNCTION(imageconvolution)
{
zval *SIM, *hash_matrix;
zval **var = NULL, **var2 = NULL;
gdImagePtr im_src = NULL;
double div, offset;
int nelem, i, j, res;
float matrix[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};

if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "radd", &SIM, &hash_matrix, &div, &offset) == FAILURE) {
RETURN_FALSE;
}

ZEND_FETCH_RESOURCE(im_src, gdImagePtr, &SIM, -1, "Image", le_gd);

nelem = zend_hash_num_elements(Z_ARRVAL_P(hash_matrix));
if (nelem != 3) {
php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");
RETURN_FALSE;
}

for (i=0; i<3; i++) {
if (zend_hash_index_find(Z_ARRVAL_P(hash_matrix), (i), (void **) &var) == SUCCESS && Z_TYPE_PP(var) == IS_ARRAY) {
if (Z_TYPE_PP(var) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_PP(var)) != 3 ) {
php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");
RETURN_FALSE;
}

for (j=0; j<3; j++) {
if (zend_hash_index_find(Z_ARRVAL_PP(var), (j), (void **) &var2) == SUCCESS) {
 SEPARATE_ZVAL(var2);
 convert_to_double(*var2);
 matrix[i][j] = (float)Z_DVAL_PP(var2);
} else {
php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have a 3x3 matrix");
RETURN_FALSE;
}
}
}
}
res = gdImageConvolution(im_src, matrix, (float)div, (float)offset);

if (res) {
RETURN_TRUE;
} else {
RETURN_FALSE;
}
}
</function>

<line num="7">float matrix[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};</line>
<line num="9">if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "radd", &SIM, &hash_matrix, &div, &offset) == FAILURE) {</line>
<line num="13">ZEND_FETCH_RESOURCE(im_src, gdImagePtr, &SIM, -1, "Image", le_gd);</line>
<line num="15">nelem = zend_hash_num_elements(Z_ARRVAL_P(hash_matrix));</line>
<line num="17">php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");</line>
<line num="22">if (zend_hash_index_find(Z_ARRVAL_P(hash_matrix), (i), (void **) &var) == SUCCESS && Z_TYPE_PP(var) == IS_ARRAY) {</line>
<line num="23">if (Z_TYPE_PP(var) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_PP(var)) != 3 ) {</line>
<line num="24">php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");</line>
<line num="29">if (zend_hash_index_find(Z_ARRVAL_PP(var), (j), (void **) &var2) == SUCCESS) {</line>
<line num="32">matrix[i][j] = (float)Z_DVAL_PP(var2);</line>

Example 2 Response: [
  {
    "content": "float matrix[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};",
    "num": "7",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "radd", &SIM, &hash_matrix, &div, &offset) == FAILURE) {",
    "num": "9",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "ZEND_FETCH_RESOURCE(im_src, gdImagePtr, &SIM, -1, "Image", le_gd);",
    "num": "13",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "nelem = zend_hash_num_elements(Z_ARRVAL_P(hash_matrix));",
    "num": "15",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");",
    "num": "17",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if (zend_hash_index_find(Z_ARRVAL_P(hash_matrix), (i), (void **) &var) == SUCCESS && Z_TYPE_PP(var) == IS_ARRAY) {",
    "num": "22",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if (Z_TYPE_PP(var) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_PP(var)) != 3 ) {",
    "isVulnerable": "false",
    "cwe": "23",
    "reason": "",
    "fix": ""
  },
  {
    "content": "php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");",
    "isVulnerable": "false",
    "cwe": "24",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if (zend_hash_index_find(Z_ARRVAL_PP(var), (j), (void **) &var2) == SUCCESS) {",
    "num": "29",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
    {
    "content": " matrix[i][j] = (float)Z_DVAL_PP(var2);"
    "num": "32",
    "isVulnerable": "true",
    "cwe": "CWE-189",
    "reason": "ext/gd/gd.c in PHP 5.5.x before 5.5.9 does not check data types, which might allow remote attackers to obtain sensitive information by using a (1) string or (2) array data type in place of a numeric data type, as demonstrated by an imagecrop function call with a string for the x dimension value, a different vulnerability than CVE-2013-7226.",
    "fix": "matrix[i][j] = (float)Z_DVAL(dval);"
  }
]

Think about the answer step by step, and only answer with JSON. I repeat, only answer with JSON.`;

const examplesV2 = `\n
Input: <function>
gray_render_span( int y,
int             count,
const FT_Span*  spans,
PWorker         worker )
{
unsigned char*  p;
FT_Bitmap*      map = &worker->target;

/* frst of all, compute the scanline offset */
p = (unsigned char*)map->buffer - y * map->pitch;
if ( map->pitch >= 0 )
      p += ( map->rows - 1 ) * map->pitch;

for ( ; count > 0; count--, spans++ )
{
unsigned char  coverage = spans->coverage;

if ( coverage )
{
/* For small-spans it is faster to do it by ourselves than
calling 'memset'. This is mainly due to the cost of the
function call.
*/
if ( spans->len >= 8 )
FT_MEM_SET( p + spans->x, (unsigned char)coverage, spans->len );
else
{
unsigned char*  q = p + spans->x;

switch ( spans->len )
{
case 7: *q++ = (unsigned char)coverage;
case 6: *q++ = (unsigned char)coverage;
case 5: *q++ = (unsigned char)coverage;
case 4: *q++ = (unsigned char)coverage;
case 3: *q++ = (unsigned char)coverage;
case 2: *q++ = (unsigned char)coverage;
case 1: *q   = (unsigned char)coverage;
default:
;
}
}
}
}
}
</function>

<line>gray_render_span(</line>
<line>p = (unsigned char*)map->buffer - y * map->pitch;</line>
<line>p += ( map->rows - 1 ) * map->pitch;</line>
<line>/* For small-spans it is faster to do it by ourselves than</line>
<line>* calling 'memset'.  This is mainly due to the cost of the</line>
<line>FT_MEM_SET( p + spans->x, (unsigned char)coverage, spans->len );</line>
<line>unsigned char*  q = p + spans->x;</line>
<line>case 7: *q++ = (unsigned char)coverage;</line>
<line>case 5: *q++ = (unsigned char)coverage;</line>
<line>case 3: *q++ = (unsigned char)coverage;</line>

Response: [
  {
    "content": "gray_render_span(",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "p = (unsigned char*)map->buffer - y * map->pitch;",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "p += ( map->rows - 1 ) * map->pitch;",
    "isVulnerable": "true",
    "cwe": "CWE-189",
    "reason": "Integer overflow in the gray_render_span function in smooth/ftgrays.c in FreeType before 2.4.0 allows remote attackers to cause a denial of service (application crash) or possibly execute arbitrary code via a crafted font file.",
    "fix": "p += (unsigned)( ( map->rows - 1 ) * map->pitch );"
  },
  {
    "content": "/* For small-spans it is faster to do it by ourselves than",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "* calling 'memset'.  This is mainly due to the cost of the",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "FT_MEM_SET( p + spans->x, (unsigned char)coverage, spans->len );",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "unsigned char*  q = p + spans->x;",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "case 7: *q++ = (unsigned char)coverage;",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "case 5: *q++ = (unsigned char)coverage;",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "case 3: *q++ = (unsigned char)coverage;",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  }
]

Input: <function>
  PHP_FUNCTION(imageconvolution)
{
zval *SIM, *hash_matrix;
zval **var = NULL, **var2 = NULL;
gdImagePtr im_src = NULL;
double div, offset;
int nelem, i, j, res;
float matrix[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};

if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "radd", &SIM, &hash_matrix, &div, &offset) == FAILURE) {
RETURN_FALSE;
}

ZEND_FETCH_RESOURCE(im_src, gdImagePtr, &SIM, -1, "Image", le_gd);

nelem = zend_hash_num_elements(Z_ARRVAL_P(hash_matrix));
if (nelem != 3) {
php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");
RETURN_FALSE;
}

for (i=0; i<3; i++) {
if (zend_hash_index_find(Z_ARRVAL_P(hash_matrix), (i), (void **) &var) == SUCCESS && Z_TYPE_PP(var) == IS_ARRAY) {
if (Z_TYPE_PP(var) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_PP(var)) != 3 ) {
php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");
RETURN_FALSE;
}

for (j=0; j<3; j++) {
if (zend_hash_index_find(Z_ARRVAL_PP(var), (j), (void **) &var2) == SUCCESS) {
 SEPARATE_ZVAL(var2);
 convert_to_double(*var2);
 matrix[i][j] = (float)Z_DVAL_PP(var2);
} else {
php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have a 3x3 matrix");
RETURN_FALSE;
}
}
}
}
res = gdImageConvolution(im_src, matrix, (float)div, (float)offset);

if (res) {
RETURN_TRUE;
} else {
RETURN_FALSE;
}
}
</function>

<line>float matrix[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};</line>
<line>if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "radd", &SIM, &hash_matrix, &div, &offset) == FAILURE) {</line>
<line>matrix[i][j] = (float)Z_DVAL_PP(var2);</line>
<line>ZEND_FETCH_RESOURCE(im_src, gdImagePtr, &SIM, -1, "Image", le_gd);</line>
<line>nelem = zend_hash_num_elements(Z_ARRVAL_P(hash_matrix));</line>
<line>php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");</line>
<line>if (zend_hash_index_find(Z_ARRVAL_P(hash_matrix), (i), (void **) &var) == SUCCESS && Z_TYPE_PP(var) == IS_ARRAY) {</line>
<line>if (Z_TYPE_PP(var) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_PP(var)) != 3 ) {</line>
<line>php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");</line>
<line>if (zend_hash_index_find(Z_ARRVAL_PP(var), (j), (void **) &var2) == SUCCESS) {</line>

Response: [
  {
    "content": "float matrix[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "radd", &SIM, &hash_matrix, &div, &offset) == FAILURE) {",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": " matrix[i][j] = (float)Z_DVAL_PP(var2);"
    "isVulnerable": "true",
    "cwe": "CWE-189",
    "reason": "ext/gd/gd.c in PHP 5.5.x before 5.5.9 does not check data types, which might allow remote attackers to obtain sensitive information by using a (1) string or (2) array data type in place of a numeric data type, as demonstrated by an imagecrop function call with a string for the x dimension value, a different vulnerability than CVE-2013-7226.",
    "fix": "matrix[i][j] = (float)Z_DVAL(dval);"
  },
  {
    "content": "ZEND_FETCH_RESOURCE(im_src, gdImagePtr, &SIM, -1, "Image", le_gd);",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "nelem = zend_hash_num_elements(Z_ARRVAL_P(hash_matrix));",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if (zend_hash_index_find(Z_ARRVAL_P(hash_matrix), (i), (void **) &var) == SUCCESS && Z_TYPE_PP(var) == IS_ARRAY) {",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if (Z_TYPE_PP(var) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_PP(var)) != 3 ) {",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if (zend_hash_index_find(Z_ARRVAL_PP(var), (j), (void **) &var2) == SUCCESS) {",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  }
]

Example 2 input: <function>
  static inline int acm_set_control(struct acm *acm, int control){ 
 if (acm->quirks & QUIRK_CONTROL_LINE_STATE) 
  return -EOPNOTSUPP; 
 
 return acm_ctrl_msg(acm, USB_CDC_REQ_SET_CONTROL_LINE_STATE, 
   control, NULL, 0); 
} 
</function>

<line>static inline int acm_set_control(struct acm *acm, int control){</line>
<line>if (acm->quirks & QUIRK_CONTROL_LINE_STATE)</line>
<line>return -EOPNOTSUPP;</line>
<line>return acm_ctrl_msg(acm, USB_CDC_REQ_SET_CONTROL_LINE_STATE,</line>
<line>control, NULL, 0);</line>

Example 2 response: [
  {
    "content": "static inline int acm_set_control(struct acm *acm, int control){",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if (acm->quirks & QUIRK_CONTROL_LINE_STATE)",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "return -EOPNOTSUPP;",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content":"return acm_ctrl_msg(acm, USB_CDC_REQ_SET_CONTROL_LINE_STATE,",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "control, NULL, 0);",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  }
]`;
