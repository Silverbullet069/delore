export const OPEN_CWE_API = 'https://www.opencve.io/api/cwe/';
export const TOP_25_MOST_DANGEROUS_CWEs = [
  'CWE-787',
  'CWE-79',
  'CWE-125',
  'CWE-20',
  'CWE-78',
  'CWE-89',
  'CWE-416',
  'CWE-22',
  'CWE-352',
  'CWE-434',
  'CWE-306',
  'CWE-190',
  'CWE-502',
  'CWE-287',
  'CWE-476',
  'CWE-798',
  'CWE-119',
  'CWE-862',
  'CWE-276',
  'CWE-200',
  'CWE-522',
  'CWE-732',
  'CWE-611',
  'CWE-918',
  'CWE-77'
];

export const TOP_10_MOST_CORRECT_PREDICTED_CWEs = [
  'CWE-284',
  'CWE-269',
  'CWE-254',
  'CWE-415',
  'CWE-311',
  'CWE-22',
  'CWE-17',
  'CWE-617',
  'CWE-358',
  'CWE-285'
];

export const learnCWEs = `
Learn about the following 35 CWEs description in JSON format. Do not print anything:

1.{"id": "CWE-787", "name": "Out-of-bounds Write", "description": "The software writes data past the end, or before the beginning, of the intended buffer."}

2.{"id": "CWE-79", "name": "Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')", "description": "The software does not neutralize or incorrectly neutralizes user-controllable input before it is placed in output that is used as a web page that is served to other users."}

3.{"id": "CWE-125", "name": "Out-of-bounds Read", "description": "The software reads data past the end, or before the beginning, of the intended buffer."}

4.{"id": "CWE-20", "name": "Improper Input Validation", "description": "The product receives input or data, but it does\n        not validate or incorrectly validates that the input has the\n        properties that are required to process the data safely and\n        correctly."}

5.{"id": "CWE-78", "name": "Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')", "description": "The software constructs all or part of an OS command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended OS command when it is sent to a downstream component."}

6.{"id": "CWE-89", "name": "Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')", "description": "The software constructs all or part of an SQL command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended SQL command when it is sent to a downstream component."}

7.{"id": "CWE-416", "name": "Use After Free", "description": "Referencing memory after it has been freed can cause a program to crash, use unexpected values, or execute code."}

8.{"id": "CWE-22", "name": "Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')", "description": "The software uses external input to construct a pathname that is intended to identify a file or directory that is located underneath a restricted parent directory, but the software does not properly neutralize special elements within the pathname that can cause the pathname to resolve to a location that is outside of the restricted directory."}

9.{"id": "CWE-352", "name": "Cross-Site Request Forgery (CSRF)", "description": "The web application does not, or can not, sufficiently verify whether a well-formed, valid, consistent request was intentionally provided by the user who submitted the request."}

10.{"id": "CWE-434", "name": "Unrestricted Upload of File with Dangerous Type", "description": "The software allows the attacker to upload or transfer files of dangerous types that can be automatically processed within the product's environment."}

11.{"id": "CWE-306", "name": "Missing Authentication for Critical Function", "description": "The software does not perform any authentication for functionality that requires a provable user identity or consumes a significant amount of resources."}

12.{"id": "CWE-190", "name": "Integer Overflow or Wraparound", "description": "The software performs a calculation that can produce an integer overflow or wraparound, when the logic assumes that the resulting value will always be larger than the original value. This can introduce other weaknesses when the calculation is used for resource management or execution control."}

13.{"id": "CWE-502", "name": "Deserialization of Untrusted Data", "description": "The application deserializes untrusted data without sufficiently verifying that the resulting data will be valid."}

14.{"id": "CWE-287", "name": "Improper Authentication", "description": "When an actor claims to have a given identity, the software does not prove or insufficiently proves that the claim is correct."}

15.{"id": "CWE-476", "name": "NULL Pointer Dereference", "description": "A NULL pointer dereference occurs when the application dereferences a pointer that it expects to be valid, but is NULL, typically causing a crash or exit."}

16.{"id": "CWE-798", "name": "Use of Hard-coded Credentials", "description": "The software contains hard-coded credentials, such as a password or cryptographic key, which it uses for its own inbound authentication, outbound communication to external components, or encryption of internal data."}

17.{"id": "CWE-119", "name": "Improper Restriction of Operations within the Bounds of a Memory Buffer", "description": "The software performs operations on a memory buffer, but it can read from or write to a memory location that is outside of the intended boundary of the buffer."}

18.{"id": "CWE-862", "name": "Missing Authorization", "description": "The software does not perform an authorization check when an actor attempts to access a resource or perform an action."}

19.{"id": "CWE-276", "name": "Incorrect Default Permissions", "description": "During installation, installed file permissions are set to allow anyone to modify those files."}

20.{"id": "CWE-200", "name": "Exposure of Sensitive Information to an Unauthorized Actor", "description": "The product exposes sensitive information to an actor that is not explicitly authorized to have access to that information."}

21.{"id": "CWE-522", "name": "Insufficiently Protected Credentials", "description": "The product transmits or stores authentication credentials, but it uses an insecure method that is susceptible to unauthorized interception and/or retrieval."}

22.{"id": "CWE-732", "name": "Incorrect Permission Assignment for Critical Resource", "description": "The product specifies permissions for a security-critical resource in a way that allows that resource to be read or modified by unintended actors."}

23.{"id": "CWE-611", "name": "Improper Restriction of XML External Entity Reference", "description": "The software processes an XML document that can contain XML entities with URIs that resolve to documents outside of the intended sphere of control, causing the product to embed incorrect documents into its output."}

24.{"id": "CWE-918", "name": "Server-Side Request Forgery (SSRF)", "description": "The web server receives a URL or similar request from an upstream component and retrieves the contents of this URL, but it does not sufficiently ensure that the request is being sent to the expected destination."}

25.{"id": "CWE-77", "name": "Improper Neutralization of Special Elements used in a Command ('Command Injection')", "description": "The software constructs all or part of a command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended command when it is sent to a downstream component."}

26.{"id": "CWE-284", "name": "Improper Access Control", "description": "The software does not restrict or incorrectly restricts access to a resource from an unauthorized actor."}

27.{"id": "CWE-269", "name": "Improper Privilege Management", "description": "The software does not properly assign, modify, track, or check privileges for an actor, creating an unintended sphere of control for that actor."}

28.{"id": "CWE-254", "name": "7PK - Security Features", "description": "Software security is not security software. Here we're concerned with topics like authentication, access control, confidentiality, cryptography, and privilege management."}

29.{"id": "CWE-415", "name": "Double Free", "description": "The product calls free() twice on the same memory address, potentially leading to modification of unexpected memory locations."}

30.{"id": "CWE-311", "name": "Missing Encryption of Sensitive Data", "description": "The software does not encrypt sensitive or critical information before storage or transmission."}

31.{"id": "CWE-22", "name": "Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')", "description": "The software uses external input to construct a pathname that is intended to identify a file or directory that is located underneath a restricted parent directory, but the software does not properly neutralize special elements within the pathname that can cause the pathname to resolve to a location that is outside of the restricted directory."}

32.{"id": "CWE-17", "name": "DEPRECATED: Code", "description": "This entry has been deprecated.  It was originally used for organizing the Development View (CWE-699) and some other views, but it introduced unnecessary complexity and depth to the resulting tree."}

33.{"id": "CWE-617", "name": "Reachable Assertion", "description": "The product contains an assert() or similar statement that can be triggered by an attacker, which leads to an application exit or other behavior that is more severe than necessary."}

34.{"id": "CWE-358", "name": "Improperly Implemented Security Check for Standard", "description": "The software does not implement or incorrectly implements one or more security-relevant checks as specified by the design of a standardized algorithm, protocol, or technique."}

35.{"id": "CWE-285", "name": "Improper Authorization", "description": "The software does not perform or incorrectly performs an authorization check when an actor attempts to access a resource or perform an action."}
`;

const instructionsV1 = `You are a security expert whose job is to fix vulnerabilities in ONE C/C++ function. Your provided input will be:
- ONE AND ONLY ONE C/C++ function (delimited with XML open and close tag <function> and </function>) that is deemed to be vulnerable by me. 
- A unique set of lines that extracted from that function (delimited with XML open and close tag: <line> and </line>). Only 1 or 2 lines have the highest vulnerability potential, the rest are false positives.

Your output will be a TS object which has the following format and value type:
{
  "isFunctionVulnerable": boolean,
  "lines": {
    "content": string,
    "isVulnerable": boolean
    "suggestion": string
  }[]
}

Context:
- "isFunctionVulnerable": Answer "Whether or not the specified function is vulnerable or not?". Rely on my judgement about saying the function is vulnerable from the start. You can only answer "true" or "false" here, since its type is boolean. You can only deduce vulnerability or not using CWE/CVE characteristics.
- "lines": an array of objects the specify the information of every lines inside the function. Each element consists of 3 properties:
+ "content": the text content of the line. It's a string.
+ "isVulnerable": Answer "Whether or not this line is vulnerable or not?". Rely on my judgement about saying some of the lines have the highest vulnerability potential from the start. You can only answer "true" or "false" here, since its type is boolean. You can only deduce vulnerability or not using CWE/CVE characteristics. 
+ "suggestion": Answer "How will you fix this line if you deemed it vulnerable?". Write CWE/CVE type of that line's vulnerability, if it had one. Don't think it's CWE-119: Buffer overflow.
End of context.`;

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
  "reason": string (you write the reason why you predicted as vulnerable with that "cwe")
  "fix": string (you rewrite a secure version of the code that whose exact functionality is preserved)
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

const example = `
Input: <function>
void nothing() {
  return;
}
</function>

<line>void nothing() {</line>
<line>return;</line>

Response: [
  {
    "content": "void nothing() {",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "return;",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  }
]`;

const example2 = `Example 2 input: <function>
xmlXPathNextPrecedingInternal(xmlXPathParserContextPtr ctxt,
xmlNodePtr cur)
{
if ((ctxt == NULL) || (ctxt->context == NULL)) return(NULL);
    if ((ctxt->context->node->type == XML_ATTRIBUTE_NODE) ||
 (ctxt->context->node->type == XML_NAMESPACE_DECL))
 return(NULL);
if (cur == NULL) {
cur = ctxt->context->node;
if (cur == NULL)
return (NULL);
ctxt->ancestor = cur->parent;
}
if ((cur->prev != NULL) && (cur->prev->type == XML_DTD_NODE))
cur = cur->prev;
while (cur->prev == NULL) {
cur = cur->parent;
if (cur == NULL)
return (NULL);
if (cur == ctxt->context->doc->children)
return (NULL);
if (cur != ctxt->ancestor)
return (cur);
ctxt->ancestor = cur->parent;
}
cur = cur->prev;
while (cur->last != NULL)
cur = cur->last;
return (cur);
}
</function>

<line>xmlXPathNextPrecedingInternal(xmlXPathParserContextPtr ctxt,</line>
<line>if ((ctxt == NULL) || (ctxt->context == NULL)) return(NULL);</line>
<line>if ((ctxt->context->node->type == XML_ATTRIBUTE_NODE) ||</line>
<line> (ctxt->context->node->type == XML_NAMESPACE_DECL))</line>
<line>if ((cur->prev != NULL) && (cur->prev->type == XML_DTD_NODE))</line>
<line>while (cur->prev == NULL) {</line>
<line>if (cur == ctxt->context->doc->children)</line>
<line>if (cur != ctxt->ancestor)</line>
<line>ctxt->ancestor = cur->parent;</line>
<line>while (cur->last != NULL)</line>

Example 2 response: [
  {
    "content": "xmlXPathNextPrecedingInternal(xmlXPathParserContextPtr ctxt,",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if ((ctxt == NULL) || (ctxt->context == NULL)) return(NULL);",
    "isVulnerable": "true",
    "cwe": "",
    "reason": "Apply behaviour change fix from upstream for previous XPath change",
    "fix": ""
  },
  {
    "content": "if ((ctxt->context->node->type == XML_ATTRIBUTE_NODE) ||",
    "isVulnerable": "true",
    "cwe": "",
    "reason": "Apply behaviour change fix from upstream for previous XPath change",
    "fix": ""
  },
  {
    "content":" (ctxt->context->node->type == XML_NAMESPACE_DECL))",
    "isVulnerable": "true",
    "cwe": "",
    "reason": "Apply behaviour change fix from upstream for previous XPath change",
    "fix": ""
  },
  {
    "content": "if ((cur->prev != NULL) && (cur->prev->type == XML_DTD_NODE))",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "while (cur->prev == NULL) {",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if (cur == ctxt->context->doc->children)",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "if (cur != ctxt->ancestor)",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "ctxt->ancestor = cur->parent;",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
  {
    "content": "while (cur->last != NULL)",
    "isVulnerable": "false",
    "cwe": "",
    "reason": "",
    "fix": ""
  },
]`;
