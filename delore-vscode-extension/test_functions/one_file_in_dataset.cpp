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

SchedulerObject::_continue(std::string key, std::string &/*reason*/, std::string &text) 
{ 
PROC_ID id = getProcByString(key.c_str()); 
       if (id.cluster < 0 || id.proc < 0) { 
dprintf(D_FULLDEBUG, "Remove: Failed to parse id: %s\ 
", key.c_str()); 
text = "Invalid Id"; 
return false; 
} 
 
scheduler.enqueueActOnJobMyself(id,JA_CONTINUE_JOBS,true); 
 
return true; 
} 
