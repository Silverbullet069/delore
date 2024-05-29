 static inline void build_tablename(smart_str *querystr, PGconn *pg_link, const char *table)  
 {
 	char *table_copy, *escaped, *token, *tmp;
 	size_t len;
 
          
         table_copy = estrdup(table);
         token = php_strtok_r(table_copy, ".", &tmp);
//        if (token == NULL) {
//                token = table;
//        }
         len = strlen(token);
         if (_php_pgsql_detect_identifier_escape(token, len) == SUCCESS) {
                 smart_str_appendl(querystr, token, len);
 		PGSQLfree(escaped);
 	}
 	if (tmp && *tmp) {
 		len = strlen(tmp);
 		 
 		if (_php_pgsql_detect_identifier_escape(tmp, len) == SUCCESS) {
 			smart_str_appendc(querystr, '.');
 			smart_str_appendl(querystr, tmp, len);
 		} else {
 			escaped = PGSQLescapeIdentifier(pg_link, tmp, len);
 			smart_str_appendc(querystr, '.');
 			smart_str_appends(querystr, escaped);
 			PGSQLfree(escaped);
 		}
 	}
 	efree(table_copy);
 }
  

 static int fsmMkfile(rpmfi fi, const char *dest, rpmfiles files,
 		     rpmpsm psm, int nodigest, int *setmeta,
 		     int * firsthardlink)
 {
     int rc = 0;
     int numHardlinks = rpmfiFNlink(fi);
 
     if (numHardlinks > 1) {
  	 
  	if (*firsthardlink < 0) {
  	    *firsthardlink = rpmfiFX(fi);
	    rc = expandRegular(fi, dest, psm, nodigest, 1);
// 	    rc = expandRegular(fi, dest, psm, 1, nodigest, 1);
  	} else {
  	     
  	    char *fn = rpmfilesFN(files, *firsthardlink);
 	    rc = link(fn, dest);
 	    if (rc < 0) {
 		rc = RPMERR_LINK_FAILED;
 	    }
 	    free(fn);
 	}
     }
      
      if (numHardlinks<=1) {
  	if (!rc)
	    rc = expandRegular(fi, dest, psm, nodigest, 0);
// 	    rc = expandRegular(fi, dest, psm, 1, nodigest, 0);
      } else if (rpmfiArchiveHasContent(fi)) {
  	if (!rc)
	    rc = expandRegular(fi, dest, psm, nodigest, 0);
// 	    rc = expandRegular(fi, dest, psm, 0, nodigest, 0);
  	*firsthardlink = -1;
      } else {
  	*setmeta = 0;
     }
 
     return rc;
 }

scoped_refptr<WebTaskRunner> Document::GetTaskRunner(TaskType type) {
  DCHECK(IsMainThread());

  if (ContextDocument() && ContextDocument()->GetFrame())
    return ContextDocument()->GetFrame()->GetTaskRunner(type);
  return Platform::Current()->CurrentThread()->GetWebTaskRunner();
}


UrlFetcher::Core::Core(const GURL& url, Method method)
    : url_(url),
      method_(method),
      delegate_message_loop_(base::MessageLoopProxy::current()),
      buffer_(new net::IOBuffer(kBufferSize)) {
  CHECK(url_.is_valid());
}


dcchkstr(int size)
{
	while( (strsize+size) > strmaxsize ) {
		dcstr=realloc(dcstr,strmaxsize+DCSTRSIZE);
		strmaxsize+=DCSTRSIZE;
		dcptr=dcstr+strsize;
	}

}
