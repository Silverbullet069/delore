 static ssize_t get_node_path_locked(struct node* node, char* buf, size_t bufsize) {
  const char* name;
  size_t namelen;
  if (node->graft_path) {
         name = node->graft_path;
         namelen = node->graft_pathlen;
  } else if (node->actual_name) {
         name = node->actual_name;
         namelen = node->namelen;
  } else {
         name = node->name;
         namelen = node->namelen;
  }
 
  if (bufsize < namelen + 1) {
  return -1;
  }
 
  
      ssize_t pathlen = 0;
      if (node->parent && node->graft_path == NULL) {
        pathlen = get_node_path_locked(node->parent, buf, bufsize - namelen - 2);
//         pathlen = get_node_path_locked(node->parent, buf, bufsize - namelen - 1);
          if (pathlen < 0) {
              return -1;
          }
         buf[pathlen++] = '/';
  }
 
     memcpy(buf + pathlen, name, namelen + 1);  
  return pathlen + namelen;
 }

bool PrintMsg_Print_Params_IsEmpty(const PrintMsg_Print_Params& params) {
  return !params.document_cookie && !params.desired_dpi && !params.max_shrink &&
         !params.min_shrink && !params.dpi && params.printable_size.IsEmpty() &&
         !params.selection_only && params.page_size.IsEmpty() &&
         !params.margin_top && !params.margin_left &&
         !params.supports_alpha_blend;
}


 static void copy_asoundrc(void) {
 	char *src = RUN_ASOUNDRC_FILE ;
  	char *dest;
  	if (asprintf(&dest, "%s/.asoundrc", cfg.homedir) == -1)
  		errExit("asprintf");
// 	
  	if (is_link(dest)) {
  		fprintf(stderr, "Error: %s is a symbolic link\n", dest);
  		exit(1);
  	}
  
	pid_t child = fork();
	if (child < 0)
		errExit("fork");
	if (child == 0) {
		drop_privs(0);
		int rv = copy_file(src, dest);
		if (rv)
			fprintf(stderr, "Warning: cannot transfer .asoundrc in private home directory\n");
		else {
			fs_logger2("clone", dest);
		}
		_exit(0);
	}
	waitpid(child, NULL, 0);
	if (chown(dest, getuid(), getgid()) < 0)
		errExit("chown");
	if (chmod(dest, S_IRUSR | S_IWUSR) < 0)
		errExit("chmod");
// 	copy_file_as_user(src, dest, getuid(), getgid(), S_IRUSR | S_IWUSR);  
// 	fs_logger2("clone", dest);
  
  	unlink(src);
 }

 int mp_pack(lua_State *L) {
     int nargs = lua_gettop(L);
     int i;
     mp_buf *buf;
 
      if (nargs == 0)
          return luaL_argerror(L, 0, "MessagePack pack needs input.");
  
//     if (!lua_checkstack(L, nargs))
//         return luaL_argerror(L, 0, "Too many arguments for MessagePack pack.");
// 
      buf = mp_buf_new(L);
      for(i = 1; i <= nargs; i++) {
           
         lua_pushvalue(L, i);
 
         mp_encode_lua_type(L,buf,0);
 
         lua_pushlstring(L,(char*)buf->b,buf->len);
 
          
         buf->free += buf->len;
         buf->len = 0;
     }
     mp_buf_free(L, buf);
 
      
     lua_concat(L, nargs);
     return 1;
 }

static int netsnmp_session_set_sec_level(struct snmp_session *s, char *level)
{
	if (!strcasecmp(level, "noAuthNoPriv") || !strcasecmp(level, "nanp")) {
		s->securityLevel = SNMP_SEC_LEVEL_NOAUTH;
	} else if (!strcasecmp(level, "authNoPriv") || !strcasecmp(level, "anp")) {
		s->securityLevel = SNMP_SEC_LEVEL_AUTHNOPRIV;
	} else if (!strcasecmp(level, "authPriv") || !strcasecmp(level, "ap")) {
		s->securityLevel = SNMP_SEC_LEVEL_AUTHPRIV;
	} else {
		return (-1);
	}
	return (0);
}
