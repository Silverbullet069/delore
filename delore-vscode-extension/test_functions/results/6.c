 int open_debug_log(void) {
// int open_debug_log(void)
// {
// 	int fh;
// 	struct stat st;
  
  	 
  	if(verify_config || test_scheduling == TRUE)
 		return OK;
 
 	 
  	if(debug_level == DEBUGL_NONE)
  		return OK;
  
	if((debug_file_fp = fopen(debug_file, "a+")) == NULL)
// 	if ((fh = open(debug_file, O_RDWR|O_APPEND|O_CREAT|O_NOFOLLOW, S_IRUSR|S_IWUSR)) == -1)
// 		return ERROR;
// 	if((debug_file_fp = fdopen(fh, "a+")) == NULL)
// 		return ERROR;
// 
// 	if ((fstat(fh, &st)) == -1) {
// 		debug_file_fp = NULL;
// 		close(fh);
// 		return ERROR;
// 	}
// 	if (st.st_nlink != 1 || (st.st_mode & S_IFMT) != S_IFREG) {
// 		debug_file_fp = NULL;
// 		close(fh);
  		return ERROR;
// 	}
  
	(void)fcntl(fileno(debug_file_fp), F_SETFD, FD_CLOEXEC);
// 	(void)fcntl(fh, F_SETFD, FD_CLOEXEC);
  
  	return OK;
  	}

tBTM_STATUS BTM_SecGetDeviceLinkKey (BD_ADDR bd_addr, LINK_KEY link_key)
{
    tBTM_SEC_DEV_REC *p_dev_rec;

 if (((p_dev_rec = btm_find_dev (bd_addr)) != NULL)
 && (p_dev_rec->sec_flags & BTM_SEC_LINK_KEY_KNOWN))
 {
        memcpy (link_key, p_dev_rec->link_key, LINK_KEY_LEN);
 return(BTM_SUCCESS);
 }
 return(BTM_UNKNOWN_ADDR);
}


 void MostVisitedSitesBridge::JavaObserver::OnMostVisitedURLsAvailable(
     const NTPTilesVector& tiles) {
   JNIEnv* env = AttachCurrentThread();
   std::vector<base::string16> titles;
   std::vector<std::string> urls;
   std::vector<std::string> whitelist_icon_paths;
   std::vector<int> sources;
 
   titles.reserve(tiles.size());
   urls.reserve(tiles.size());
   whitelist_icon_paths.reserve(tiles.size());
   sources.reserve(tiles.size());
   for (const auto& tile : tiles) {
     titles.emplace_back(tile.title);
     urls.emplace_back(tile.url.spec());
      whitelist_icon_paths.emplace_back(tile.whitelist_icon_path.value());
      sources.emplace_back(static_cast<int>(tile.source));
    }
  Java_MostVisitedURLsObserver_onMostVisitedURLsAvailable(
//   Java_Observer_onMostVisitedURLsAvailable(
        env, observer_, ToJavaArrayOfStrings(env, titles),
        ToJavaArrayOfStrings(env, urls),
        ToJavaArrayOfStrings(env, whitelist_icon_paths),
       ToJavaIntArray(env, sources));
 }

 static int __init ipip_init(void)
 {
 	int err;
  
  	printk(banner);
  
	if (xfrm4_tunnel_register(&ipip_handler, AF_INET)) {
// 	err = register_pernet_device(&ipip_net_ops);
// 	if (err < 0)
// 		return err;
// 	err = xfrm4_tunnel_register(&ipip_handler, AF_INET);
// 	if (err < 0) {
// 		unregister_pernet_device(&ipip_net_ops);
  		printk(KERN_INFO "ipip init: can't register tunnel\n");
		return -EAGAIN;
  	}
	err = register_pernet_device(&ipip_net_ops);
	if (err)
		xfrm4_tunnel_deregister(&ipip_handler, AF_INET);
  	return err;
  }

  bool FrameworkListener::onDataAvailable(SocketClient *c) {
  char buffer[CMD_BUF_SIZE];
  int len;
 
     len = TEMP_FAILURE_RETRY(read(c->getSocket(), buffer, sizeof(buffer)));
 
      if (len < 0) {
          SLOGE("read() failed (%s)", strerror(errno));
          return false;
    } else if (!len)
//     } else if (!len) {
          return false;
   if(buffer[len-1] != '\0')
//     } else if (buffer[len-1] != '\0') {
          SLOGW("String is not zero-terminated");
//         android_errorWriteLog(0x534e4554, "29831647");
//         c->sendMsg(500, "Command too large for buffer", false);
//         mSkipToNextNullByte = true;
//         return false;
//     }
  
      int offset = 0;
      int i;
 
 
      for (i = 0; i < len; i++) {
          if (buffer[i] == '\0') {
               
            dispatchCommand(c, buffer + offset);
//             if (mSkipToNextNullByte) {
//                 mSkipToNextNullByte = false;
//             } else {
//                 dispatchCommand(c, buffer + offset);
//             }
              offset = i + 1;
          }
      }
  
//     mSkipToNextNullByte = false;
      return true;
  }