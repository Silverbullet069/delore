void RenameThread(const char* name)
{
#if defined(PR_SET_NAME)
	prctl(PR_SET_NAME, name, 0, 0, 0);
#elif defined(__APPLE__)
	pthread_setname_np(name);
#elif (defined(__FreeBSD__) || defined(__OpenBSD__))
	pthread_set_name_np(pthread_self(), name);
#else
	(void)name;
#endif
}


  bool HasProcess(RenderProcessHost* process) {
    for (auto iter : map_) {
      std::map<ProcessID, Count>& counts_per_process = iter.second;
      for (auto iter_process : counts_per_process) {
        if (iter_process.first == process->GetID())
          return true;
      }
    }
    return false;
  }


void DecodeIPV6RegisterTests(void)
{
#ifdef UNITTESTS
    UtRegisterTest("DecodeIPV6FragTest01", DecodeIPV6FragTest01);
    UtRegisterTest("DecodeIPV6RouteTest01", DecodeIPV6RouteTest01);
    UtRegisterTest("DecodeIPV6HopTest01", DecodeIPV6HopTest01);
#endif  
}


 static pdf_creator_t *new_creator(int *n_elements)
 {
     pdf_creator_t *daddy;
 
     static const pdf_creator_t creator_template[] = 
     {
         {"Title",        ""},
         {"Author",       ""},
         {"Subject",      ""},
         {"Keywords",     ""},
         {"Creator",      ""},
         {"Producer",     ""},
         {"CreationDate", ""},
         {"ModDate",      ""},
          {"Trapped",      ""},
      };
  
    daddy = malloc(sizeof(creator_template));
//     daddy = safe_calloc(sizeof(creator_template));
      memcpy(daddy, creator_template, sizeof(creator_template));
  
      if (n_elements)
       *n_elements = sizeof(creator_template) / sizeof(creator_template[0]);
 
     return daddy;
 }

void fscrypt_fname_free_buffer(struct fscrypt_str *crypto_str)
{
	if (!crypto_str)
		return;
	kfree(crypto_str->name);
	crypto_str->name = NULL;
}
