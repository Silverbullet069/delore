  void SocketStream::set_context(URLRequestContext* context) {
  const URLRequestContext* prev_context = context_.get();
//   const URLRequestContext* prev_context = context_;
  
  if (context) {
    context_ = context->AsWeakPtr();
  } else {
    context_.reset();
  }
//   context_ = context;
  
    if (prev_context != context) {
      if (prev_context && pac_request_) {
       prev_context->proxy_service()->CancelPacRequest(pac_request_);
       pac_request_ = NULL;
     }
 
     net_log_.EndEvent(NetLog::TYPE_REQUEST_ALIVE);
     net_log_ = BoundNetLog();
 
     if (context) {
       net_log_ = BoundNetLog::Make(
           context->net_log(),
           NetLog::SOURCE_SOCKET_STREAM);
 
       net_log_.BeginEvent(NetLog::TYPE_REQUEST_ALIVE);
     }
   }
  }

 int php_wddx_deserialize_ex(char *value, int vallen, zval *return_value)
 {
 	wddx_stack stack;
 	XML_Parser parser;
 	st_entry *ent;
 	int retval;
 
 	wddx_stack_init(&stack);
 	parser = XML_ParserCreate("UTF-8");
 
 	XML_SetUserData(parser, &stack);
 	XML_SetElementHandler(parser, php_wddx_push_element, php_wddx_pop_element);
 	XML_SetCharacterDataHandler(parser, php_wddx_process_data);
 
 	XML_Parse(parser, value, vallen, 1);
 
 	XML_ParserFree(parser);
  
  	if (stack.top == 1) {
  		wddx_stack_top(&stack, (void**)&ent);
		*return_value = *(ent->data);
		zval_copy_ctor(return_value);
		retval = SUCCESS;
// 		if(ent->data == NULL) {
// 			retval = FAILURE;
// 		} else {
// 			*return_value = *(ent->data);
// 			zval_copy_ctor(return_value);
// 			retval = SUCCESS;
// 		}
  	} else {
  		retval = FAILURE;
  	}
 
 	wddx_stack_destroy(&stack);
 
 	return retval;
 }

 zend_object_iterator *spl_filesystem_dir_get_iterator(zend_class_entry *ce, zval *object, int by_ref TSRMLS_DC)
 {
 	spl_filesystem_iterator *iterator;
 	spl_filesystem_object   *dir_object;
 
 	if (by_ref) {
 		zend_error(E_ERROR, "An iterator cannot be used with foreach by reference");
 	}
 	dir_object = (spl_filesystem_object*)zend_object_store_get_object(object TSRMLS_CC);
 	iterator   = spl_filesystem_object_to_iterator(dir_object);
 
 	 
 	if (iterator->intern.data == NULL) {
 		iterator->intern.data = object;
 		iterator->intern.funcs = &spl_filesystem_dir_it_funcs;
 		 
  		iterator->current = object;
  	}
  	zval_add_ref(&object);
// 
  	return (zend_object_iterator*)iterator;
  }

 SplashBitmap::SplashBitmap(int widthA, int heightA, int rowPad,
 			   SplashColorMode modeA, GBool alphaA,
 			   GBool topDown) {
   width = widthA;
   height = heightA;
   mode = modeA;
   switch (mode) {
   case splashModeMono1:
     rowSize = (width + 7) >> 3;
     break;
   case splashModeMono8:
     rowSize = width;
     break;
   case splashModeRGB8:
   case splashModeBGR8:
     rowSize = width * 3;
     break;
   case splashModeXBGR8:
     rowSize = width * 4;
     break;
 #if SPLASH_CMYK
   case splashModeCMYK8:
     rowSize = width * 4;
     break;
 #endif
    }
    rowSize += rowPad - 1;
    rowSize -= rowSize % rowPad;
  data = (SplashColorPtr)gmalloc(rowSize * height);
//   data = (SplashColorPtr)gmallocn(rowSize, height);
    if (!topDown) {
      data += (height - 1) * rowSize;
      rowSize = -rowSize;
    }
    if (alphaA) {
    alpha = (Guchar *)gmalloc(width * height);
//     alpha = (Guchar *)gmallocn(width, height);
    } else {
      alpha = NULL;
    }
 }

 void WebstoreStandaloneInstaller::BeginInstall() {
    AddRef();
  
    if (!crx_file::id_util::IdIsValid(id_)) {
    CompleteInstall(webstore_install::INVALID_ID, kInvalidWebstoreItemId);
//     CompleteInstall(webstore_install::INVALID_ID,
//                     webstore_install::kInvalidWebstoreItemId);
      return;
    }
  
   webstore_install::Result result = webstore_install::OTHER_ERROR;
   std::string error;
   if (!EnsureUniqueInstall(&result, &error)) {
     CompleteInstall(result, error);
     return;
   }
 
   webstore_data_fetcher_.reset(new WebstoreDataFetcher(
       this,
       profile_->GetRequestContext(),
       GetRequestorURL(),
       id_));
   webstore_data_fetcher_->Start();
 }