  normalize_color_encoding(color_encoding *encoding)
  {
   PNG_CONST double whiteY = encoding->red.Y + encoding->green.Y +
//    const double whiteY = encoding->red.Y + encoding->green.Y +
        encoding->blue.Y;
  
     if (whiteY != 1)
  {
       encoding->red.X /= whiteY;
       encoding->red.Y /= whiteY;
       encoding->red.Z /= whiteY;
       encoding->green.X /= whiteY;
       encoding->green.Y /= whiteY;
       encoding->green.Z /= whiteY;
       encoding->blue.X /= whiteY;
       encoding->blue.Y /= whiteY;
       encoding->blue.Z /= whiteY;
  }
 
  }

 PHP_FUNCTION(xml_parse_into_struct)
 {
 	xml_parser *parser;
 	zval *pind, **xdata, **info = NULL;
 	char *data;
 	int data_len, ret;
 
         if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "rsZ|Z", &pind, &data, &data_len, &xdata, &info) == FAILURE) {
                 return;
         }
       if (info) {     
// 
//        if (info) {
                 zval_dtor(*info);
                 array_init(*info);
         }
 
 	ZEND_FETCH_RESOURCE(parser,xml_parser *, &pind, -1, "XML Parser", le_xml_parser);
 
 	zval_dtor(*xdata);
         array_init(*xdata);
  
         parser->data = *xdata;
// 
         if (info) {
                 parser->info = *info;
         }
// 
         parser->level = 0;
         parser->ltags = safe_emalloc(XML_MAXLEVEL, sizeof(char *), 0);
  
 	XML_SetDefaultHandler(parser->parser, _xml_defaultHandler);
 	XML_SetElementHandler(parser->parser, _xml_startElementHandler, _xml_endElementHandler);
 	XML_SetCharacterDataHandler(parser->parser, _xml_characterDataHandler);
 
 	parser->isparsing = 1;
 	ret = XML_Parse(parser->parser, data, data_len, 1);
 	parser->isparsing = 0;
 
 	RETVAL_LONG(ret);
  }

static inline void hfsplus_instantiate(struct dentry *dentry,
				       struct inode *inode, u32 cnid)
{
	dentry->d_fsdata = (void *)(unsigned long)cnid;
	d_instantiate(dentry, inode);
}


  bool PasswordAutofillAgent::TryToShowTouchToFill(
      const WebFormControlElement& control_element) {
    const WebInputElement* element = ToWebInputElement(&control_element);
  if (!element || (!base::Contains(web_input_to_password_info_, *element) &&
                   !base::Contains(password_to_username_, *element))) {
//   WebInputElement username_element;
//   WebInputElement password_element;
//   PasswordInfo* password_info = nullptr;
//   if (!element ||
//       !FindPasswordInfoForElement(*element, &username_element,
//                                   &password_element, &password_info)) {
      return false;
    }
    if (was_touch_to_fill_ui_shown_)
     return false;
   was_touch_to_fill_ui_shown_ = true;
 
   GetPasswordManagerDriver()->ShowTouchToFill();
   return true;
 }

static const struct platform_device_id *platform_match_id(
			const struct platform_device_id *id,
			struct platform_device *pdev)
{
	while (id->name[0]) {
		if (strcmp(pdev->name, id->name) == 0) {
			pdev->id_entry = id;
			return id;
		}
		id++;
	}
	return NULL;
}
