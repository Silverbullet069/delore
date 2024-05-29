  ExtensionInfoBar::ExtensionInfoBar(ExtensionInfoBarDelegate* delegate)
    : InfoBar(delegate),
//     : InfoBarView(delegate),
        delegate_(delegate),
        ALLOW_THIS_IN_INITIALIZER_LIST(tracker_(this)) {
    delegate_->set_observer(this);
 
   ExtensionHost* extension_host = delegate_->extension_host();
 
   gfx::Size sz = extension_host->view()->GetPreferredSize();
   if (sz.height() > 0)
     sz.set_height(sz.height() + 1);
   set_target_height(sz.height());
 
   SetupIconAndMenu();
 
   extension_host->view()->SetContainer(this);
   extension_host->view()->set_parent_owned(false);
   AddChildView(extension_host->view());
 }

 static ssize_t ucma_write(struct file *filp, const char __user *buf,
 			  size_t len, loff_t *pos)
 {
 	struct ucma_file *file = filp->private_data;
  	struct rdma_ucm_cmd_hdr hdr;
  	ssize_t ret;
  
// 	if (WARN_ON_ONCE(!ib_safe_file_access(filp)))
// 		return -EACCES;
// 
  	if (len < sizeof(hdr))
  		return -EINVAL;
  
 	if (copy_from_user(&hdr, buf, sizeof(hdr)))
 		return -EFAULT;
 
 	if (hdr.cmd >= ARRAY_SIZE(ucma_cmd_table))
 		return -EINVAL;
 
 	if (hdr.in + sizeof(hdr) > len)
 		return -EINVAL;
 
 	if (!ucma_cmd_table[hdr.cmd])
 		return -ENOSYS;
 
 	ret = ucma_cmd_table[hdr.cmd](file, buf + sizeof(hdr), hdr.in, hdr.out);
 	if (!ret)
 		ret = len;
 
 	return ret;
 }

  chrm_modification_init(chrm_modification *me, png_modifier *pm,
   PNG_CONST color_encoding *encoding)
//    const color_encoding *encoding)
  {
     CIE_color white = white_point(encoding);
  
   
    me->encoding = encoding;
 
   
    me->wx = fix(chromaticity_x(white));
    me->wy = fix(chromaticity_y(white));
 
    me->rx = fix(chromaticity_x(encoding->red));
    me->ry = fix(chromaticity_y(encoding->red));
    me->gx = fix(chromaticity_x(encoding->green));
    me->gy = fix(chromaticity_y(encoding->green));
    me->bx = fix(chromaticity_x(encoding->blue));
    me->by = fix(chromaticity_y(encoding->blue));
 
    modification_init(&me->this);
    me->this.chunk = CHUNK_cHRM;
    me->this.modify_fn = chrm_modify;
    me->this.add = CHUNK_PLTE;
    me->this.next = pm->modifications;
    pm->modifications = &me->this;
 }

static int can_open_delegated(struct nfs_delegation *delegation, fmode_t fmode)
{
	if (delegation == NULL)
		return 0;
	if ((delegation->type & fmode) != fmode)
		return 0;
	if (test_bit(NFS_DELEGATION_RETURNING, &delegation->flags))
		return 0;
	nfs_mark_delegation_referenced(delegation);
	return 1;
}


 PHP_METHOD(Phar, createDefaultStub)
 {
 	char *index = NULL, *webindex = NULL, *error;
         zend_string *stub;
         size_t index_len = 0, webindex_len = 0;
  
       if (zend_parse_parameters(ZEND_NUM_ARGS(), "|ss", &index, &index_len, &webindex, &webindex_len) == FAILURE) {
//        if (zend_parse_parameters(ZEND_NUM_ARGS(), "|pp", &index, &index_len, &webindex, &webindex_len) == FAILURE) {
                 return;
         }
  
 	stub = phar_create_default_stub(index, webindex, &error);
 
 	if (error) {
 		zend_throw_exception_ex(phar_ce_PharException, 0, "%s", error);
 		efree(error);
 		return;
 	}
 	RETURN_NEW_STR(stub);
 }