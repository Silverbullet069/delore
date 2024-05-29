 void NavigatorImpl::DidFailProvisionalLoadWithError(
     RenderFrameHostImpl* render_frame_host,
     const FrameHostMsg_DidFailProvisionalLoadWithError_Params& params) {
   VLOG(1) << "Failed Provisional Load: " << params.url.possibly_invalid_spec()
           << ", error_code: " << params.error_code
           << ", error_description: " << params.error_description
           << ", showing_repost_interstitial: " <<
             params.showing_repost_interstitial
           << ", frame_id: " << render_frame_host->GetRoutingID();
   GURL validated_url(params.url);
   RenderProcessHost* render_process_host = render_frame_host->GetProcess();
   render_process_host->FilterURL(false, &validated_url);
 
   if (net::ERR_ABORTED == params.error_code) {
    FrameTreeNode* root =
        render_frame_host->frame_tree_node()->frame_tree()->root();
    if (root->render_manager()->interstitial_page() != NULL) {
//     if (delegate_ && delegate_->ShowingInterstitialPage()) {
        LOG(WARNING) << "Discarding message during interstitial.";
        return;
      }
 
   }
 
   int expected_pending_entry_id =
       render_frame_host->navigation_handle()
           ? render_frame_host->navigation_handle()->pending_nav_entry_id()
           : 0;
   DiscardPendingEntryIfNeeded(expected_pending_entry_id);
 }

  struct lib_t* MACH0_(get_libs)(struct MACH0_(obj_t)* bin) {
  	struct lib_t *libs;
  	int i;
  
	if (!bin->nlibs)
// 	if (!bin->nlibs) {
  		return NULL;
	if (!(libs = calloc ((bin->nlibs + 1), sizeof(struct lib_t))))
// 	}
// 	if (!(libs = calloc ((bin->nlibs + 1), sizeof(struct lib_t)))) {
  		return NULL;
// 	}
  	for (i = 0; i < bin->nlibs; i++) {
  		strncpy (libs[i].name, bin->libs[i], R_BIN_MACH0_STRING_LENGTH);
  		libs[i].name[R_BIN_MACH0_STRING_LENGTH-1] = '\0';
 		libs[i].last = 0;
 	}
 	libs[i].last = 1;
 	return libs;
 }

static enum TIFFReadDirEntryErr TIFFReadDirEntryCheckRangeShortLong(uint32 value)
{
	if (value>0xFFFF)
		return(TIFFReadDirEntryErrRange);
	else
		return(TIFFReadDirEntryErrOk);
}


 int nntp_add_group(char *line, void *data)
  {
    struct NntpServer *nserv = data;
    struct NntpData *nntp_data = NULL;
  char group[LONG_STRING];
//   char group[LONG_STRING] = "";
    char desc[HUGE_STRING] = "";
    char mod;
    anum_t first, last;
  
    if (!nserv || !line)
      return 0;
  
  if (sscanf(line, "%s " ANUM " " ANUM " %c %[^\n]", group, &last, &first, &mod, desc) < 4)
//    
//   if (sscanf(line, "%1023s " ANUM " " ANUM " %c %8191[^\n]", group, &last, &first, &mod, desc) < 4)
//   {
//     mutt_debug(4, "Cannot parse server line: %s\n", line);
      return 0;
//   }
  
    nntp_data = nntp_data_find(nserv, group);
    nntp_data->deleted = false;
   nntp_data->first_message = first;
   nntp_data->last_message = last;
   nntp_data->allowed = (mod == 'y') || (mod == 'm');
   mutt_str_replace(&nntp_data->desc, desc);
   if (nntp_data->newsrc_ent || nntp_data->last_cached)
     nntp_group_unread_stat(nntp_data);
   else if (nntp_data->last_message && nntp_data->first_message <= nntp_data->last_message)
     nntp_data->unread = nntp_data->last_message - nntp_data->first_message + 1;
   else
     nntp_data->unread = 0;
   return 0;
 }

 static inline int verify_replay(struct xfrm_usersa_info *p,
  				struct nlattr **attrs)
  {
  	struct nlattr *rt = attrs[XFRMA_REPLAY_ESN_VAL];
// 	struct xfrm_replay_state_esn *rs;
  
	if ((p->flags & XFRM_STATE_ESN) && !rt)
		return -EINVAL;
// 	if (p->flags & XFRM_STATE_ESN) {
// 		if (!rt)
// 			return -EINVAL;
// 
// 		rs = nla_data(rt);
// 
// 		if (rs->bmp_len > XFRMA_REPLAY_ESN_MAX / sizeof(rs->bmp[0]) / 8)
// 			return -EINVAL;
// 
// 		if (nla_len(rt) < xfrm_replay_state_esn_len(rs) &&
// 		    nla_len(rt) != sizeof(*rs))
// 			return -EINVAL;
// 	}
  
  	if (!rt)
  		return 0;
 
 	if (p->id.proto != IPPROTO_ESP)
 		return -EINVAL;
 
 	if (p->replay_window != 0)
 		return -EINVAL;
 
 	return 0;
 }