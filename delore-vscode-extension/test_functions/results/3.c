PHP_METHOD(Phar, __destruct)
{
	phar_archive_object *phar_obj = (phar_archive_object*)zend_object_store_get_object(getThis() TSRMLS_CC);

	if (phar_obj->arc.archive && phar_obj->arc.archive->is_persistent) {
		zend_hash_del(&PHAR_GLOBALS->phar_persist_map, (const char *) phar_obj->arc.archive, sizeof(phar_obj->arc.archive));
	}
}


 bool DownloadManagerImpl::InterceptDownload(
     const download::DownloadCreateInfo& info) {
   WebContents* web_contents = WebContentsImpl::FromRenderFrameHostID(
       info.render_process_id, info.render_frame_id);
   if (info.is_new_download &&
       info.result ==
           download::DOWNLOAD_INTERRUPT_REASON_SERVER_CROSS_ORIGIN_REDIRECT) {
     if (web_contents) {
       std::vector<GURL> url_chain(info.url_chain);
       GURL url = url_chain.back();
       url_chain.pop_back();
       NavigationController::LoadURLParams params(url);
       params.has_user_gesture = info.has_user_gesture;
       params.referrer = Referrer(
            info.referrer_url, Referrer::NetReferrerPolicyToBlinkReferrerPolicy(
                                   info.referrer_policy));
        params.redirect_chain = url_chain;
//       params.frame_tree_node_id =
//           RenderFrameHost::GetFrameTreeNodeIdForRoutingId(
//               info.render_process_id, info.render_frame_id);
        web_contents->GetController().LoadURLWithParams(params);
      }
      if (info.request_handle)
       info.request_handle->CancelRequest(false);
     return true;
   }
   if (!delegate_ ||
       !delegate_->InterceptDownloadIfApplicable(
           info.url(), info.mime_type, info.request_origin, web_contents)) {
     return false;
   }
   if (info.request_handle)
     info.request_handle->CancelRequest(false);
   return true;
 }

int tls1_new(SSL *s)
	{
	if (!ssl3_new(s)) return(0);
	s->method->ssl_clear(s);
	return(1);
	}


static void iscsi_release_extra_responses(struct iscsi_param_list *param_list)
{
	struct iscsi_extra_response *er, *er_tmp;

	list_for_each_entry_safe(er, er_tmp, &param_list->extra_response_list,
			er_list) {
		list_del(&er->er_list);
		kfree(er);
	}
}


static void anyCallbackFunctionOptionalAnyArgAttributeAttributeGetterCallback(v8::Local<v8::String>, const v8::PropertyCallbackInfo<v8::Value>& info)
{
    TRACE_EVENT_SET_SAMPLING_STATE("Blink", "DOMGetter");
    TestObjectPythonV8Internal::anyCallbackFunctionOptionalAnyArgAttributeAttributeGetter(info);
    TRACE_EVENT_SET_SAMPLING_STATE("V8", "V8Execution");
}
