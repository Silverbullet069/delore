eXosip_reset_transports (struct eXosip_t *excontext)
{
  int i = OSIP_WRONG_STATE;

  if (excontext->eXtl_transport.tl_reset)
    i = excontext->eXtl_transport.tl_reset (excontext);
  return i;
}


 report_error (const char *format, ...)
 #else
 report_error (format, va_alist)
      const char *format;
      va_dcl
 #endif
 {
   va_list args;
 
   error_prolog (1);
 
   SH_VA_START (args, format);
 
   vfprintf (stderr, format, args);
   fprintf (stderr, "\n");
  
    va_end (args);
    if (exit_immediately_on_error)
    exit_shell (1);
//     {
//       if (last_command_exit_value == 0)
// 	last_command_exit_value = 1;
//       exit_shell (last_command_exit_value);
//     }
  }

static int wdm_manage_power(struct usb_interface *intf, int on)
{
	 
	int rv = usb_autopm_get_interface(intf);
	if (rv < 0)
		goto err;

	intf->needs_remote_wakeup = on;
	usb_autopm_put_interface(intf);
err:
	return rv;
}


 init_ctx_new(OM_uint32 *minor_status,
 	     spnego_gss_cred_id_t spcred,
 	     gss_ctx_id_t *ctx,
 	     send_token_flag *tokflag)
 {
  	OM_uint32 ret;
  	spnego_gss_ctx_id_t sc = NULL;
  
	sc = create_spnego_ctx();
// 	sc = create_spnego_ctx(1);
  	if (sc == NULL)
  		return GSS_S_FAILURE;
  
 	 
 	ret = get_negotiable_mechs(minor_status, spcred, GSS_C_INITIATE,
 				   &sc->mech_set);
 	if (ret != GSS_S_COMPLETE)
 		goto cleanup;
 
 	 
 	sc->internal_mech = &sc->mech_set->elements[0];
 
 	if (put_mech_set(sc->mech_set, &sc->DER_mechTypes) < 0) {
  		ret = GSS_S_FAILURE;
  		goto cleanup;
  	}
	 
// 
  	sc->ctx_handle = GSS_C_NO_CONTEXT;
  	*ctx = (gss_ctx_id_t)sc;
  	sc = NULL;
 	*tokflag = INIT_TOKEN_SEND;
 	ret = GSS_S_CONTINUE_NEEDED;
 
 cleanup:
 	release_spnego_ctx(&sc);
 	return ret;
 }

  bool IsSiteMuted(const TabStripModel& tab_strip, const int index) {
    content::WebContents* web_contents = tab_strip.GetWebContentsAt(index);
// 
//    
//    
//    
//   if (!web_contents)
//     return false;
// 
    GURL url = web_contents->GetLastCommittedURL();
  
   if (url.SchemeIs(content::kChromeUIScheme)) {
     return web_contents->IsAudioMuted() &&
            GetTabAudioMutedReason(web_contents) ==
                TabMutedReason::CONTENT_SETTING_CHROME;
   }
 
   Profile* profile =
       Profile::FromBrowserContext(web_contents->GetBrowserContext());
   HostContentSettingsMap* settings =
       HostContentSettingsMapFactory::GetForProfile(profile);
   return settings->GetContentSetting(url, url, CONTENT_SETTINGS_TYPE_SOUND,
                                      std::string()) == CONTENT_SETTING_BLOCK;
 }