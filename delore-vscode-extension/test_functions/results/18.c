void ChromeClientImpl::openPasswordGenerator(HTMLInputElement* input)
{
    ASSERT(isPasswordGenerationEnabled());
    WebInputElement webInput(input);
    m_webView->passwordGeneratorClient()->openPasswordGenerator(webInput);
}


 juniper_atm1_print(netdissect_options *ndo,
                    const struct pcap_pkthdr *h, register const u_char *p)
 {
         int llc_hdrlen;
 
         struct juniper_l2info_t l2info;
 
         l2info.pictype = DLT_JUNIPER_ATM1;
         if (juniper_parse_header(ndo, p, h, &l2info) == 0)
             return l2info.header_len;
 
         p+=l2info.header_len;
 
         if (l2info.cookie[0] == 0x80) {  
             oam_print(ndo, p, l2info.length, ATM_OAM_NOHEC);
              return l2info.header_len;
          }
  
//         ND_TCHECK2(p[0], 3);
          if (EXTRACT_24BITS(p) == 0xfefe03 ||  
              EXTRACT_24BITS(p) == 0xaaaa03) {  
  
             llc_hdrlen = llc_print(ndo, p, l2info.length, l2info.caplen, NULL, NULL);
             if (llc_hdrlen > 0)
                 return l2info.header_len;
         }
 
         if (p[0] == 0x03) {  
             isoclns_print(ndo, p + 1, l2info.length - 1);
              
             return l2info.header_len;
         }
 
         if (ip_heuristic_guess(ndo, p, l2info.length) != 0)  
              return l2info.header_len;
  
  	return l2info.header_len;
// 
// trunc:
// 	ND_PRINT((ndo, "[|juniper_atm1]"));
// 	return l2info.header_len;
  }

  horAcc16(TIFF* tif, uint8* cp0, tmsize_t cc)
  {
  	tmsize_t stride = PredictorState(tif)->stride;
  	uint16* wp = (uint16*) cp0;
  	tmsize_t wc = cc / 2;
  
	assert((cc%(2*stride))==0);
//     if((cc%(2*stride))!=0)
//     {
//         TIFFErrorExt(tif->tif_clientdata, "horAcc16",
//                      "%s", "cc%(2*stride))!=0");
//         return 0;
//     }
  
  	if (wc > stride) {
  		wc -= stride;
 		do {
 			REPEAT4(stride, wp[stride] = (uint16)(((unsigned int)wp[stride] + (unsigned int)wp[0]) & 0xffff); wp++)
  			wc -= stride;
  		} while (wc > 0);
  	}
// 	return 1;
  }

WebContentsViewPort* CreateWebContentsView(
    WebContentsImpl* web_contents,
    WebContentsViewDelegate* delegate,
    RenderViewHostDelegateView** render_view_host_delegate_view) {
  WebContentsViewAura* rv = new WebContentsViewAura(web_contents, delegate);
  *render_view_host_delegate_view = rv;
  return rv;
}


static int nfs4_proc_getattr(struct nfs_server *server, struct nfs_fh *fhandle,
				struct nfs_fattr *fattr, struct nfs4_label *label)
{
	struct nfs4_exception exception = { };
	int err;
	do {
		err = _nfs4_proc_getattr(server, fhandle, fattr, label);
		trace_nfs4_getattr(server, fhandle, fattr, err);
		err = nfs4_handle_exception(server, err,
				&exception);
	} while (exception.retry);
	return err;
}
