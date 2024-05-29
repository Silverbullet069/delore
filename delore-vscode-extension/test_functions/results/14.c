gst_asf_demux_mark_discont (GstASFDemux * demux)
{
  guint n;

  GST_DEBUG_OBJECT (demux, "Mark stream discont");

  for (n = 0; n < demux->num_streams; n++)
    demux->stream[n].discont = TRUE;
}


 void GfxImageColorMap::getRGBLine(Guchar *in, unsigned int *out, int length) {
   int i, j;
   Guchar *inp, *tmp_line;
 
    switch (colorSpace->getMode()) {
    case csIndexed:
    case csSeparation:
    tmp_line = (Guchar *) gmalloc (length * nComps2);
//     tmp_line = (Guchar *) gmallocn (length, nComps2);
      for (i = 0; i < length; i++) {
        for (j = 0; j < nComps2; j++) {
  	tmp_line[i * nComps2 + j] = byte_lookup[in[i] * nComps2 + j];
       }
     }
     colorSpace2->getRGBLine(tmp_line, out, length);
     gfree (tmp_line);
     break;
 
   default:
     inp = in;
     for (j = 0; j < length; j++)
       for (i = 0; i < nComps; i++) {
 	*inp = byte_lookup[*inp * nComps + i];
 	inp++;
       }
     colorSpace->getRGBLine(in, out, length);
     break;
   }
 
 }

void PasswordAutofillAgent::TryFixAutofilledForm(
    std::vector<WebFormControlElement>* control_elements) const {
  for (auto& element : *control_elements) {
    const unsigned element_id = element.UniqueRendererFormControlId();
    auto cached_element = autofilled_elements_cache_.find(element_id);
    if (cached_element == autofilled_elements_cache_.end())
      continue;

    const WebString& cached_value = cached_element->second;
    if (cached_value != element.SuggestedValue())
      element.SetSuggestedValue(cached_value);
  }
}


 void InspectorNetworkAgent::DidBlockRequest(
     ExecutionContext* execution_context,
      const ResourceRequest& request,
      DocumentLoader* loader,
      const FetchInitiatorInfo& initiator_info,
    ResourceRequestBlockedReason reason) {
//     ResourceRequestBlockedReason reason,
//     Resource::Type resource_type) {
    unsigned long identifier = CreateUniqueIdentifier();
//   InspectorPageAgent::ResourceType type =
//       InspectorPageAgent::ToResourceType(resource_type);
// 
    WillSendRequestInternal(execution_context, identifier, loader, request,
                          ResourceResponse(), initiator_info);
//                           ResourceResponse(), initiator_info, type);
  
    String request_id = IdentifiersFactory::RequestId(identifier);
    String protocol_reason = BuildBlockedReason(reason);
   GetFrontend()->loadingFailed(
       request_id, MonotonicallyIncreasingTime(),
       InspectorPageAgent::ResourceTypeJson(
           resources_data_->GetResourceType(request_id)),
       String(), false, protocol_reason);
 }

LayoutUnit LayoutBlockFlow::adjustLogicalRightOffsetForLine(LayoutUnit offsetFromFloats, bool applyTextIndent) const
{
    LayoutUnit right = offsetFromFloats;

    if (applyTextIndent && !style()->isLeftToRightDirection())
        right -= textIndentOffset();

    return right;
}
