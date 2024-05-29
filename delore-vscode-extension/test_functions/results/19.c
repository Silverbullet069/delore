 void BrowserEventRouter::TabDetachedAt(TabContents* contents, int index) {
  if (!GetTabEntry(contents->web_contents())) {
// void BrowserEventRouter::TabDetachedAt(WebContents* contents, int index) {
//   if (!GetTabEntry(contents)) {
      return;
    }
  
    scoped_ptr<ListValue> args(new ListValue());
  args->Append(Value::CreateIntegerValue(
      ExtensionTabUtil::GetTabId(contents->web_contents())));
//   args->Append(Value::CreateIntegerValue(ExtensionTabUtil::GetTabId(contents)));
  
    DictionaryValue* object_args = new DictionaryValue();
    object_args->Set(tab_keys::kOldWindowIdKey, Value::CreateIntegerValue(
      ExtensionTabUtil::GetWindowIdOfTab(contents->web_contents())));
//       ExtensionTabUtil::GetWindowIdOfTab(contents)));
    object_args->Set(tab_keys::kOldPositionKey, Value::CreateIntegerValue(
        index));
    args->Append(object_args);
  
  DispatchEvent(contents->profile(), events::kOnTabDetached, args.Pass(),
//   Profile* profile = Profile::FromBrowserContext(contents->GetBrowserContext());
//   DispatchEvent(profile, events::kOnTabDetached, args.Pass(),
                  EventRouter::USER_GESTURE_UNKNOWN);
  }

   void VerifyDailyContentLengthPrefLists(
//    
//   void VerifyDailyDataSavingContentLengthPrefLists(
        const int64* original_values, size_t original_count,
        const int64* received_values, size_t received_count,
        const int64* original_with_data_reduction_proxy_enabled_values,
       size_t original_with_data_reduction_proxy_enabled_count,
       const int64* received_with_data_reduction_proxy_enabled_values,
       size_t received_with_data_reduction_proxy_count,
       const int64* original_via_data_reduction_proxy_values,
       size_t original_via_data_reduction_proxy_count,
       const int64* received_via_data_reduction_proxy_values,
       size_t received_via_data_reduction_proxy_count) {
     VerifyPrefList(prefs::kDailyHttpOriginalContentLength,
                    original_values, original_count);
     VerifyPrefList(prefs::kDailyHttpReceivedContentLength,
                    received_values, received_count);
     VerifyPrefList(
         prefs::kDailyOriginalContentLengthWithDataReductionProxyEnabled,
         original_with_data_reduction_proxy_enabled_values,
         original_with_data_reduction_proxy_enabled_count);
     VerifyPrefList(
         prefs::kDailyContentLengthWithDataReductionProxyEnabled,
         received_with_data_reduction_proxy_enabled_values,
         received_with_data_reduction_proxy_count);
     VerifyPrefList(
         prefs::kDailyOriginalContentLengthViaDataReductionProxy,
         original_via_data_reduction_proxy_values,
         original_via_data_reduction_proxy_count);
     VerifyPrefList(
         prefs::kDailyContentLengthViaDataReductionProxy,
         received_via_data_reduction_proxy_values,
          received_via_data_reduction_proxy_count);
    }

 void SetLeftUnavailable() {
    mbptr_->left_available = 0;
 for (int p = 0; p < num_planes_; p++)
 for (int i = -1; i < block_size_; ++i)
        data_ptr_[p][stride_ * i - 1] = 129;
 }


static int find_snapshot_by_id_or_name(BlockDriverState *bs,
                                       const char *id_or_name)
{
    int ret;

    ret = find_snapshot_by_id_and_name(bs, id_or_name, NULL);
    if (ret >= 0) {
        return ret;
    }
    return find_snapshot_by_id_and_name(bs, NULL, id_or_name);
}


  off_t HFSForkReadStream::Seek(off_t offset, int whence) {
    DCHECK_EQ(SEEK_SET, whence);
    DCHECK_GE(offset, 0);
  DCHECK_LT(static_cast<uint64_t>(offset), fork_.logicalSize);
//   DCHECK(offset == 0 || static_cast<uint64_t>(offset) < fork_.logicalSize);
    size_t target_block = offset / hfs_->block_size();
    size_t block_count = 0;
    for (size_t i = 0; i < arraysize(fork_.extents); ++i) {
     const HFSPlusExtentDescriptor* extent = &fork_.extents[i];
 
     if (extent->startBlock == 0 && extent->blockCount == 0)
       break;
 
     base::CheckedNumeric<size_t> new_block_count(block_count);
     new_block_count += extent->blockCount;
     if (!new_block_count.IsValid()) {
       DLOG(ERROR) << "Seek offset block count overflows";
       return false;
     }
 
     if (target_block < new_block_count.ValueOrDie()) {
       if (current_extent_ != i) {
         read_current_extent_ = false;
         current_extent_ = i;
       }
       auto iterator_block_offset =
           base::CheckedNumeric<size_t>(block_count) * hfs_->block_size();
       if (!iterator_block_offset.IsValid()) {
         DLOG(ERROR) << "Seek block offset overflows";
         return false;
       }
       fork_logical_offset_ = offset;
       return offset;
     }
 
     block_count = new_block_count.ValueOrDie();
   }
   return -1;
 }