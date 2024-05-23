void sco_disconn_cfm(struct hci_conn *hcon, __u8 reason)
{
	BT_DBG("hcon %p reason %d", hcon, reason);

	sco_conn_del(hcon, bt_to_errno(reason));
}


static int fcnvsd(struct sh_fpu_soft_struct *fregs, int n)
{
	FP_DECL_EX;
	FP_DECL_S(Fn);
	FP_DECL_D(Fr);
	UNPACK_S(Fn, FPUL);
	FP_CONV(D, S, 2, 1, Fr, Fn);
	PACK_D(DRn, Fr);
	return 0;
}


 static ssize_t aio_setup_single_vector(struct kiocb *kiocb,
 				       int rw, char __user *buf,
  				       unsigned long *nr_segs,
  				       struct iovec *iovec)
  {
	if (unlikely(!access_ok(!rw, buf, kiocb->ki_nbytes)))
// 	size_t len = kiocb->ki_nbytes;
// 
// 	if (len > MAX_RW_COUNT)
// 		len = MAX_RW_COUNT;
// 
// 	if (unlikely(!access_ok(!rw, buf, len)))
  		return -EFAULT;
  
  	iovec->iov_base = buf;
	iovec->iov_len = kiocb->ki_nbytes;
// 	iovec->iov_len = len;
  	*nr_segs = 1;
  	return 0;
  }

void ClientUsageTracker::GetGlobalUsage(GlobalUsageCallback* callback) {
  if (global_usage_retrieved_) {
    callback->Run(type_, global_usage_, GetCachedGlobalUnlimitedUsage());
    delete callback;
    return;
  }
  DCHECK(!global_usage_callback_.HasCallbacks());
  global_usage_callback_.Add(callback);
  global_usage_task_ = new GatherGlobalUsageTask(tracker_, client_);
  global_usage_task_->Start();
}


static int btrfs_fiemap(struct inode *inode, struct fiemap_extent_info *fieinfo,
		__u64 start, __u64 len)
{
	int	ret;

	ret = fiemap_check_flags(fieinfo, BTRFS_FIEMAP_FLAGS);
	if (ret)
		return ret;

	return extent_fiemap(inode, fieinfo, start, len, btrfs_get_extent_fiemap);
}