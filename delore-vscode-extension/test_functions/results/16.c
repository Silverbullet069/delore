static inline int snd_compr_get_poll(struct snd_compr_stream *stream)
{
	if (stream->direction == SND_COMPRESS_PLAYBACK)
		return POLLOUT | POLLWRNORM;
	else
		return POLLIN | POLLRDNORM;
}


 txid_current_snapshot(PG_FUNCTION_ARGS)
 {
 	TxidSnapshot *snap;
 	uint32		nxip,
 				i,
 				size;
 	TxidEpoch	state;
 	Snapshot	cur;
 
 	cur = GetActiveSnapshot();
 	if (cur == NULL)
 		elog(ERROR, "no active snapshot set");
  
  	load_xid_epoch(&state);
  
// 	 
// 	StaticAssertStmt(MAX_BACKENDS * 2 <= TXID_SNAPSHOT_MAX_NXIP,
// 					 "possible overflow in txid_current_snapshot()");
// 
  	 
  	nxip = cur->xcnt;
  	size = TXID_SNAPSHOT_SIZE(nxip);
 	snap = palloc(size);
 	SET_VARSIZE(snap, size);
 
 	 
 	snap->xmin = convert_xid(cur->xmin, &state);
 	snap->xmax = convert_xid(cur->xmax, &state);
 	snap->nxip = nxip;
 	for (i = 0; i < nxip; i++)
 		snap->xip[i] = convert_xid(cur->xip[i], &state);
 
 	 
 	sort_snapshot(snap);
 
 	PG_RETURN_POINTER(snap);
 }

void NavigationControllerImpl::UpdateVirtualURLToURL(
    NavigationEntryImpl* entry, const GURL& new_url) {
  GURL new_virtual_url(new_url);
  if (BrowserURLHandlerImpl::GetInstance()->ReverseURLRewrite(
          &new_virtual_url, entry->GetVirtualURL(), browser_context_)) {
    entry->SetVirtualURL(new_virtual_url);
  }
}


void nfs_invalidate_atime(struct inode *inode)
{
	spin_lock(&inode->i_lock);
	NFS_I(inode)->cache_validity |= NFS_INO_INVALID_ATIME;
	spin_unlock(&inode->i_lock);
}


 static void copy_xauthority(void) {
 	char *src = RUN_XAUTHORITY_FILE ;
 	char *dest;
 	if (asprintf(&dest, "%s/.Xauthority", cfg.homedir) == -1)
 		errExit("asprintf");
 	
 	if (is_link(dest)) {
 		fprintf(stderr, "Error: %s is a symbolic link\n", dest);
  		exit(1);
  	}
  
	pid_t child = fork();
	if (child < 0)
		errExit("fork");
	if (child == 0) {
		drop_privs(0);
		int rv = copy_file(src, dest, getuid(), getgid(), S_IRUSR | S_IWUSR);
		if (rv)
			fprintf(stderr, "Warning: cannot transfer .Xauthority in private home directory\n");
		else {
			fs_logger2("clone", dest);
		}
		_exit(0);
	}
	waitpid(child, NULL, 0);
// 	copy_file_as_user(src, dest, getuid(), getgid(), S_IRUSR | S_IWUSR);
// 	fs_logger2("clone", dest);
  	
  	unlink(src);
 }