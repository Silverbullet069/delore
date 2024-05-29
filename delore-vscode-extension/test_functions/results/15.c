static void des3_ede_decrypt(struct crypto_tfm *tfm, u8 *dst, const u8 *src)
{
	struct des3_ede_sparc64_ctx *ctx = crypto_tfm_ctx(tfm);
	const u64 *K = ctx->decrypt_expkey;

	des3_ede_sparc64_crypt(K, (const u64 *) src, (u64 *) dst);
}


  bool scoped_pixel_buffer_object::Init(CGLContextObj cgl_context,
                                        int size_in_bytes) {
//    
//    
//    
//    
//   if (base::mac::IsOSLeopardOrEarlier()) {
//     return false;
//   }
    cgl_context_ = cgl_context;
    CGLContextObj CGL_MACRO_CONTEXT = cgl_context_;
    glGenBuffersARB(1, &pixel_buffer_object_);
   if (glGetError() == GL_NO_ERROR) {
     glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, pixel_buffer_object_);
     glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, size_in_bytes, NULL,
                     GL_STREAM_READ_ARB);
     glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
     if (glGetError() != GL_NO_ERROR) {
       Release();
     }
   } else {
     cgl_context_ = NULL;
     pixel_buffer_object_ = 0;
   }
   return pixel_buffer_object_ != 0;
 }

void DriveFsHost::OnMountEvent(
    chromeos::disks::DiskMountManager::MountEvent event,
    chromeos::MountError error_code,
    const chromeos::disks::DiskMountManager::MountPointInfo& mount_info) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  if (!mount_state_) {
    return;
  }
  if (!mount_state_->OnMountEvent(event, error_code, mount_info)) {
    Unmount();
  }
}


static inline int bitmap_position_extended(const unsigned char *sha1)
{
	khash_sha1_pos *positions = bitmap_git.ext_index.positions;
	khiter_t pos = kh_get_sha1_pos(positions, sha1);

	if (pos < kh_end(positions)) {
		int bitmap_pos = kh_value(positions, pos);
		return bitmap_pos + bitmap_git.pack->num_objects;
	}

	return -1;
}


 static int store_icy(URLContext *h, int size)
  {
      HTTPContext *s = h->priv_data;
       
    int remaining = s->icy_metaint - s->icy_data_read;
//     uint64_t remaining;
  
    if (remaining < 0)
//     if (s->icy_metaint < s->icy_data_read)
          return AVERROR_INVALIDDATA;
//     remaining = s->icy_metaint - s->icy_data_read;
  
      if (!remaining) {
           
         uint8_t ch;
         int len = http_read_stream_all(h, &ch, 1);
         if (len < 0)
             return len;
         if (ch > 0) {
             char data[255 * 16 + 1];
             int ret;
             len = ch * 16;
             ret = http_read_stream_all(h, data, len);
             if (ret < 0)
                 return ret;
             data[len + 1] = 0;
             if ((ret = av_opt_set(s, "icy_metadata_packet", data, 0)) < 0)
                 return ret;
             update_metadata(s, data);
         }
         s->icy_data_read = 0;
         remaining        = s->icy_metaint;
     }
 
     return FFMIN(size, remaining);
 }