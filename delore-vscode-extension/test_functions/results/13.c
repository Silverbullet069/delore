 void utf32_to_utf8(const char32_t* src, size_t src_len, char* dst)
// void utf32_to_utf8(const char32_t* src, size_t src_len, char* dst, size_t dst_len)
  {
      if (src == NULL || src_len == 0 || dst == NULL) {
          return;
  }
 
  const char32_t *cur_utf32 = src;
  const char32_t *end_utf32 = src + src_len;
 
      char *cur = dst;
      while (cur_utf32 < end_utf32) {
          size_t len = utf32_codepoint_utf8_length(*cur_utf32);
//         LOG_ALWAYS_FATAL_IF(dst_len < len, "%zu < %zu", dst_len, len);
          utf32_codepoint_to_utf8((uint8_t *)cur, *cur_utf32++, len);
          cur += len;
//         dst_len -= len;
      }
//     LOG_ALWAYS_FATAL_IF(dst_len < 1, "dst_len < 1: %zu < 1", dst_len);
      *cur = '\0';
  }

void WriteVirtIODeviceRegister(ULONG_PTR ulRegister, u32 ulValue)
{
    DPrintf(6, ("[%s]R[%x]=%x\n", __FUNCTION__, (ULONG)ulRegister, ulValue) );

    NdisRawWritePortUlong(ulRegister, ulValue);
}


static u32 get_acqseq(void)
{
	u32 res;
	static atomic_t acqseq;

	do {
		res = atomic_inc_return(&acqseq);
	} while (!res);
	return res;
}


 static int copy_verifier_state(struct bpf_verifier_state *dst_state,
 			       const struct bpf_verifier_state *src)
 {
 	struct bpf_func_state *dst;
 	int i, err;
 
 	 
 	for (i = src->curframe + 1; i <= dst_state->curframe; i++) {
  		free_func_state(dst_state->frame[i]);
  		dst_state->frame[i] = NULL;
  	}
// 	dst_state->speculative = src->speculative;
  	dst_state->curframe = src->curframe;
  	for (i = 0; i <= src->curframe; i++) {
  		dst = dst_state->frame[i];
 		if (!dst) {
 			dst = kzalloc(sizeof(*dst), GFP_KERNEL);
 			if (!dst)
 				return -ENOMEM;
 			dst_state->frame[i] = dst;
 		}
 		err = copy_func_state(dst, src->frame[i]);
 		if (err)
 			return err;
 	}
 	return 0;
 }

 static void perf_event_mmap_output(struct perf_event *event,
 				     struct perf_mmap_event *mmap_event)
 {
 	struct perf_output_handle handle;
 	struct perf_sample_data sample;
 	int size = mmap_event->event_id.header.size;
 	int ret;
  
  	perf_event_header__init_id(&mmap_event->event_id.header, &sample, event);
  	ret = perf_output_begin(&handle, event,
				mmap_event->event_id.header.size, 0, 0);
// 				mmap_event->event_id.header.size, 0);
  	if (ret)
  		goto out;
  
 	mmap_event->event_id.pid = perf_event_pid(event, current);
 	mmap_event->event_id.tid = perf_event_tid(event, current);
 
 	perf_output_put(&handle, mmap_event->event_id);
 	__output_copy(&handle, mmap_event->file_name,
 				   mmap_event->file_size);
 
 	perf_event__output_id_sample(event, &handle, &sample);
 
 	perf_output_end(&handle);
 out:
 	mmap_event->event_id.header.size = size;
 }