
// vul

static int r3d_read_rdvo(AVFormatContext *s, Atom *atom)
{
    R3DContext *r3d = s->priv_data;
    AVStream *st = s->streams[0];
    int i;
    r3d->video_offsets_count = (atom->size - 8) / 4;
    r3d->video_offsets = av_malloc(atom->size);
    if (!r3d->video_offsets)
        return AVERROR(ENOMEM);
    for (i = 0; i < r3d->video_offsets_count; i++) {
        r3d->video_offsets[i] = avio_rb32(s->pb);
        if (!r3d->video_offsets[i]) {
            r3d->video_offsets_count = i;
            break;
        }
        av_dlog(s, "video offset %d: %#x\
", i, r3d->video_offsets[i]);
    }
    if (st->r_frame_rate.num)
        st->duration = av_rescale_q(r3d->video_offsets_count,
                                    (AVRational){st->r_frame_rate.den,
                                                 st->r_frame_rate.num},
                                    st->time_base);
    av_dlog(s, "duration %"PRId64"\
", st->duration);
    return 0;
}
static void check_lowpass_line(int depth){
    LOCAL_ALIGNED_32(uint8_t, src,     [SRC_SIZE]);
    LOCAL_ALIGNED_32(uint8_t, dst_ref, [WIDTH_PADDED]);
    LOCAL_ALIGNED_32(uint8_t, dst_new, [WIDTH_PADDED]);
    int w = WIDTH;
    int mref = WIDTH_PADDED * -1;
    int pref = WIDTH_PADDED;
    int i, depth_byte;
    InterlaceContext s;
    declare_func(void, uint8_t *dstp, ptrdiff_t linesize, const uint8_t *srcp,
                 ptrdiff_t mref, ptrdiff_t pref, int clip_max);
    s.lowpass = 1;
    s.lowpass = VLPF_LIN;
    depth_byte = depth >> 3;
    w /= depth_byte;
    memset(src,     0, SRC_SIZE);
    memset(dst_ref, 0, WIDTH_PADDED);
    memset(dst_new, 0, WIDTH_PADDED);
    randomize_buffers(src, SRC_SIZE);
    ff_interlace_init(&s, depth);
    if (check_func(s.lowpass_line, "lowpass_line_%d", depth)) {
        for (i = 0; i < 32; i++) { /* simulate crop */
            call_ref(dst_ref, w, src + WIDTH_PADDED, mref - i*depth_byte, pref, 0);
            call_new(dst_new, w, src + WIDTH_PADDED, mref - i*depth_byte, pref, 0);
            if (memcmp(dst_ref, dst_new, WIDTH - i))
                fail();
        }
        bench_new(dst_new, w, src + WIDTH_PADDED, mref, pref, 0);
    }
}
void assert_avoptions(AVDictionary *m)
{
    AVDictionaryEntry *t;
    if ((t = av_dict_get(m, "", NULL, AV_DICT_IGNORE_SUFFIX))) {
        av_log(NULL, AV_LOG_FATAL, "Option %s not found.\
", t->key);
        exit(1);
    }
}
static void vp6_parse_coeff_huffman(VP56Context *s)
{
    VP56Model *model = s->modelp;
    uint8_t *permute = s->scantable.permutated;
    VLC *vlc_coeff;
    int coeff, sign, coeff_idx;
    int b, cg, idx;
    int pt = 0;    /* plane type (0 for Y, 1 for U or V) */
    for (b=0; b<6; b++) {
        int ct = 0;    /* code type */
        if (b > 3) pt = 1;
        vlc_coeff = &s->dccv_vlc[pt];
        for (coeff_idx=0; coeff_idx<64; ) {
            int run = 1;
            if (coeff_idx<2 && s->nb_null[coeff_idx][pt]) {
                s->nb_null[coeff_idx][pt]--;
                if (coeff_idx)
                    break;
            } else {
                if (get_bits_count(&s->gb) >= s->gb.size_in_bits)
                    return;
                coeff = get_vlc2(&s->gb, vlc_coeff->table, 9, 3);
                if (coeff == 0) {
                    if (coeff_idx) {
                        int pt = (coeff_idx >= 6);
                        run += get_vlc2(&s->gb, s->runv_vlc[pt].table, 9, 3);
                        if (run >= 9)
                            run += get_bits(&s->gb, 6);
                    } else
                        s->nb_null[0][pt] = vp6_get_nb_null(s);
                    ct = 0;
                } else if (coeff == 11) {  /* end of block */
                    if (coeff_idx == 1)    /* first AC coeff ? */
                        s->nb_null[1][pt] = vp6_get_nb_null(s);
                    break;
                } else {
                    int coeff2 = vp56_coeff_bias[coeff];
                    if (coeff > 4)
                        coeff2 += get_bits(&s->gb, coeff <= 9 ? coeff - 4 : 11);
                    ct = 1 + (coeff2 > 1);
                    sign = get_bits1(&s->gb);
                    coeff2 = (coeff2 ^ -sign) + sign;
                    if (coeff_idx)
                        coeff2 *= s->dequant_ac;
                    idx = model->coeff_index_to_pos[coeff_idx];
                    s->block_coeff[b][permute[idx]] = coeff2;
                }
            }
            coeff_idx+=run;
            cg = FFMIN(vp6_coeff_groups[coeff_idx], 3);
            vlc_coeff = &s->ract_vlc[pt][ct][cg];
        }
    }
}
static void quantize_mantissas(AC3EncodeContext *s)
{
    int blk, ch;
    for (blk = 0; blk < AC3_MAX_BLOCKS; blk++) {
        AC3Block *block = &s->blocks[blk];
        s->mant1_cnt  = s->mant2_cnt  = s->mant4_cnt  = 0;
        s->qmant1_ptr = s->qmant2_ptr = s->qmant4_ptr = NULL;
        for (ch = 0; ch < s->channels; ch++) {
            quantize_mantissas_blk_ch(s, block->fixed_coef[ch], block->exp_shift[ch],
                                      block->exp[ch], block->bap[ch],
                                      block->qmant[ch], s->nb_coefs[ch]);
        }
}
static int iv_alloc_frames(Indeo3DecodeContext *s)
{
    int luma_width    = (s->width           + 3) & ~3,
        luma_height   = (s->height          + 3) & ~3,
        chroma_width  = ((luma_width  >> 2) + 3) & ~3,
        chroma_height = ((luma_height >> 2) + 3) & ~3,
        luma_pixels   = luma_width   * luma_height,
        chroma_pixels = chroma_width * chroma_height,
        i;
    unsigned int bufsize = luma_pixels * 2 + luma_width * 3 +
                          (chroma_pixels   + chroma_width) * 4;
    if(!(s->buf = av_malloc(bufsize)))
        return AVERROR(ENOMEM);
    s->iv_frame[0].y_w = s->iv_frame[1].y_w = luma_width;
    s->iv_frame[0].y_h = s->iv_frame[1].y_h = luma_height;
    s->iv_frame[0].uv_w = s->iv_frame[1].uv_w = chroma_width;
    s->iv_frame[0].uv_h = s->iv_frame[1].uv_h = chroma_height;
    s->iv_frame[0].Ybuf = s->buf + luma_width;
    i = luma_pixels + luma_width * 2;
    s->iv_frame[1].Ybuf = s->buf + i;
    i += (luma_pixels + luma_width);
    s->iv_frame[0].Ubuf = s->buf + i;
    i += (chroma_pixels + chroma_width);
    s->iv_frame[1].Ubuf = s->buf + i;
    i += (chroma_pixels + chroma_width);
    s->iv_frame[0].Vbuf = s->buf + i;
    i += (chroma_pixels + chroma_width);
    s->iv_frame[1].Vbuf = s->buf + i;
    for(i = 1; i <= luma_width; i++)
        s->iv_frame[0].Ybuf[-i] = s->iv_frame[1].Ybuf[-i] =
            s->iv_frame[0].Ubuf[-i] = 0x80;
    for(i = 1; i <= chroma_width; i++) {
        s->iv_frame[1].Ubuf[-i] = 0x80;
        s->iv_frame[0].Vbuf[-i] = 0x80;
        s->iv_frame[1].Vbuf[-i] = 0x80;
        s->iv_frame[1].Vbuf[chroma_pixels+i-1] = 0x80;
    }
    return 0;
}

// Non-vul

static void v4l2_free_buffer(void *opaque, uint8_t *unused)
{
    V4L2Buffer* avbuf = opaque;
    V4L2m2mContext *s = buf_to_m2mctx(avbuf);
    if (atomic_fetch_sub(&avbuf->context_refcount, 1) == 1) {
        atomic_fetch_sub_explicit(&s->refcount, 1, memory_order_acq_rel);
        if (s->reinit) {  
            if (!atomic_load(&s->refcount))
                sem_post(&s->refsync);
        } else if (avbuf->context->streamon)
            ff_v4l2_buffer_enqueue(avbuf);
        av_buffer_unref(&avbuf->context_ref);
    }
}

int av_opencl_buffer_write(cl_mem dst_cl_buf, uint8_t *src_buf, size_t buf_size)
{
    cl_int status;
    void *mapped = clEnqueueMapBuffer(gpu_env.command_queue, dst_cl_buf,
                                      CL_TRUE,CL_MAP_WRITE, 0, sizeof(uint8_t) * buf_size,
                                      0, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
        av_log(&openclutils, AV_LOG_ERROR, "Could not map OpenCL buffer: %s\
", opencl_errstr(status));
        return AVERROR_EXTERNAL;
    }
    memcpy(mapped, src_buf, buf_size);
    status = clEnqueueUnmapMemObject(gpu_env.command_queue, dst_cl_buf, mapped, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        av_log(&openclutils, AV_LOG_ERROR, "Could not unmap OpenCL buffer: %s\
", opencl_errstr(status));
        return AVERROR_EXTERNAL;
    }
    return 0;
}

void ff_af_queue_init(AVCodecContext *avctx, AudioFrameQueue *afq)
{
    afq->avctx             = avctx;
    afq->next_pts          = AV_NOPTS_VALUE;
    afq->remaining_delay   = avctx->delay;
    afq->remaining_samples = avctx->delay;
    afq->frame_queue       = NULL;
}

int av_packet_ref(AVPacket *dst, AVPacket *src)
{
    int ret;
    ret = av_packet_copy_props(dst, src);
    if (ret < 0)
        return ret;
    if (!src->buf) {
        ret = packet_alloc(&dst->buf, src->size);
        if (ret < 0)
            goto fail;
        memcpy(dst->buf->data, src->data, src->size);
    } else
        dst->buf = av_buffer_ref(src->buf);
    dst->size = src->size;
    dst->data = dst->buf->data;
    return 0;
fail:
    av_packet_free_side_data(dst);
    return ret;
}

void ff_float_dsp_init_ppc(AVFloatDSPContext *fdsp, int bit_exact)
{
    if (!(av_get_cpu_flags() & AV_CPU_FLAG_ALTIVEC))
        return;
    fdsp->vector_fmul = ff_vector_fmul_altivec;
    fdsp->vector_fmul_add = ff_vector_fmul_add_altivec;
    fdsp->vector_fmul_reverse = ff_vector_fmul_reverse_altivec;
    if (!bit_exact) {
        fdsp->vector_fmul_window = ff_vector_fmul_window_altivec;
    }
}