
static void test_init(TestData *d)
{
    QPCIBus *bus;
    QTestState *qs;
    char *s;
    s = g_strdup_printf("-machine q35 %s %s",
                        d->noreboot ? "" : "-global ICH9-LPC.noreboot=false",
                        !d->args ? "" : d->args);
    qs = qtest_start(s);
    qtest_irq_intercept_in(qs, "ioapic");
    g_free(s);
    bus = qpci_init_pc(NULL);
    d->dev = qpci_device_find(bus, QPCI_DEVFN(0x1f, 0x00));
    g_assert(d->dev != NULL);
    qpci_device_enable(d->dev);
    /* set ACPI PM I/O space base address */
    qpci_config_writel(d->dev, ICH9_LPC_PMBASE, PM_IO_BASE_ADDR | 0x1);
    /* enable ACPI I/O */
    qpci_config_writeb(d->dev, ICH9_LPC_ACPI_CTRL, 0x80);
    /* set Root Complex BAR */
    qpci_config_writel(d->dev, ICH9_LPC_RCBA, RCBA_BASE_ADDR | 0x1);
    d->tco_io_base = qpci_legacy_iomap(d->dev, PM_IO_BASE_ADDR + 0x60);
}
static int xen_9pfs_connect(struct XenDevice *xendev)
{
    int i;
    Xen9pfsDev *xen_9pdev = container_of(xendev, Xen9pfsDev, xendev);
    V9fsState *s = &xen_9pdev->state;
    QemuOpts *fsdev;
    if (xenstore_read_fe_int(&xen_9pdev->xendev, "num-rings",
                             &xen_9pdev->num_rings) == -1 ||
        xen_9pdev->num_rings > MAX_RINGS || xen_9pdev->num_rings < 1) {
        return -1;
    }
    xen_9pdev->rings = g_malloc0(xen_9pdev->num_rings * sizeof(Xen9pfsRing));
    for (i = 0; i < xen_9pdev->num_rings; i++) {
        char *str;
        int ring_order;
        xen_9pdev->rings[i].priv = xen_9pdev;
        xen_9pdev->rings[i].evtchn = -1;
        xen_9pdev->rings[i].local_port = -1;
        str = g_strdup_printf("ring-ref%u", i);
        if (xenstore_read_fe_int(&xen_9pdev->xendev, str,
                                 &xen_9pdev->rings[i].ref) == -1) {
            goto out;
        }
        str = g_strdup_printf("event-channel-%u", i);
        if (xenstore_read_fe_int(&xen_9pdev->xendev, str,
                                 &xen_9pdev->rings[i].evtchn) == -1) {
            goto out;
        }
        xen_9pdev->rings[i].intf =  xengnttab_map_grant_ref(
                xen_9pdev->xendev.gnttabdev,
                xen_9pdev->xendev.dom,
                xen_9pdev->rings[i].ref,
                PROT_READ | PROT_WRITE);
        if (!xen_9pdev->rings[i].intf) {
            goto out;
        }
        ring_order = xen_9pdev->rings[i].intf->ring_order;
        if (ring_order > MAX_RING_ORDER) {
            goto out;
        }
        xen_9pdev->rings[i].ring_order = ring_order;
        xen_9pdev->rings[i].data = xengnttab_map_domain_grant_refs(
                xen_9pdev->xendev.gnttabdev,
                (1 << ring_order),
                xen_9pdev->xendev.dom,
                xen_9pdev->rings[i].intf->ref,
                PROT_READ | PROT_WRITE);
        if (!xen_9pdev->rings[i].data) {
            goto out;
        }
        xen_9pdev->rings[i].ring.in = xen_9pdev->rings[i].data;
        xen_9pdev->rings[i].ring.out = xen_9pdev->rings[i].data +
                                       XEN_FLEX_RING_SIZE(ring_order);
        xen_9pdev->rings[i].bh = qemu_bh_new(xen_9pfs_bh, &xen_9pdev->rings[i]);
        xen_9pdev->rings[i].out_cons = 0;
        xen_9pdev->rings[i].out_size = 0;
        xen_9pdev->rings[i].inprogress = false;
        xen_9pdev->rings[i].evtchndev = xenevtchn_open(NULL, 0);
        if (xen_9pdev->rings[i].evtchndev == NULL) {
            goto out;
        }
        fcntl(xenevtchn_fd(xen_9pdev->rings[i].evtchndev), F_SETFD, FD_CLOEXEC);
        xen_9pdev->rings[i].local_port = xenevtchn_bind_interdomain
                                            (xen_9pdev->rings[i].evtchndev,
                                             xendev->dom,
                                             xen_9pdev->rings[i].evtchn);
        if (xen_9pdev->rings[i].local_port == -1) {
            xen_pv_printf(xendev, 0,
                          "xenevtchn_bind_interdomain failed port=%d\
",
                          xen_9pdev->rings[i].evtchn);
            goto out;
        }
        xen_pv_printf(xendev, 2, "bind evtchn port %d\
", xendev->local_port);
        qemu_set_fd_handler(xenevtchn_fd(xen_9pdev->rings[i].evtchndev),
                xen_9pfs_evtchn_event, NULL, &xen_9pdev->rings[i]);
    }
    xen_9pdev->security_model = xenstore_read_be_str(xendev, "security_model");
    xen_9pdev->path = xenstore_read_be_str(xendev, "path");
    xen_9pdev->id = s->fsconf.fsdev_id =
        g_strdup_printf("xen9p%d", xendev->dev);
    xen_9pdev->tag = s->fsconf.tag = xenstore_read_fe_str(xendev, "tag");
    v9fs_register_transport(s, &xen_9p_transport);
    fsdev = qemu_opts_create(qemu_find_opts("fsdev"),
            s->fsconf.tag,
            1, NULL);
    qemu_opt_set(fsdev, "fsdriver", "local", NULL);
    qemu_opt_set(fsdev, "path", xen_9pdev->path, NULL);
    qemu_opt_set(fsdev, "security_model", xen_9pdev->security_model, NULL);
    qemu_opts_set_id(fsdev, s->fsconf.fsdev_id);
    qemu_fsdev_add(fsdev);
    v9fs_device_realize_common(s, NULL);
    return 0;
out:
    xen_9pfs_free(xendev);
    return -1;
}
static void nbd_refresh_filename(BlockDriverState *bs, QDict *options)
{
    BDRVNBDState *s = bs->opaque;
    QDict *opts = qdict_new();
    QObject *saddr_qdict;
    Visitor *ov;
    const char *host = NULL, *port = NULL, *path = NULL;
    if (s->saddr->type == SOCKET_ADDRESS_KIND_INET) {
        const InetSocketAddress *inet = s->saddr->u.inet.data;
        if (!inet->has_ipv4 && !inet->has_ipv6 && !inet->has_to) {
            host = inet->host;
            port = inet->port;
        }
    } else if (s->saddr->type == SOCKET_ADDRESS_KIND_UNIX) {
        path = s->saddr->u.q_unix.data->path;
    }
    qdict_put(opts, "driver", qstring_from_str("nbd"));
    if (path && s->export) {
        snprintf(bs->exact_filename, sizeof(bs->exact_filename),
                 "nbd+unix:///%s?socket=%s", s->export, path);
    } else if (path && !s->export) {
        snprintf(bs->exact_filename, sizeof(bs->exact_filename),
                 "nbd+unix://?socket=%s", path);
    } else if (host && s->export) {
        snprintf(bs->exact_filename, sizeof(bs->exact_filename),
                 "nbd://%s:%s/%s", host, port, s->export);
    } else if (host && !s->export) {
        snprintf(bs->exact_filename, sizeof(bs->exact_filename),
                 "nbd://%s:%s", host, port);
    }
    ov = qobject_output_visitor_new(&saddr_qdict);
    visit_type_SocketAddress(ov, NULL, &s->saddr, &error_abort);
    visit_complete(ov, &saddr_qdict);
    assert(qobject_type(saddr_qdict) == QTYPE_QDICT);
    qdict_put_obj(opts, "server", saddr_qdict);
    if (s->export) {
        qdict_put(opts, "export", qstring_from_str(s->export));
    }
    if (s->tlscredsid) {
        qdict_put(opts, "tls-creds", qstring_from_str(s->tlscredsid));
    }
    qdict_flatten(opts);
    bs->full_open_options = opts;
}
void fw_cfg_add_callback(FWCfgState *s, uint16_t key, FWCfgCallback callback,
                         void *callback_opaque, uint8_t *data, size_t len)
{
    int arch = !!(key & FW_CFG_ARCH_LOCAL);
    assert(key & FW_CFG_WRITE_CHANNEL);
    key &= FW_CFG_ENTRY_MASK;
    assert(key < FW_CFG_MAX_ENTRY && len <= 65535);
    s->entries[arch][key].data = data;
    s->entries[arch][key].len = len;
    s->entries[arch][key].callback_opaque = callback_opaque;
    s->entries[arch][key].callback = callback;
}
static int vncws_start_tls_handshake(VncState *vs)
{
    int ret = gnutls_handshake(vs->tls.session);
    if (ret < 0) {
        if (!gnutls_error_is_fatal(ret)) {
            VNC_DEBUG("Handshake interrupted (blocking)\
");
            if (!gnutls_record_get_direction(vs->tls.session)) {
                qemu_set_fd_handler(vs->csock, vncws_tls_handshake_io,
                                    NULL, vs);
            } else {
                qemu_set_fd_handler(vs->csock, NULL, vncws_tls_handshake_io,
                                    vs);
            }
            return 0;
        }
        VNC_DEBUG("Handshake failed %s\
", gnutls_strerror(ret));
        vnc_client_error(vs);
        return -1;
    }
    if (vs->vd->tls.x509verify) {
        if (vnc_tls_validate_certificate(vs) < 0) {
            VNC_DEBUG("Client verification failed\
");
            vnc_client_error(vs);
            return -1;
        } else {
            VNC_DEBUG("Client verification passed\
");
        }
    }
    VNC_DEBUG("Handshake done, switching to TLS data mode\
");
    qemu_set_fd_handler(vs->csock, vncws_handshake_read, NULL, vs);
    return 0;
}
static int qcow2_create(const char *filename, QemuOpts *opts, Error **errp)
{
    char *backing_file = NULL;
    char *backing_fmt = NULL;
    char *buf = NULL;
    uint64_t size = 0;
    int flags = 0;
    size_t cluster_size = DEFAULT_CLUSTER_SIZE;
    PreallocMode prealloc;
    int version;
    uint64_t refcount_bits;
    int refcount_order;
    const char *encryptfmt = NULL;
    Error *local_err = NULL;
    int ret;
    /* Read out options */
    size = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0),
                    BDRV_SECTOR_SIZE);
    backing_file = qemu_opt_get_del(opts, BLOCK_OPT_BACKING_FILE);
    backing_fmt = qemu_opt_get_del(opts, BLOCK_OPT_BACKING_FMT);
    encryptfmt = qemu_opt_get_del(opts, BLOCK_OPT_ENCRYPT_FORMAT);
    if (encryptfmt) {
        if (qemu_opt_get_del(opts, BLOCK_OPT_ENCRYPT)) {
            error_setg(errp, "Options " BLOCK_OPT_ENCRYPT " and "
                       BLOCK_OPT_ENCRYPT_FORMAT " are mutually exclusive");
            ret = -EINVAL;
            goto finish;
        }
    } else if (qemu_opt_get_bool_del(opts, BLOCK_OPT_ENCRYPT, false)) {
        encryptfmt = "aes";
    }
    cluster_size = qcow2_opt_get_cluster_size_del(opts, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        ret = -EINVAL;
        goto finish;
    }
    buf = qemu_opt_get_del(opts, BLOCK_OPT_PREALLOC);
    prealloc = qapi_enum_parse(PreallocMode_lookup, buf,
                               PREALLOC_MODE__MAX, PREALLOC_MODE_OFF,
                               &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        ret = -EINVAL;
        goto finish;
    }
    version = qcow2_opt_get_version_del(opts, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        ret = -EINVAL;
        goto finish;
    }
    if (qemu_opt_get_bool_del(opts, BLOCK_OPT_LAZY_REFCOUNTS, false)) {
        flags |= BLOCK_FLAG_LAZY_REFCOUNTS;
    }
    if (backing_file && prealloc != PREALLOC_MODE_OFF) {
        error_setg(errp, "Backing file and preallocation cannot be used at "
                   "the same time");
        ret = -EINVAL;
        goto finish;
    }
    if (version < 3 && (flags & BLOCK_FLAG_LAZY_REFCOUNTS)) {
        error_setg(errp, "Lazy refcounts only supported with compatibility "
                   "level 1.1 and above (use compat=1.1 or greater)");
        ret = -EINVAL;
        goto finish;
    }
    refcount_bits = qcow2_opt_get_refcount_bits_del(opts, version, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        ret = -EINVAL;
        goto finish;
    }
    refcount_order = ctz32(refcount_bits);
    ret = qcow2_create2(filename, size, backing_file, backing_fmt, flags,
                        cluster_size, prealloc, opts, version, refcount_order,
                        encryptfmt, &local_err);
    error_propagate(errp, local_err);
finish:
    g_free(backing_file);
    g_free(backing_fmt);
    g_free(buf);
    return ret;
}
long do_sigreturn(CPUPPCState *env)
{
    struct target_sigcontext *sc = NULL;
    struct target_mcontext *sr = NULL;
    target_ulong sr_addr = 0, sc_addr;
    sigset_t blocked;
    target_sigset_t set;
    sc_addr = env->gpr[1] + SIGNAL_FRAMESIZE;
    if (!lock_user_struct(VERIFY_READ, sc, sc_addr, 1))
        goto sigsegv;
#if defined(TARGET_PPC64)
    set.sig[0] = sc->oldmask + ((uint64_t)(sc->_unused[3]) << 32);
#else
    __get_user(set.sig[0], &sc->oldmask);
    __get_user(set.sig[1], &sc->_unused[3]);
#endif
    target_to_host_sigset_internal(&blocked, &set);
    set_sigmask(&blocked);
    __get_user(sr_addr, &sc->regs);
    if (!lock_user_struct(VERIFY_READ, sr, sr_addr, 1))
        goto sigsegv;
    restore_user_regs(env, sr, 1);
    unlock_user_struct(sr, sr_addr, 1);
    unlock_user_struct(sc, sc_addr, 1);
    return -TARGET_QEMU_ESIGRETURN;
sigsegv:
    unlock_user_struct(sr, sr_addr, 1);
    unlock_user_struct(sc, sc_addr, 1);
    force_sig(TARGET_SIGSEGV);
    return 0;
}
static int virtio_gpu_load(QEMUFile *f, void *opaque, size_t size)
{
    VirtIOGPU *g = opaque;
    struct virtio_gpu_simple_resource *res;
    struct virtio_gpu_scanout *scanout;
    uint32_t resource_id, pformat;
    int i;
    g->hostmem = 0;
    resource_id = qemu_get_be32(f);
    while (resource_id != 0) {
        res = g_new0(struct virtio_gpu_simple_resource, 1);
        res->resource_id = resource_id;
        res->width = qemu_get_be32(f);
        res->height = qemu_get_be32(f);
        res->format = qemu_get_be32(f);
        res->iov_cnt = qemu_get_be32(f);
        /* allocate */
        pformat = get_pixman_format(res->format);
        if (!pformat) {
            return -EINVAL;
        res->image = pixman_image_create_bits(pformat,
                                              res->width, res->height,
                                              NULL, 0);
        if (!res->image) {
            return -EINVAL;
        res->hostmem = PIXMAN_FORMAT_BPP(pformat) * res->width * res->height;
        res->addrs = g_new(uint64_t, res->iov_cnt);
        res->iov = g_new(struct iovec, res->iov_cnt);
        /* read data */
        for (i = 0; i < res->iov_cnt; i++) {
            res->addrs[i] = qemu_get_be64(f);
            res->iov[i].iov_len = qemu_get_be32(f);
        qemu_get_buffer(f, (void *)pixman_image_get_data(res->image),
                        pixman_image_get_stridex(res->image) * res->height);
        /* restore mapping */
        for (i = 0; i < res->iov_cnt; i++) {
            hwaddr len = res->iov[i].iov_len;
            res->iov[i].iov_base =
                cpu_physical_memory_map(res->addrs[i], &len, 1);
            if (!res->iov[i].iov_base || len != res->iov[i].iov_len) {

                return -EINVAL;
        QTAILQ_INSERT_HEAD(&g->reslist, res, next);
        g->hostmem += res->hostmem;
        resource_id = qemu_get_be32(f);
    /* load & apply scanout state */
    vmstate_load_state(f, &vmstate_virtio_gpu_scanouts, g, 1);
    for (i = 0; i < g->conf.max_outputs; i++) {
        scanout = &g->scanout[i];
        if (!scanout->resource_id) {
            continue;
        res = virtio_gpu_find_resource(g, scanout->resource_id);
        if (!res) {
            return -EINVAL;
        scanout->ds = qemu_create_displaysurface_pixman(res->image);
        if (!scanout->ds) {
            return -EINVAL;
        dpy_gfx_replace_surface(scanout->con, scanout->ds);
        dpy_gfx_update(scanout->con, 0, 0, scanout->width, scanout->height);
        update_cursor(g, &scanout->cursor);
        res->scanout_bitmask |= (1 << i);
    return 0;
void helper_slbie(CPUPPCState *env, target_ulong addr)
{
    PowerPCCPU *cpu = ppc_env_get_cpu(env);
    ppc_slb_t *slb;
    slb = slb_lookup(cpu, addr);
    if (!slb) {
        return;
    }
    if (slb->esid & SLB_ESID_V) {
        slb->esid &= ~SLB_ESID_V;
        /* XXX: given the fact that segment size is 256 MB or 1TB,
         *      and we still don't have a tlb_flush_mask(env, n, mask)
         *      in QEMU, we just invalidate all TLBs
         */
        tlb_flush(CPU(cpu), 1);
    }
}
static inline void do_rfi(CPUPPCState *env, target_ulong nip, target_ulong msr)
{
    CPUState *cs = CPU(ppc_env_get_cpu(env));

    /* MSR:POW cannot be set by any form of rfi */
    msr &= ~(1ULL << MSR_POW);

#if defined(TARGET_PPC64)
    /* Switching to 32-bit ? Crop the nip */
    if (!msr_is_64bit(env, msr)) {
        nip = (uint32_t)nip;
    }
#else
    nip = (uint32_t)nip;
#endif
    /* XXX: beware: this is false if VLE is supported */
    env->nip = nip & ~((target_ulong)0x00000003);
    hreg_store_msr(env, msr, 1);
#if defined(DEBUG_OP)
    cpu_dump_rfi(env->nip, env->msr);
#endif
    /* No need to raise an exception here,
     * as rfi is always the last insn of a TB
     */
    cs->interrupt_request |= CPU_INTERRUPT_EXITTB;

    /* Context synchronizing: check if TCG TLB needs flush */
    check_tlb_flush(env);
}

static void test_acpi_asl(test_data *data)
{
    int i;
    AcpiSdtTable *sdt, *exp_sdt;
    test_data exp_data;
    gboolean exp_err, err;

    memset(&exp_data, 0, sizeof(exp_data));
    exp_data.tables = load_expected_aml(data);
    dump_aml_files(data, false);
    for (i = 0; i < data->tables->len; ++i) {
        GString *asl, *exp_asl;

        sdt = &g_array_index(data->tables, AcpiSdtTable, i);
        exp_sdt = &g_array_index(exp_data.tables, AcpiSdtTable, i);

        err = load_asl(data->tables, sdt);
        asl = normalize_asl(sdt->asl);

        exp_err = load_asl(exp_data.tables, exp_sdt);
        exp_asl = normalize_asl(exp_sdt->asl);

        /* TODO: check for warnings */
        g_assert(!err || exp_err);

        if (g_strcmp0(asl->str, exp_asl->str)) {
            uint32_t signature = cpu_to_le32(exp_sdt->header.signature);
            sdt->tmp_files_retain = true;
            exp_sdt->tmp_files_retain = true;
            fprintf(stderr,
                    "acpi-test: Warning! %.4s mismatch. "
                    "Actual [asl:%s, aml:%s], Expected [asl:%s, aml:%s].\
",
                    (gchar *)&signature,
                    sdt->asl_file, sdt->aml_file,
                    exp_sdt->asl_file, exp_sdt->aml_file);
        }
        g_string_free(asl, true);
        g_string_free(exp_asl, true);
    }

    free_test_data(&exp_data);
}
