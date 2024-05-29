 void InterstitialPage::Observe(NotificationType type,
                                const NotificationSource& source,
                                const NotificationDetails& details) {
   switch (type.value) {
     case NotificationType::NAV_ENTRY_PENDING:
        Disable();
      DCHECK(!resource_dispatcher_host_notified_);
        TakeActionOnResourceDispatcher(CANCEL);
        break;
      case NotificationType::RENDER_WIDGET_HOST_DESTROYED:
       if (action_taken_ == NO_ACTION) {
         RenderViewHost* rvh = Source<RenderViewHost>(source).ptr();
         DCHECK(rvh->process()->id() == original_child_id_ &&
                rvh->routing_id() == original_rvh_id_);
         TakeActionOnResourceDispatcher(CANCEL);
       }
       break;
     case NotificationType::TAB_CONTENTS_DESTROYED:
     case NotificationType::NAV_ENTRY_COMMITTED:
       if (action_taken_ == NO_ACTION) {
         DontProceed();
       } else {
         Hide();
       }
       break;
     default:
       NOTREACHED();
   }
 }

static ssize_t ap_hwtype_show(struct device *dev,
			      struct device_attribute *attr, char *buf)
{
	struct ap_device *ap_dev = to_ap_dev(dev);
	return snprintf(buf, PAGE_SIZE, "%d\n", ap_dev->device_type);
}


struct dentry *d_find_alias(struct inode *inode)
{
	struct dentry *de = NULL;

	if (!hlist_empty(&inode->i_dentry)) {
		spin_lock(&inode->i_lock);
		de = __d_find_alias(inode);
		spin_unlock(&inode->i_lock);
	}
	return de;
}


void AudioInputRendererHost::OnRecordStream(int stream_id) {
  DCHECK(BrowserThread::CurrentlyOn(BrowserThread::IO));

  AudioEntry* entry = LookupById(stream_id);
  if (!entry) {
    SendErrorMessage(stream_id);
    return;
  }

  entry->controller->Record();
}


 static int db_interception(struct vcpu_svm *svm)
 {
 	struct kvm_run *kvm_run = svm->vcpu.run;
 
 	if (!(svm->vcpu.guest_debug &
 	      (KVM_GUESTDBG_SINGLESTEP | KVM_GUESTDBG_USE_HW_BP)) &&
 		!svm->nmi_singlestep) {
 		kvm_queue_exception(&svm->vcpu, DB_VECTOR);
 		return 1;
 	}
 
 	if (svm->nmi_singlestep) {
 		svm->nmi_singlestep = false;
  		if (!(svm->vcpu.guest_debug & KVM_GUESTDBG_SINGLESTEP))
  			svm->vmcb->save.rflags &=
  				~(X86_EFLAGS_TF | X86_EFLAGS_RF);
		update_db_bp_intercept(&svm->vcpu);
  	}
  
  	if (svm->vcpu.guest_debug &
 	    (KVM_GUESTDBG_SINGLESTEP | KVM_GUESTDBG_USE_HW_BP)) {
 		kvm_run->exit_reason = KVM_EXIT_DEBUG;
 		kvm_run->debug.arch.pc =
 			svm->vmcb->save.cs.base + svm->vmcb->save.rip;
 		kvm_run->debug.arch.exception = DB_VECTOR;
 		return 0;
 	}
 
 	return 1;
 }