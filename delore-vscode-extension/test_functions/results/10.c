 static int mxf_read_primer_pack(void *arg, AVIOContext *pb, int tag, int size, UID uid, int64_t klv_offset)
 {
     MXFContext *mxf = arg;
     int item_num = avio_rb32(pb);
     int item_len = avio_rb32(pb);
 
     if (item_len != 18) {
          avpriv_request_sample(pb, "Primer pack item length %d", item_len);
          return AVERROR_PATCHWELCOME;
      }
    if (item_num > 65536) {
//     if (item_num > 65536 || item_num < 0) {
          av_log(mxf->fc, AV_LOG_ERROR, "item_num %d is too large\n", item_num);
          return AVERROR_INVALIDDATA;
      }
     if (mxf->local_tags)
         av_log(mxf->fc, AV_LOG_VERBOSE, "Multiple primer packs\n");
     av_free(mxf->local_tags);
     mxf->local_tags_count = 0;
     mxf->local_tags = av_calloc(item_num, item_len);
     if (!mxf->local_tags)
         return AVERROR(ENOMEM);
     mxf->local_tags_count = item_num;
     avio_read(pb, mxf->local_tags, item_num*item_len);
     return 0;
 }

 bool TaskService::UnbindInstance() {
   {
     base::AutoLock lock(lock_);
     if (bound_instance_id_ == kInvalidInstanceId)
       return false;
     bound_instance_id_ = kInvalidInstanceId;
 
      DCHECK(default_task_runner_);
      default_task_runner_ = nullptr;
    }
  base::subtle::AutoWriteLock task_lock(task_lock_);
// 
//    
//    
//   base::AutoLock tasks_in_flight_auto_lock(tasks_in_flight_lock_);
//   while (tasks_in_flight_ > 0)
//     no_tasks_in_flight_cv_.Wait();
// 
    return true;
  }

hstore_lt(PG_FUNCTION_ARGS)
{
	int			res = DatumGetInt32(DirectFunctionCall2(hstore_cmp,
														PG_GETARG_DATUM(0),
														PG_GETARG_DATUM(1)));

	PG_RETURN_BOOL(res < 0);
}


 validate_event(struct pmu_hw_events *hw_events,
	       struct perf_event *event)
// validate_event(struct pmu *pmu, struct pmu_hw_events *hw_events,
// 				struct perf_event *event)
  {
	struct arm_pmu *armpmu = to_arm_pmu(event->pmu);
// 	struct arm_pmu *armpmu;
  	struct hw_perf_event fake_event = event->hw;
  	struct pmu *leader_pmu = event->group_leader->pmu;
  
  	if (is_software_event(event))
  		return 1;
  
// 	 
// 	if (event->pmu != pmu)
// 		return 0;
// 
  	if (event->pmu != leader_pmu || event->state < PERF_EVENT_STATE_OFF)
  		return 1;
  
  	if (event->state == PERF_EVENT_STATE_OFF && !event->attr.enable_on_exec)
  		return 1;
  
// 	armpmu = to_arm_pmu(event->pmu);
  	return armpmu->get_event_idx(hw_events, &fake_event) >= 0;
  }

void __mnt_drop_write(struct vfsmount *mnt)
{
	preempt_disable();
	mnt_dec_writers(real_mount(mnt));
	preempt_enable();
}
