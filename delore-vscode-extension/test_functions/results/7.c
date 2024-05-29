move_file_to_low_priority_queue (NautilusDirectory *directory,
                                 NautilusFile      *file)
{
     
    nautilus_file_queue_enqueue (directory->details->low_priority_queue,
                                 file);
    nautilus_file_queue_remove (directory->details->high_priority_queue,
                                file);
}


  virtual bool RunOnDBThread(history::HistoryBackend* backend,
                             history::HistoryDatabase* db) {
    work_->Run();
    done_->Signal();
    return true;
  }


bool ValidateRangeChecksum(const HistogramBase& histogram,
                           uint32_t range_checksum) {
  const Histogram& casted_histogram =
      static_cast<const Histogram&>(histogram);

  return casted_histogram.bucket_ranges()->checksum() == range_checksum;
}


BrowserDevToolsAgentHost::BrowserDevToolsAgentHost(
    scoped_refptr<base::SingleThreadTaskRunner> tethering_task_runner,
    const CreateServerSocketCallback& socket_callback,
    bool only_discovery)
    : DevToolsAgentHostImpl(base::GenerateGUID()),
      tethering_task_runner_(tethering_task_runner),
      socket_callback_(socket_callback),
      only_discovery_(only_discovery) {
  NotifyCreated();
}


static void finish_ordered_fn(struct btrfs_work *work)
{
	struct btrfs_ordered_extent *ordered_extent;
	ordered_extent = container_of(work, struct btrfs_ordered_extent, work);
	btrfs_finish_ordered_io(ordered_extent);
}
