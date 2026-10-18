import os
import time
import threading
import psutil

class WindowsMemoryTracker:
    """
    Background daemon sampler that tracks process RSS and peak working set (peak_wset)
    specifically designed for Windows environments.
    """
    def __init__(self, sample_interval_sec=0.1):
        self.sample_interval = sample_interval_sec
        self.process = psutil.Process(os.getpid())
        self.peak_rss_bytes = 0
        self.peak_wset_bytes = 0
        self.start_rss_bytes = 0
        self._running = False
        self._thread = None

    def start(self):
        mem = self.process.memory_info()
        self.start_rss_bytes = mem.rss
        self.peak_rss_bytes = mem.rss
        self.peak_wset_bytes = getattr(mem, 'peak_wset', mem.rss)
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        while self._running:
            try:
                mem = self.process.memory_info()
                if mem.rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = mem.rss
                wset = getattr(mem, 'peak_wset', mem.rss)
                if wset > self.peak_wset_bytes:
                    self.peak_wset_bytes = wset
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.sample_interval)

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        # Final capture
        try:
            mem = self.process.memory_info()
            self.peak_rss_bytes = max(self.peak_rss_bytes, mem.rss)
            self.peak_wset_bytes = max(self.peak_wset_bytes, getattr(mem, 'peak_wset', mem.rss))
        except Exception:
            pass

    def get_summary(self):
        return {
            "start_rss_mb": round(self.start_rss_bytes / (1024 * 1024), 2),
            "peak_rss_mb": round(self.peak_rss_bytes / (1024 * 1024), 2),
            "peak_wset_mb": round(self.peak_wset_bytes / (1024 * 1024), 2),
            "net_rss_increase_mb": round(max(0, self.peak_rss_bytes - self.start_rss_bytes) / (1024 * 1024), 2)
        }

def get_system_memory_status():
    """Returns available and total system memory in MB."""
    vm = psutil.virtual_memory()
    return {
        "total_mb": round(vm.total / (1024 * 1024), 1),
        "available_mb": round(vm.available / (1024 * 1024), 1),
        "percent_used": vm.percent
    }

def get_adaptive_n_jobs():
    """
    Dynamically select n_jobs for scikit-learn based on available RAM:
      - < 2 GB free: n_jobs=1
      - 2-4 GB free: n_jobs=2
      - > 4 GB free: min(4, cpu_count)
    """
    vm = psutil.virtual_memory()
    avail_gb = vm.available / (1024 ** 3)
    cpu_count = os.cpu_count() or 1

    if avail_gb < 2.0:
        return 1
    elif avail_gb < 4.0:
        return min(2, cpu_count)
    else:
        return min(4, cpu_count)
