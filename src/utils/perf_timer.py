import time


class PerfTimer:
    def __init__(self, label):
        self.label = label
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.label}] Elapsed time: {elapsed:.6f} seconds")
