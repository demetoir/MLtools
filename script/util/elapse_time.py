import time


class elapse_time:
    def __init__(self, title=None):
        self.start_time = time.time()
        self.title = title

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.title:
            print(self.title)
        print(f"time {time.time() - self.start_time:.4f}'s elapsed")
        return True

    def __enter__(self):
        return None