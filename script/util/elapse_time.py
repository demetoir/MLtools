import time


class elapse_time:
    def __init__(self, title=None):
        self.start_time = time.time()
        self.title = title
        if title is None:
            self.title = ''
        else:
            self.title = title

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{self.title}, time {time.time() - self.start_time:.4f}'s elapsed")
        if (exc_type, exc_val, exc_tb) == (None, None, None):
            return True
        else:
            # print(exc_type)
            # print(exc_val)
            print(exc_tb)
            return False

    def __enter__(self):
        return None
