import random
from collections import deque


class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self._memory = deque()

    def __len__(self):
        return len(self._memory)

    def sample(self, n):
        return random.sample(self._memory, n)

    def store(self, item):
        self._memory.append(item)
        if len(self._memory) > self.max_size:
            self._memory.popleft()

    def batch_update(self, tree_idxs, abs_errors):
        raise NotImplementedError
