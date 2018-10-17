from script.model.sklearn_like_model.callback.BaseEpochCallback import BaseEpochCallback
import math


class CosineResetLRScheduler(BaseEpochCallback):
    def __init__(
            self, cycle, max_lr, min_lr, name='CosineResetLRScheduler', exponential_decay=True, base=10,
            log=print
    ):
        if not cycle > 0:
            raise ValueError(f'cycle expect cycle > 0, but {cycle}')
        if not 0 < max_lr < 1:
            raise ValueError(f'max_lr expect 0 < max_lr < 1, but {max_lr}')
        if not 0 < min_lr < 1:
            raise ValueError(f'min_lr expect 0 < min_lr < 1, but {min_lr}')

        self.cycle = cycle
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.name = name
        self.log = log
        self.exponential = exponential_decay
        self.base = base

        self.cycle_count = 1
        self.reset()

    def __str__(self):
        s = f"{self.__class__.__name__}"
        s += f"cycle = {self.cycle}\n"
        s += f"max_lr = {self.max_lr}\n"
        s += f"min_lr = {self.min_lr}\n"
        s += f"exponential = {self.exponential}\n"
        s += f"base = {self.base}\n"

        return s

    def __call__(self, model, dataset, metric, epoch):
        if self.cycle_count > self.cycle:
            self.reset()
            new_lr = self.max_lr

            if self.exponential:
                new_lr = self.base ** new_lr

            model.update_learning_rate(new_lr)
            self.log(f'in {self.name}, reset learning rate to {new_lr}')
        else:
            lr = model.learning_rate

            radian = self.cycle_count * (math.pi / self.cycle)
            val = (math.cos(radian) + 1) / 2

            diff = self.max_lr - self.min_lr
            delta = diff - diff * val
            new_lr = self.max_lr - delta
            if self.exponential:
                new_lr = self.base ** new_lr

            model.update_learning_rate(new_lr)
            self.log(f'in {self.name}, update learning rate from {lr} to {new_lr}')

            self.cycle_count += 1

    def reset(self):
        self.cycle_count = 1
