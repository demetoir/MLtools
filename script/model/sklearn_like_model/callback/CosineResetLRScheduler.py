from script.model.sklearn_like_model.callback.BaseEpochCallback import BaseEpochCallback
import math


class CosineResetLRScheduler(BaseEpochCallback):
    def __init__(self, cycle, max_lr, min_lr, name='CosineResetLRScheduler', exponential=True, base=10, log=print):
        self.cycle = cycle
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.name = name
        self.log = log
        self.exponential = exponential
        self.base = base

        self.cycle_count = 1
        self.reset()

    def reset(self):
        self.cycle_count = 1

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
