from script.model.sklearn_like_model.callback.BaseEpochCallback import BaseEpochCallback


class TriangleLRScheduler(BaseEpochCallback):
    def __init__(self, cycle, max_lr, min_lr, name='TriangleLRScheduler', decay_factor=0.99,
                 log=print):
        self.cycle = cycle
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.name = name
        self.log = log
        self.decay_factor = decay_factor

        self.cycle_count = 1
        self.reset()
        self.sign = -1

    def reset(self):
        self.cycle_count = 1

    def __call__(self, model, dataset, metric, epoch):
        if self.decay_factor:
            self.max_lr = max(self.max_lr * self.decay_factor, self.min_lr)

        lr = model.learning_rate
        if self.cycle_count > self.cycle:
            new_lr = self.min_lr if self.sign == -1 else self.max_lr

            self.sign *= -1
            self.reset()
            model.update_learning_rate(new_lr)
        else:
            delta = (self.max_lr - self.min_lr) / self.cycle
            new_lr = delta * self.sign + lr

            self.cycle_count += 1
            model.update_learning_rate(new_lr)
        self.log(f'in {self.name}, update learning rate from {lr} to {new_lr}')
