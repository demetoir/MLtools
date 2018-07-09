from hyperopt import Trials


class FreeTrials(Trials):
    def __init__(self, exp_key=None, refresh=True):
        super().__init__(exp_key, refresh)

    def __getitem__(self, item):
        return self.trials.__getitem__(item)

    def partial_deepcopy(self, low, high):
        new_ = self.__class__()

        _dynamic_trials = self[low: high]
        for i, trials in enumerate(_dynamic_trials):
            trials['tid'] = i
        new_._dynamic_trials = _dynamic_trials

        new_.refresh()
        return new_

    def deepcopy(self, refresh=True):
        from copy import deepcopy
        new_ = self.__class__()

        _dynamic_trials = deepcopy(self._dynamic_trials)
        for i, trials in enumerate(_dynamic_trials):
            trials['tid'] = i
        new_._dynamic_trials = _dynamic_trials

        if refresh:
            new_.refresh()
        return new_

    def concat(self, a, refresh=True):
        _dynamic_trials = self._dynamic_trials + a._dynamic_trials
        for i, trials in enumerate(_dynamic_trials):
            trials['tid'] = i
        self._dynamic_trials = _dynamic_trials

        if refresh:
            self.refresh()
        return self
