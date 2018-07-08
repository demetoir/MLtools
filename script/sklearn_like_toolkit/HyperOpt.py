from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp
from tqdm import tqdm
from script.util.misc_util import log_error_trace
import time
import numpy as np
from functools import wraps


# import cloud

def deco_hyperOpt(func, min_best=True, pbar=None):
    @wraps(func)
    def wrapper(kwargs):
        start_time = time.time()
        trial = {
            'loss': None,
            'status': None,
            'eval_time': None,
            # 'other_stuff': None,
            # -- attachments are handled differently
            # 'attachments':
            #     {'time_module': None},
            'params': kwargs['params']
        }

        try:
            ret = func(kwargs)
            if type(ret) is dict:
                trial.update(ret)
            else:
                trial['loss'] = ret

            trial['status'] = STATUS_OK

        except BaseException as e:
            log_error_trace(print, e)
            trial['loss'] = np.inf
            trial['status'] = STATUS_FAIL
        finally:
            trial['eval_time'] = time.time() - start_time

            if min_best is False:
                trial['loss'] = -trial['loss']

            if pbar is not None:
                pbar.update(1)
            return trial

    wrapper.__name__ = deco_hyperOpt.__name__
    return wrapper


class HyperOpt:

    def __init__(self, min_best=True):
        self.min_best = min_best
        self._trials = None
        self._best_param = None

    @property
    def outer_trials(self):
        return self._trials

    @property
    def trials(self):
        return self._trials.trials

    @property
    def result(self):
        return self._trials.results

    @property
    def losses(self):
        return self._trials.losses()

    @property
    def statuses(self):
        return self._trials.statuses()

    @property
    def best_param(self):
        return self._best_param

    @property
    def best_loss(self):
        if self.min_best:
            return min(self.losses)
        else:
            return max(self.losses)

    @property
    def opt_info(self):
        eval_times = [d['eval_time'] for d in self.result]
        info = {
            'mean_eval_time': np.mean(eval_times),
            'std_eval_time': np.std(eval_times),
            'mean_loss': np.mean(self.losses),
            'std_loss': np.std(self.losses),

        }
        return info

    @staticmethod
    def _make_feed_space(data_pack, space):
        return hp.choice(
            'kwargs', [{
                'params': space,
                'data_pack': data_pack
            }]
        )

    def fit(self, func, data_pack, space, max_eval, algo=tpe.suggest, trials=None, min_best=None):
        if min_best is None:
            min_best = self.min_best

        if trials is None:
            self._trials = Trials()
        else:
            self._trials = trials

        with tqdm(range(max_eval)) as pbar:
            self._best_param = fmin(
                fn=deco_hyperOpt(func, min_best, pbar),
                space=self._make_feed_space(data_pack, space),
                algo=algo,
                max_evals=max_eval,
                trials=self._trials)
        return self._trials

    def resume_fit(self, func, data_pack, space, max_eval, trials, algo=tpe.suggest, min_best=None):
        if min_best is None:
            min_best = self.min_best
        space = hp.choice(
            'kwargs', [{
                'params': space,
                'data_pack': data_pack
            }]
        )

        with tqdm(range(max_eval)) as pbar:
            func = deco_hyperOpt(func, min_best, pbar)

            self._best_param = fmin(
                # fn=deco_hyperOpt(func, min_best, pbar),
                fn=func,
                space=space,
                algo=algo,
                max_evals=max_eval,
                trials=trials)

        self._trials = trials
        return trials
