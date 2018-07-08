import time
from functools import wraps
import numpy as np
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from hyperopt.mongoexp import MongoTrials
from tqdm import tqdm
from script.util.misc_util import log_error_trace


def deco_hyperOpt(func, min_best=True, pbar=None):
    @wraps(func)
    def deco_hyperOpt_wrapper(kwargs):
        start_time = time.time()
        trial = {
            'loss': None,
            'status': None,
            'eval_time': None,
            # 'other_stuff': None,
            # -- attachments are handled differently
            # 'attachments':
            #     {'time_module': None},
            # 'params': kwargs['params']
            'params': kwargs
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

            try:
                pbar.update(1)
            except BaseException:
                pass
            return trial

    deco_hyperOpt_wrapper.__name__ = deco_hyperOpt.__name__
    return deco_hyperOpt_wrapper


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
        return {
            'params': space,
            'data_pack': data_pack
        }

    @staticmethod
    def _check_Trials(trials):
        if isinstance(trials, MongoTrials):
            raise TypeError("MongoTrials not support")

    def fit(self, func, space, n_iter, algo=tpe.suggest, trials=None, min_best=None, pbar=True):
        if min_best is None:
            min_best = self.min_best

        self._check_Trials(trials)

        if trials is None:
            self._trials = Trials()
        else:
            trials.refresh()
            self._trials = trials

        if pbar is True:
            pbar = tqdm(range(n_iter))
        else:
            pbar = None

        self._best_param = fmin(
            fn=deco_hyperOpt(func, min_best, pbar),
            space=space,
            algo=algo,
            max_evals=n_iter + len(trials),
            trials=self._trials,
        )

        return self._trials
