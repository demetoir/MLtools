import time
from functools import wraps
import numpy as np
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, rand, mix, anneal
from hyperopt.mongoexp import MongoTrials
from joblib import Parallel, delayed
from tqdm import tqdm, trange
import multiprocessing as mp
from script.sklearn_like_toolkit.HyperOpt.FreeTrials import FreeTrials
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


CPU_COUNT = mp.cpu_count() - 1


class HyperOpt:

    def __init__(self, min_best=True, n_job=CPU_COUNT):
        self.min_best = min_best
        self._trials = None
        self._best_param = None
        self.n_job = n_job

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

    def fit_serial(self, func, space, n_iter, algo=tpe.suggest, trials=None, min_best=None, pbar=True):
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

    def fit_parallel(self, func, space, n_iter, algo=rand.suggest, trials=None, min_best=None, pbar=True):
        self._check_Trials(trials)
        min_best = self.min_best if min_best is None else min_best
        trials = FreeTrials() if trials is None else trials
        range_ = trange if pbar else range

        # current_done = len(trials)
        base_trials = trials
        opt = HyperOpt()
        ret = Parallel(n_jobs=self.n_job)(
            delayed(opt.fit_serial)(
                func, space, 1,
                algo=algo,
                trials=base_trials.deepcopy(refresh=False),
                min_best=min_best,
                pbar=False
            ) for _ in range_(n_iter))

        for trials in ret:
            # partial = trials.partial_deepcopy(current_done, current_done + 1)
            base_trials = base_trials.concat(trials, refresh=False)

        base_trials.refresh()
        self._trials = base_trials

        return base_trials

    def fit_parallel_async(self, func, data_pack, space, max_eval, algo=tpe.suggest, trials=None, min_best=None):
        raise NotImplementedError
