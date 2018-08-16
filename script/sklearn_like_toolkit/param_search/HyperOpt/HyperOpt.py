import multiprocessing as mp
import time
import numpy as np
from functools import wraps
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, hp
from hyperopt.mongoexp import MongoTrials
from tqdm import tqdm
from script.sklearn_like_toolkit.param_search.HyperOpt.FreeTrials import FreeTrials
from script.util.MixIn import singletonPoolMixIn
from script.util.misc_util import log_error_trace


def deco_hyperOpt_fn(func, min_best=True, pbar=None, feed_args=None,
                     feed_kwargs=None):
    @wraps(func)
    def deco_hyperOpt_fn_wrapper(params):
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
            'params': params
        }

        try:
            if issubclass(func, HyperOpt_fn):
                ret = func.fn(params, feed_args, feed_kwargs)
            else:
                ret = func(params, feed_args, feed_kwargs)

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

    deco_hyperOpt_fn_wrapper.__name__ = deco_hyperOpt_fn.__name__
    return deco_hyperOpt_fn_wrapper


class HyperOpt_fn:
    @staticmethod
    def fn(params, feed_args, feed_kwargs):
        raise NotImplementedError


CPU_COUNT = mp.cpu_count() - 1


class HyperOpt(singletonPoolMixIn):
    _pool_singleton = None

    def __init__(self, min_best=True, n_jobs=CPU_COUNT):
        singletonPoolMixIn.__init__(self, n_jobs)
        self._min_best = min_best
        self._trials = None
        self._best_param = None

    @property
    def min_best(self):
        return self._min_best

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
        idx = np.argmin(self.losses)
        return self.result[idx]['params']

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

    @property
    def best_result(self):
        return self._trials.best_trial

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

    def fit_serial(self, func, space, n_iter, feed_args=None, feed_kwargs=None,
                   trials=None, algo=tpe.suggest, min_best=None, pbar=True):
        self._check_Trials(trials)
        if trials is None:
            trials = FreeTrials()
        trials.refresh()

        if min_best is None:
            min_best = self.min_best

        if pbar:
            pbar = tqdm(range(n_iter))
        else:
            pbar = None

        if len(space) == 0:
            space = hp.choice('dummy', [{}])

        fmin(
            fn=deco_hyperOpt_fn(
                func, min_best, pbar,
                feed_args=feed_args,
                feed_kwargs=feed_kwargs
            ),
            space=space,
            algo=algo,
            max_evals=n_iter + len(trials),
            trials=trials,
        )

        pbar.close()
        self._trials = trials
        return self._trials

    def fit_parallel(self, func, space, n_iter, feed_args=None,
                     feed_kwargs=None, trials=None, algo=tpe.suggest,
                     pbar=True, min_best=None):
        self._check_Trials(trials)
        if trials is None:
            trials = FreeTrials()
        trials.refresh()

        if min_best is None:
            min_best = self.min_best

        if len(space) == 0:
            return self.fit_serial(func, space, n_iter, feed_args, feed_kwargs, trials, algo, pbar, min_best)

        base_trials = trials

        opt = HyperOpt()
        childs = []
        for _ in range(n_iter):
            child = self.pool.apply_async(
                opt.fit_serial,
                args=(func, space, 1),
                kwds={
                    'feed_args': feed_args,
                    'feed_kwargs': feed_kwargs,
                    'algo': algo,
                    'trials': base_trials.deepcopy(refresh=False),
                    'min_best': min_best,
                    'pbar': False
                })
            childs += [child]

        if pbar:
            childs = tqdm(childs)

        print('collect job')
        n_already_done = len(base_trials)
        for child in childs:
            trials = child.get()
            partial = trials.partial_deepcopy(n_already_done, n_already_done + 1)
            base_trials = base_trials.concat(partial, refresh=False)

        base_trials.refresh()
        self._trials = base_trials

        return base_trials

    def fit_parallel_async(self, func, data_pack, space, n_iter, feed_args=None,
                           feed_kwargs=None, algo=tpe.suggest,
                           trials=None, min_best=None):
        raise NotImplementedError
