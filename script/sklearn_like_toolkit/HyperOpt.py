from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp
from tqdm import tqdm
# from hyperopt.mongoexp import MongoTrials
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

    def fit(self, func, data_pack, space, max_eval, algo=tpe.suggest, min_best=None, mongoTrials=False):
        if min_best is None:
            min_best = self.min_best
        space = hp.choice(
            'kwargs', [{
                'params': space,
                'data_pack': data_pack
            }]
        )

        if mongoTrials:
            # todo
            raise NotImplementedError
            # self._trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')
        else:
            self._trials = Trials()

        with tqdm(range(max_eval)) as pbar:
            func = deco_hyperOpt(func, min_best, pbar)

            self._best_param = fmin(
                # fn=deco_hyperOpt(func, min_best, pbar),
                fn=func,
                space=space,
                algo=algo,
                max_evals=max_eval,
                trials=self._trials)

        return self._best_param