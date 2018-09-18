import os
import numpy as np
import shutil
from pprint import pprint
from script.util.misc_util import path_join, load_json, setup_directory, dump_json, error_trace


class Top_k_save:
    def __init__(self, path, k=5, max_best=True, save_model=True, name='top_k_save'):
        self.path = path
        self.k = k
        self.max_best = max_best
        self.save_model = save_model
        self.name = name

        self.top_k_json_path = path_join(self.path, 'top_k.json')
        if os.path.exists(self.top_k_json_path):
            self.top_k = load_json(self.top_k_json_path)
        else:
            if self.max_best:
                self.top_k = [np.Inf] + [-np.Inf for _ in range(self.k)]
            else:
                self.top_k = [-np.Inf] + [np.Inf for _ in range(self.k)]

        if self.save_model:
            for i in range(1, self.k + 1):
                setup_directory(path_join(self.path, f'top_{i}'))

    def __call__(self, metric, model):
        metric = float(metric)
        sign = 1 if self.max_best else -1


        print()
        print(f'{self.name} current top_k')
        pprint(self.top_k[1:])

        try:
            for i in reversed(range(1, self.k + 1)):
                if sign * self.top_k[i - 1] >= sign * metric >= sign * self.top_k[i]:
                    # update top_k
                    self.top_k.insert(i, metric)
                    self.top_k.pop(self.k + 1)

                    # dump top_k json
                    dump_json(self.top_k, path_join(self.path, 'top_k.json'))
                    print(f'update top_k at {i}th, metric = {metric}')
                    pprint(self.top_k[1:])

                    if self.save_model:
                        # del worst dir
                        shutil.rmtree(path_join(self.path, f'top_{self.k}'))

                        # shift dir
                        path_pairs = [
                            (
                                path_join(self.path, f'top_{idx}'),
                                path_join(self.path, f'top_{idx+1}')
                            )
                            for idx in range(i, self.k)
                        ]
                        path_pairs = list(reversed(path_pairs))
                        for src, dst in path_pairs:
                            os.rename(src, dst)

                        # save model
                        save_path = path_join(self.path, f'top_{i}')
                        model.save(save_path)

                    break
        except BaseException as e:
            print(error_trace(e))
            raise RuntimeError(f'while Top k save, raise {e}')
