from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, precision_score
from util.MixIn import PickleMixIn
from util.numpy_utils import reformat_np_arr, NP_ARRAY_TYPE_INDEX, NP_ARRAY_TYPE_ONEHOT


class meta_BaseWrapperClf(type):
    def __init__(cls, name, bases, cls_dict):
        type.__init__(cls, name, bases, cls_dict)

        def deco_reformat_y(func):
            def wrapper(*args, **kwargs):
                y = args[2]
                args = list(args[:2]) + [Reformat_Ys_MixIn.np_arr_to_index(y)] + list(args[3:])

                ret = func(*args, **kwargs)
                return ret

            return wrapper

        func_names = ['score', 'score_pack', 'fit']
        for func_name in func_names:
            if hasattr(cls, func_name):
                func = getattr(cls, func_name)
                setattr(cls, func_name, deco_reformat_y(func))


class meta_BaseWrapperClf_with_ABC(meta_BaseWrapperClf, ABCMeta):
    pass


class Reformat_Ys_MixIn:
    @staticmethod
    def np_arr_to_index(Xs):
        return reformat_np_arr(Xs, NP_ARRAY_TYPE_INDEX)

    @staticmethod
    def np_arr_to_onehot(Xs):
        return reformat_np_arr(Xs, NP_ARRAY_TYPE_ONEHOT)


class DummyParamMixIN:
    def get_params(self, deep=True):
        if hasattr(self, 'get_params'):
            return getattr(self, 'get_params')(self, deep=deep)
        else:
            return {}

    def set_params(self, **params):
        if hasattr(self, 'get_params'):
            return getattr(self, 'get_params')(self, **params)
        else:
            return None

CLF_METRICS = {
    'accuracy': accuracy_score,
    'confusion_matrix': confusion_matrix,
    'roc_auc_score': roc_auc_score,
    'recall_score': recall_score,
    'precision_score': precision_score,
}

    def _apply_metric(self, Y_true, Y_predict, metric):
        return self._metrics[metric](Y_true, Y_predict)

    def _apply_metric_pack(self, Y_true, Y_predict):
        ret = {}
        for key in self._metrics:
            try:
                ret[key] = self._apply_metric(Y_true, Y_predict, key)
            except BaseException as e:
                log_error_trace(getattr(self, 'log').warn, e,
                                head=f'while {self.__class__} execute score_pack, skip to applying metric {key}\n')
        return ret

    def score_pack(self, X, y):
        y = self.np_arr_to_index(y)
        return self._apply_metric_pack(y, getattr(self, 'predict')(X))


class DummyParamMixIN:
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return None


class ClfWrapperMixIn(Clf_metric_MixIn, Reformat_Ys_MixIn, PickleMixIn):
    pass
