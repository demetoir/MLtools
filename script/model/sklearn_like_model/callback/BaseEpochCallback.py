class metaEpochCallback(type):
    """Metaclass for hook inherited class's function
    metaclass ref from 'https://code.i-harness.com/ko/q/11fc307'
    """

    def __init__(cls, name, bases, cls_dict):
        type.__init__(cls, name, bases, cls_dict)

        # hook __call__
        f_name = '__call__'
        if f_name in cls_dict:
            func = cls_dict[f_name]

            def new_func(self, model, dataset, metric, epoch, *args, **kwargs):
                if getattr(self, 'is_trance_on', False):
                    dc = getattr(self, 'dc')
                    dc_key = getattr(self, 'dc_key')
                    metric = getattr(dc, dc_key)

                return func(self, model, dataset, metric, epoch, *args, **kwargs)

            new_func.__name__ = f_name + '_wrap'
            setattr(cls, f_name, new_func)

        def wrap_return_self(f_name, cls_dict, cls):
            func = cls_dict[f_name]

            def new_func(self, *args, **kwargs):
                func(self, *args, **kwargs)
                return self

            new_func.__name__ = f_name + '_wrap'
            setattr(cls, f_name, new_func)

        # hook return self
        f_name = 'trace_on'
        if f_name in cls_dict:
            wrap_return_self(f_name, cls_dict, cls)

        f_name = 'trace_off'
        if f_name in cls_dict:
            wrap_return_self(f_name, cls_dict, cls)


class BaseEpochCallback(metaclass=metaEpochCallback):
    def __call__(self, model, dataset, metric, epoch):
        raise NotImplementedError

    def trace_on(self, dc, key):
        self.dc = dc
        self.dc_key = key
        self.is_trance_on = True

    def trace_off(self):
        del self.dc
        del self.dc_key
        self.is_trance_on = False