from inspect import signature


class async_class_code_builder:
    import_code = """
from script.util.JobPool import JobPool
import multiprocessing as mp

CPU_COUNT = mp.cpu_count() - 1

"""
    class_code = """
class async{class_name}(JobPool):
"""
    init_method_code = """
    def __init__(self, n_job=CPU_COUNT):
        super().__init__(n_job)
        self.instance = None
"""
    method_code = """
    def {func_name}({func_signature}):
        args = {args_names} 
        kwargs.update({kwargs_names})

        self.apply_async(self.instance.{func_name}, args=args, kwargs=kwargs)

"""

    @staticmethod
    def _split_args_and_kwargs(sig):
        kwargs_name = []
        args_name = []
        for param in sig.parameters.values():
            if '=' in str(param):
                kwargs_name += [param]
            else:
                args_name += [param]

        args_name = [str(v) for v in args_name if '**' not in str(v)]
        kwargs_name = [str(v).split('=')[0] for v in kwargs_name]
        return args_name, kwargs_name

    @staticmethod
    def _name_to_dict_format(names):
        code = '{'
        for name in names:
            code += f"'{name}':{name},"
        code += '}'
        return code

    def build(self, cls):
        class_name = cls.__name__
        code = self.import_code
        code += self.class_code.format(class_name=class_name)
        code += self.init_method_code

        class_dict = cls.__dict__

        funcs = {key: val for key, val in class_dict.items() if callable(val)}

        for key, val in funcs.items():
            func_name = key

            sig = signature(val)
            func_signature = str(sig)[1:-1]

            args, kwargs = self._split_args_and_kwargs(sig)
            args_self, args = args[0], args[1:]
            args_names = str(args).replace("'", '')
            kwargs_names = self._name_to_dict_format(kwargs)

            method_code = self.method_code.format(
                class_name=class_name,
                func_name=func_name,
                args_names=args_names,
                kwargs_names=kwargs_names,
                func_signature=func_signature,
            )

            code += method_code

        path = './gencode.py'
        if path is not None:
            with open(path, mode='w', encoding='UTF8') as f:
                f.write(code)
