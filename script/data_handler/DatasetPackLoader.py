from script.data_handler.BaseDatasetPack import BaseDatasetPack
from env_settting import *
from script.util.Logger import Logger
from script.util.misc_util import *


class DatasetPackLoader:

    def __init__(self, root_path=LOG_PATH, verbose='INFO'):
        self.root_path = root_path
        self.logger = Logger(self.__class__.__name__, self.root_path, level=verbose)
        self.log = self.logger

    def __repr__(self):
        return self.__class__.__name__

    def load_dataset(self, dataset_name, limit=None, **kwargs):
        """load dataset, return dataset, input_shapes

        :type dataset_name: str
        :type limit: int
        :param dataset_name: dataset name to load
        :param limit: limit dataset_size

        :return: dataset, input_shapes

        :raise KeyError
        invalid dataset_name
        """
        try:
            dataset_class = self.import_dataset_class(dataset_name=dataset_name)
            path = os.path.join(DATA_PATH, dataset_class.__name__)
            dataset = dataset_class(**kwargs)
            dataset.load(path=path, limit=limit)

        except KeyError:
            raise KeyError("dataset_name %s not found" % dataset_name)

        return dataset

    def import_dataset_class(self, dataset_name):
        """ import dataset class by dataset name

        :type dataset_name: str
        :param dataset_name:
        """
        self.log.info('load %s dataset module' % dataset_name)
        paths = glob(os.path.join(DATA_HANDLER_PATH, '**', '*.py'), recursive=True)

        dataset_path = None
        for path in paths:
            _, file_name = os.path.split(path)
            dataset_name_ = file_name.replace('.py', '')
            if dataset_name != dataset_name_:
                continue
            dataset_path = path

        if dataset_path is None:
            raise ModuleNotFoundError("dataset %s not found" % dataset_name)

        module_ = import_module_from_module_path(dataset_path)
        dataset_cls = None
        for key in module_.__dict__:
            value = module_.__dict__[key]
            try:
                if issubclass(value, BaseDatasetPack):
                    dataset_cls = value
            except TypeError:
                pass

        if dataset_cls is None:
            raise ModuleNotFoundError("dataset class %s not found" % dataset_name)

        return dataset_cls
