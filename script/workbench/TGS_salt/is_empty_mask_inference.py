from pprint import pprint

from imgaug import augmenters as iaa
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from script.data_handler.ImgMaskAug import ActivatorMask, ImgMaskAug
from script.model.sklearn_like_model.BaseModel import BaseEpochCallback, BaseDatasetCallback
from script.model.sklearn_like_model.ImageClf import ImageClf
from script.model.sklearn_like_model.TFSummary import TFSummaryScalar
from script.model.sklearn_like_model.callback.EarlyStop import EarlyStop
from script.model.sklearn_like_model.callback.Top_k_save import Top_k_save
from script.model.sklearn_like_model.callback.TriangleLRScheduler import TriangleLRScheduler
from script.util.misc_util import time_stamp, path_join, to_dict
from script.util.numpy_utils import *
from script.workbench.TGS_salt.TGS_salt_inference import TGS_salt_DataHelper, plot, save_tf_summary_params

SUMMARY_PATH = f'./tf_summary/TGS_salt/empty_mask_clf'
INSTANCE_PATH = f'./instance/TGS_salt/empty_mask_clf'
PLOT_PATH = f'./matplot/TGS_salt/empty_mask_clf'


class collect_data_callback(BaseEpochCallback):
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def __call__(self, model, dataset, metric, epoch):
        self.train_loss = model.metric(self.train_x, self.train_y)
        self.test_predict = model.predict(self.test_x)
        self.train_predict = model.predict(self.train_x)

        test_y_decode = np_onehot_to_index(self.test_y)
        self.test_score = accuracy_score(test_y_decode, self.test_predict)
        self.test_confusion = confusion_matrix(test_y_decode, self.test_predict)

        train_y_decode = np_onehot_to_index(self.train_y)
        self.train_score = accuracy_score(train_y_decode, self.train_predict)
        self.train_confusion = confusion_matrix(train_y_decode, self.train_predict)


class summary_callback(BaseEpochCallback):
    def __init__(self, data_collection, run_id):
        self.data_collection = data_collection
        self.dc = data_collection

        self.run_id = run_id
        self.summary_train_loss = TFSummaryScalar(path_join(SUMMARY_PATH, self.run_id, 'train'), 'train_loss')
        self.summary_train_acc = TFSummaryScalar(path_join(SUMMARY_PATH, self.run_id, 'train'), 'train_acc')
        self.summary_test_acc = TFSummaryScalar(path_join(SUMMARY_PATH, self.run_id, 'test'), 'test_acc')

    def __call__(self, model, dataset, metric, epoch):
        sess = model.sess
        self.summary_train_loss.update(sess, self.dc.train_loss, epoch)
        self.summary_train_acc.update(sess, self.dc.train_score, epoch)
        self.summary_test_acc.update(sess, self.dc.test_score, epoch)


class log_callback(BaseEpochCallback):
    def __init__(self, data_collection, log=print):
        self.data_collection = data_collection
        self.dc = data_collection
        self.log = log

    def __call__(self, model, dataset, metric, epoch):
        self.log(
            f'e={epoch}, '
            f'train_score = {self.dc.train_score},\n'
            f'train_confusion = {self.dc.train_confusion},\n'
            f'test_score = {self.dc.test_score},\n'
            f'test_confusion ={self.dc.test_confusion}\n'
        )


class aug_callback(BaseDatasetCallback):
    def __init__(self, x, y, batch_size, n_job=2, q_size=100, enc=None):
        super().__init__(x, y, batch_size)

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential([
            # sometimes(
            #     iaa.OneOf([
            #         # iaa.PiecewiseAffine((0.002, 0.1), name='PiecewiseAffine'),
            #         iaa.Affine(rotate=(-10, 10)),
            #         iaa.Affine(shear=(-20, 20)),
            #         iaa.Affine(translate_percent=(0, 0.2), mode='symmetric'),
            #         iaa.Affine(translate_percent=(0, 0.2), mode='wrap'),
            #         # iaa.PerspectiveTransform((0.0, 0.3))
            #     ], name='affine')
            # ),
            iaa.Fliplr(0.5, name="horizontal flip"),
            # sometimes(iaa.Crop(percent=(0, 0.2), name='crop')),

            # image only
            # sometimes(
            #     iaa.OneOf([
            #         iaa.Add((-45, 45), name='bright'),
            #         iaa.Multiply((0.5, 1.5), name='contrast')]
            #     )
            # ),
            # sometimes(
            #     iaa.OneOf([
            #         iaa.AverageBlur((1, 5), name='AverageBlur'),
            #         # iaa.BilateralBlur(),
            #         iaa.GaussianBlur((0.1, 2), name='GaussianBlur'),
            #         # iaa.MedianBlur((1, 7), name='MedianBlur'),
            #     ], name='blur')
            # ),
            # scale to  128 * 128
            # iaa.Scale((128, 128), name='to 128 * 128'),
        ])
        self.activator = ActivatorMask(['bright', 'contrast', 'AverageBlur', 'GaussianBlur', 'MedianBlur'])
        self.aug = ImgMaskAug(self.x, self.y, self.seq, self.activator, self.batch_size, n_jobs=n_job, q_size=q_size)

        self.enc = enc

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    @property
    def size(self):
        return len(self.x)

    def shuffle(self):
        pass

    def next_batch(self, batch_size, batch_keys=None, update_cursor=True, balanced_class=False, out_type='concat'):
        x, y = self.aug.get_batch()
        y = y.reshape([batch_size, -1])
        y = np.mean(y, axis=1)
        y = y == 0
        y = y.reshape([-1, 1])
        y = self.enc.transform(y).toarray()

        return x[:batch_size], y[:batch_size]


class is_emtpy_mask_clf_pipeline:
    def __init__(self):
        self.plot = plot
        import random
        self.random_sate = random.randint(1, 1234567)
        self.init_dataset()
        self.init_aug_set()

    def init_dataset(self):
        self.data_helper = TGS_salt_DataHelper()
        # self.data_helper.train_set.y_keys = ['empty_mask']

        train_set = self.data_helper.train_set_with_depth_image
        train_set.y_keys = ['empty_mask']

        train_x, train_y = train_set.full_batch()
        train_x = train_x.reshape([-1, 101, 101, 1])
        train_y = train_y.reshape([-1, 1])
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        enc.fit(train_y)

        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            train_x, train_y, test_size=0.33, random_state=self.random_sate)
        self.enc = enc
        train_y_onehot = enc.transform(train_y).toarray()
        test_y_onehot = enc.transform(test_y).toarray()

        sample_size = 100
        sample_x = train_x[:sample_size]
        sample_y = train_y[:sample_size]
        sample_y_onehot = train_y_onehot[:sample_size]

        self.train_set = train_set
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.test_y_onehot = test_y_onehot
        self.train_y_onehot = train_y_onehot
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.sample_y_onehot = sample_y_onehot

    def init_aug_set(self):
        self.data_helper.train_set.y_keys = ['mask']
        train_set = self.data_helper.train_set
        x_full, y_full = train_set.full_batch()
        x_full = x_full.reshape([-1, 101, 101, 1])
        y_full = y_full.reshape([-1, 101, 101, 1])

        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            x_full, y_full, test_size=0.33, random_state=self.random_sate)

        self.aug_train_x = train_x
        self.aug_test_x = test_x
        self.aug_train_y = train_y
        self.aug_test_y = test_y

    def params(
            self,
            run_id=None,
            learning_rate=0.01,
            beta1=0.9,
            batch_size=100,
            net_type='VGG16',
            n_classes=2,
            capacity=64,
            use_l1_norm=False,
            l1_norm_rate=0.01,
            use_l2_norm=False,
            l2_norm_rate=0.01,
            dropout_rate=0.5,
            fc_depth=2,
            fc_capacity=1024,
            comment=''
    ):
        # net_type = 'InceptionV1'
        # net_type = 'InceptionV2'
        # net_type = 'InceptionV4'
        # net_type = 'ResNet18'
        # net_type = 'ResNet34'
        # net_type = 'ResNet50'
        # net_type = 'ResNet101'
        # net_type = 'ResNet152'

        if run_id is None:
            run_id = time_stamp()

        return to_dict(
            run_id=run_id,
            batch_size=batch_size,
            net_type=net_type,
            capacity=capacity,
            learning_rate=learning_rate,
            beta1=beta1,
            n_classes=n_classes,
            use_l1_norm=use_l1_norm,
            l1_norm_rate=l1_norm_rate,
            use_l2_norm=use_l2_norm,
            l2_norm_rate=l2_norm_rate,
            dropout_rate=dropout_rate,
            fc_depth=fc_depth,
            fc_capacity=fc_capacity,
            comment=comment,
        )

    def train(self, params, n_epoch, augmentation=False, path=None, callbacks=None):
        save_tf_summary_params(SUMMARY_PATH, params)
        pprint(f'train {params}')

        if path:
            clf = ImageClf().load(path)
            clf.build(Xs=self.train_x, Ys=self.train_y_onehot)
            clf.restore(path)
        else:
            clf = ImageClf(**params)
            clf.build(Xs=self.train_x, Ys=self.train_y_onehot)

        print('build callback')
        dc = collect_data_callback(
            self.train_x, self.train_y_onehot,
            self.test_x, self.test_y_onehot
        )
        callbacks = [
            dc,
            log_callback(dc),
            summary_callback(dc, clf.run_id),
            Top_k_save(path_join(INSTANCE_PATH, clf.run_id, 'top_k'), k=1, dc=dc, key='test_score'),
            TriangleLRScheduler(7, 0.001, 0.0005),
            EarlyStop(16),
        ]

        if augmentation:
            dataset_callback = aug_callback(
                self.aug_train_x,
                self.aug_train_y,
                params['batch_size'],
                enc=self.enc
            )
        else:
            dataset_callback = None
        clf.update_learning_rate(0.001)
        clf.train(
            self.train_x, self.train_y_onehot, epoch=n_epoch, epoch_callbacks=callbacks,
            dataset_callback=dataset_callback
        )
