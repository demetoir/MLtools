from imgaug import augmenters as iaa
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from script.data_handler.ImgMaskAug import ActivatorMask, ImgMaskAug
from script.model.sklearn_like_model.BaseModel import BaseDatasetCallback
from script.model.sklearn_like_model.ImageClf import ImageClf
from script.model.sklearn_like_model.TFSummary import TFSummaryScalar
from script.model.sklearn_like_model.callback.BaseEpochCallback import BaseEpochCallback
from script.model.sklearn_like_model.callback.BestSave import BestSave
from script.model.sklearn_like_model.callback.ReduceLrOnPlateau import ReduceLrOnPlateau
from script.util.misc_util import time_stamp, path_join, to_dict
from script.util.numpy_utils import *
from script.workbench.TGS_salt.TGS_salt_inference import TGS_salt_DataHelper

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
            f'\n'
            f'e={epoch}, '
            f'train_score = {self.dc.train_score},\n'
            f'train_confusion\n'
            f'{self.dc.train_confusion}\n'
            f'test_score = {self.dc.test_score},\n'
            f'test_confusion\n'
            f'{self.dc.test_confusion}\n'
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


class is_emtpy_mask_clf_baseline:
    def init_dataset(self, fold=7, fold_index=3):
        helper = TGS_salt_DataHelper()
        train_set = helper.train_set
        train_set = helper.add_depth_image_channel(train_set)
        # train_set = helper.get_non_empty_mask(train_set)
        train_set = helper.lr_flip(train_set)
        train_set.y_keys = ['empty_mask']
        train_set.x_keys = ['x_with_depth']
        train_set, hold_out = helper.split_hold_out(train_set)
        self.hold_out_set = hold_out
        kfold_sets = helper.k_fold_split(train_set, k=fold)
        train_set_fold1, valid_set_fold1 = kfold_sets[fold_index]

        print(f' {fold} fold, index {fold_index}')
        print('train_set')
        print(train_set_fold1)
        print('valid_set')
        print(valid_set_fold1)

        train_x, train_y = train_set_fold1.full_batch()
        valid_x, valid_y = valid_set_fold1.full_batch()

        return train_x, train_y, valid_x, valid_y

    def params(
            self,
            run_id=None,
            learning_rate=0.01,
            beta1=0.9,
            batch_size=64,
            net_type='InceptionV1',
            n_classes=2,
            capacity=4,
            use_l1_norm=False,
            l1_norm_rate=0.01,
            use_l2_norm=False,
            l2_norm_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
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

    def encode_x(self, x):
        return x / 255.

    def encode_y(self, y):
        y = y.reshape([-1, 1])
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        enc.fit(y)
        y = enc.transform(y).toarray()

        return y

    def encode(self, x, y):
        x = self.encode_x(x)
        y = self.encode_y(y)
        return x, y

    def encode_datas(self, datas):
        train_x, train_y, valid_x, valid_y = datas

        train_x_enc = self.encode_x(train_x)
        valid_x_enc = self.encode_x(valid_x)
        train_y_enc = self.encode_y(train_y)
        valid_y_enc = self.encode_y(valid_y)

        return train_x_enc, train_y_enc, valid_x_enc, valid_y_enc

    def train(self, n_epoch=None, callbacks=None, datas=None):
        clf = self.model

        if datas is None:
            datas = self.init_dataset()

        train_x_enc, train_y_onehot, valid_x_enc, valid_y_onehot = self.encode_datas(datas)

        if callbacks is None:
            dc = collect_data_callback(
                train_x_enc, train_y_onehot,
                valid_x_enc, valid_y_onehot
            )
            callbacks = [
                dc,
                log_callback(dc),
                summary_callback(dc, clf.run_id),
                BestSave(path_join(INSTANCE_PATH, clf.run_id), max_best=True).trace_on(dc, 'test_score'),
                # TriangleLRScheduler(7, 0.001, 0.0005),
                ReduceLrOnPlateau(0.5, 5, 0.0001, min_best=False).trace_on(dc, 'test_score'),
                # EarlyStop(10).trace_on(dc, 'test_score'),
            ]

        n_epoch = 50
        # clf.init_adam_momentum()
        clf.update_learning_rate(0.01)
        clf.train(
            train_x_enc, train_y_onehot, epoch=n_epoch, epoch_callbacks=callbacks,
        )

    def fold_train(self, epoch=50, k=7):

        models = []
        for fold in range(2, k):
            clf = self.new_model()
            models += [clf]

            datas = self.init_dataset(k, fold)
            train_x_enc, train_y_onehot, valid_x_enc, valid_y_onehot = self.encode_datas(datas)

            dc = collect_data_callback(
                train_x_enc, train_y_onehot,
                valid_x_enc, valid_y_onehot
            )
            callbacks = [
                dc,
                log_callback(dc),
                summary_callback(dc, clf.run_id),
                BestSave(path_join(INSTANCE_PATH, f'fold_{fold}'), max_best=True).trace_on(dc, 'test_score'),
                # TriangleLRScheduler(7, 0.001, 0.0005),
                ReduceLrOnPlateau(0.7, 5, 0.0001),
                # EarlyStop(16),
            ]

            clf.update_learning_rate(0.01)
            clf.train(
                train_x_enc, train_y_onehot, epoch=epoch, epoch_callbacks=callbacks
            )

    def fold_score(self, k=7):
        models = []

        train_scores = []
        valid_scores = []

        hold_out_predicts = []
        hold_out_probas = []
        for fold in range(k):
            path = f'./instance/TGS_salt/empty_mask_clf/fold_{fold}'
            clf = self.load_model(path)
            models += [clf]

            datas = self.init_dataset(k, fold)
            train_x_enc, train_y_onehot, valid_x_enc, valid_y_onehot = self.encode_datas(datas)
            train_score = clf.score(train_x_enc, train_y_onehot)
            valid_score = clf.score(valid_x_enc, valid_y_onehot)

            holdout_x, holdout_y = self.hold_out_set.full_batch()
            holdout_x_enc, holdout_y_enc = self.encode(holdout_x, holdout_y)

            hold_out_proba = clf.predict_proba(holdout_x_enc)
            hold_out_predict = clf.predict(holdout_x_enc)
            hold_out_predicts += [hold_out_predict]
            hold_out_probas += [hold_out_proba]

            train_scores += [train_score]
            valid_scores += [valid_score]

        print(f'train score')
        for i, score in enumerate(train_scores):
            print(i, score)

        print(f'valid score')
        for i, score in enumerate(valid_scores):
            print(i, score)

        hold_out_probas = np.array(hold_out_probas)
        probas = np.sum(hold_out_probas, axis=0)
        probas /= k
        predict = np.argmax(probas, axis=1)
        from sklearn.metrics import accuracy_score
        score = accuracy_score(holdout_y, predict)
        print(f'ensemble soft score = {score}')

    def fold_predict(self, k=7):
        helper = TGS_salt_DataHelper()
        test_set = helper.test_set
        test_set = helper.add_depth_image_channel(test_set)
        test_set.x_keys = ['x_with_depth']
        x = test_set.full_batch(['x_with_depth'])['x_with_depth']
        id = test_set.full_batch(['id'])['id']
        print(id)

        x_enc = self.encode_x(x)

        probas = []
        for fold in range(k):
            path = f'./instance/TGS_salt/empty_mask_clf/fold_{fold}'
            clf = self.load_model(path)

            proba = clf.predict_proba(x_enc)

            print(proba.shape)
            probas += [proba]

        probas = np.array(probas)
        probas = np.sum(probas, axis=0)
        probas /= k
        print(probas.shape)
        predict = np.argmax(probas, axis=1)
        print(predict.shape)
        np.save(f'./empty_mask_predict.np', predict)
        return predict

    def load_model(self, path):
        clf = ImageClf().load_meta(path)
        clf.build(x=(101, 101, 2), y=(2,))
        clf.restore(path)
        self.model = clf
        return clf

    def new_model(self):
        params = self.params()
        clf = ImageClf(**params)
        clf.build(x=(101, 101, 2), y=(2,))
        self.model = clf
        return clf

    def load_baseline(self):
        path = f'./instance/TGS_salt/empty_mask_clf/fold_3'
        self.load_model(path)

    def new_train(self):
        # top base line but suspicious 0.94 test
        path = f'./instance/TGS_salt/empty_mask_clf/inceptionv1'
        pipe = is_emtpy_mask_clf_baseline()
        params = pipe.params(
            capacity=4,
            net_type='InceptionV1',
            batch_size=64,
            learning_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
            comment='inception v1',
        )
        self.model.train()
        pipe.train(params)

    def top_baseline(self):
        # top base line but suspicious 0.94 test
        path = f'./instance/TGS_salt/empty_mask_clf/inceptionv1'
        pipe = is_emtpy_mask_clf_baseline()
        params = pipe.params(
            capacity=4,
            net_type='InceptionV1',
            batch_size=64,
            learning_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
            comment='inception v1',
        )
        pipe.train(params, )

    def inceptionv1(self):
        # top base line but suspicious 0.94 test
        path = f'./instance/TGS_salt/empty_mask_clf/inceptionv1'
        pipe = is_emtpy_mask_clf_baseline()
        params = pipe.params(
            capacity=4,
            net_type='InceptionV1',
            batch_size=64,
            learning_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
            comment='inception v1',
        )
        pipe.train(params, )

    def inceptionv2(self):
        path = f'./instance/TGS_salt/empty_mask_clf/inceptionv2'
        pipe = is_emtpy_mask_clf_baseline()
        params = pipe.params(
            capacity=4,
            net_type='InceptionV2',
            # net_type='ResNet18',
            batch_size=32,
            learning_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
            comment='inception v1',
        )
        pipe.train(params, )

    def Resnet34(self):
        path = f'./instance/TGS_salt/empty_mask_clf/resnet34'
        pipe = is_emtpy_mask_clf_baseline()
        params = pipe.params(
            capacity=4,
            net_type='ResNet34',
            batch_size=32,
            learning_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
            comment='inception v1',
        )
        pipe.train(params, )

    def Resnet50(self):
        path = f'./instance/TGS_salt/empty_mask_clf/resnet50'
        pipe = is_emtpy_mask_clf_baseline()
        params = pipe.params(
            capacity=4,
            net_type='ResNet50',
            batch_size=32,
            learning_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
            comment='inception v1',
        )
        pipe.train(params, )
