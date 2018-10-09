from pprint import pprint

import numpy as np
from tqdm import tqdm

from script.data_handler.TGS_salt import mask_label_encoder
from script.model.sklearn_like_model.BaseModel import BaseDataCollector
from script.model.sklearn_like_model.SemanticSegmentation import SemanticSegmentation
from script.model.sklearn_like_model.TFSummary import TFSummaryScalar
from script.model.sklearn_like_model.callback.BaseEpochCallback import BaseEpochCallback
from script.model.sklearn_like_model.callback.Top_k_save import Top_k_save
from script.util.misc_util import time_stamp, path_join, to_dict
from script.workbench.TGS_salt.TGS_salt_inference import plot, TGS_salt_DataHelper, \
    masks_rate, Metrics

SUMMARY_PATH = f'./tf_summary/TGS_salt/SS'
INSTANCE_PATH = f'./instance/TGS_salt/SS'
PLOT_PATH = f'./matplot/TGS_salt/SS'


class CollectDataCallback(BaseDataCollector):
    def __init__(self, test_x_enc, test_y_enc, **kwargs):
        super().__init__(**kwargs)
        self.test_x_enc = test_x_enc
        self.test_y_enc = test_y_enc

        def decode(x):
            return x * 255

        self.test_x_dec = decode(self.test_x_enc)
        self.test_y_dec = decode(self.test_y_enc)

    def update_data(self, model, dataset, metric, epoch):
        def decode(x):
            return x * 255

        self.train_loss = metric

        train_x_enc, train_y_enc = dataset.next_batch(500, update_cursor=False)
        self.train_x_enc = train_x_enc
        self.train_x_dec = decode(self.train_x_enc)
        self.train_y_enc = train_y_enc
        self.train_y_dec = decode(self.train_y_enc)

        self.train_predict_enc = model.predict(self.train_x_enc)
        self.train_predict_dec = decode(self.train_predict_enc)

        self.test_predict_enc = model.predict(self.test_x_enc)
        self.test_predict_dec = decode(self.test_predict_enc)

        self.train_score = Metrics.TGS_salt_score(self.train_y_enc, self.train_predict_enc)
        self.test_score = Metrics.TGS_salt_score(self.test_y_enc, self.test_predict_enc)

        self.train_iou_score = Metrics.miou(self.train_y_enc, self.train_predict_enc)
        self.test_iou_score = Metrics.miou(self.test_y_enc, self.test_predict_enc)

        self.train_predict_sample = self.train_predict_dec[:20]


class LoggingCallback(BaseEpochCallback):
    def __init__(self, data_collection):
        super().__init__()
        self.data_collection = data_collection
        self.dc = self.data_collection

    def log_TGS_salt_metric(self, epoch):
        msg = f'\n'
        msg += f'e:{epoch}, '

        msg += f'TGS_salt_metric score\n'
        msg += f'train        = {self.dc.train_score}\n'
        msg += f'test         = {self.dc.test_score}\n'

        msg += f'iou score\n'
        msg += f'train        = {self.dc.train_iou_score}\n'
        msg += f'test         = {self.dc.test_iou_score}\n'
        msg += f'\n'

        tqdm.write(msg)

    def __call__(self, model, dataset, metric, epoch):
        self.log_TGS_salt_metric(epoch)


class TFSummaryCallback(BaseEpochCallback):
    def __init__(self, data_collection, run_id, **kwargs):
        self.data_collection = data_collection
        self.run_id = run_id
        self.kwargs = kwargs

        base_path = path_join(SUMMARY_PATH, self.run_id)
        test_path = path_join(base_path, 'test')
        train_path = path_join(base_path, 'train')
        self.test_path = test_path
        self.train_path = train_path
        self.summary_train_loss = TFSummaryScalar(train_path, 'train_loss')
        self.summary_train_acc = TFSummaryScalar(train_path, 'train_acc')
        self.summary_test_acc = TFSummaryScalar(test_path, 'test_acc')
        self.summary_iou_train_score = TFSummaryScalar(train_path, 'train_iou')
        self.summary_iou_test_score = TFSummaryScalar(test_path, 'test_iou')

    def __call__(self, model, dataset, metric, epoch):
        sess = model.sess
        train_loss = self.data_collection.train_loss
        train_score = self.data_collection.train_score
        test_score = self.data_collection.test_score

        train_iou_score = self.data_collection.train_iou_score
        test_iou_score = self.data_collection.test_iou_score

        self.summary_train_loss.update(sess, train_loss, epoch)
        self.summary_train_acc.update(sess, train_score, epoch)
        self.summary_test_acc.update(sess, test_score, epoch)
        self.summary_iou_train_score.update(sess, train_iou_score, epoch)
        self.summary_iou_test_score.update(sess, test_iou_score, epoch)


class SummaryAllCallback(BaseEpochCallback):
    def __init__(self, data_collection, run_id, **kwargs):
        self.data_collection = data_collection
        self.run_id = run_id
        self.kwargs = kwargs

        base_path = path_join(SUMMARY_PATH, self.run_id)
        test_path = path_join(base_path, 'test')
        train_path = path_join(base_path, 'train')
        self.test_path = test_path
        self.train_path = train_path
        self.summary_train_loss = TFSummaryScalar(train_path, 'train_loss')
        self.summary_train_acc = TFSummaryScalar(train_path, 'train_acc')
        self.summary_test_acc = TFSummaryScalar(test_path, 'test_acc')
        self.summary_iou_train_score = TFSummaryScalar(train_path, 'train_iou')
        self.summary_iou_test_score = TFSummaryScalar(test_path, 'test_iou')

    def __call__(self, model, dataset, metric, epoch):
        sess = model.sess
        train_loss = self.data_collection.train_loss
        train_score = self.data_collection.train_score
        test_score = self.data_collection.test_score

        train_iou_score = self.data_collection.train_iou_score
        test_iou_score = self.data_collection.test_iou_score

        self.summary_train_loss.update(sess, train_loss, epoch)
        self.summary_train_acc.update(sess, train_score, epoch)
        self.summary_test_acc.update(sess, test_score, epoch)
        self.summary_iou_train_score.update(sess, train_iou_score, epoch)
        self.summary_iou_test_score.update(sess, test_iou_score, epoch)


class PlotToolsCallback(BaseEpochCallback):
    def __init__(self, data_collection, **kwargs):
        self.plot = plot
        self.data_collection = data_collection
        self.kwargs = kwargs

    def plot_mask_image(self, model, dataset, metric, epoch):
        run_id = model.run_id

        x, y = dataset.next_batch(20, update_cursor=False)

        predict = model.predict(x)
        predict = predict * 255
        proba = model.predict_proba(x)
        proba = proba.reshape([-1, 101, 101, 1]) * 255
        predict = predict.reshape([-1, 101, 101, 1])

        def scramble_column(*args, size=10):
            ret = []
            for i in range(0, len(args[0]), size):
                for j in range(len(args)):
                    ret += [args[j][i:i + size]]

            return np.concatenate(ret, axis=0)

        np_tile = scramble_column(x, y, predict, proba)
        self.plot.plot_image_tile(
            np_tile,
            title=f'predict_epoch({epoch})',
            column=10,
            path=path_join(PLOT_PATH, run_id, f'predict_mask/({epoch}).png'))

    def plot_non_mask_rate_iou(self, model, dataset, metric, epoch):
        run_id = model.run_id
        test_y = self.data_collection.test_y_dec
        test_predict = self.data_collection.test_predict_dec

        size = 512
        test_y = test_y[:size]
        test_predict = test_predict[:size]

        xs = masks_rate(test_y)
        xs = xs.reshape([-1])
        xs /= 255

        ys = np.array([Metrics.miou(true, predict) for true, predict in zip(test_y, test_predict)])

        dots = np.array([[x, y] for x, y in zip(xs, ys)])

        self.plot.scatter_2d(
            dots,
            title=f'test set mask rate and iou',
            path=path_join(PLOT_PATH, run_id, f'test_set_mask_rate_iou/({epoch}).png'),
            x_label='mask_rate',
            y_label='iou'
        )

    def __call__(self, model, dataset, metric, epoch):
        self.plot_mask_image(model, dataset, metric, epoch)
        self.plot_non_mask_rate_iou(model, dataset, metric, epoch)


class SS_baseline:
    """
        param = pipe.params(
        stage=4,
        net_type='FusionNet',
        batch_size=32,
        dropout_rate=0.5,
        learning_rate=0.01,
        depth=2,
        loss_type='BCE+dice_soft',
        comment='change dropout location'


    """

    def params(
            self,
            run_id=None,
            verbose=10,
            learning_rate=0.01,
            beta1=0.9,
            batch_size=32,
            stage=4,
            loss_type='BCE+dice_soft',
            n_classes=1,
            net_type='FusionNet',
            capacity=16,
            depth=2,
            dropout_rate=0.5,
            comment=''
    ):
        # loss_type = 'pixel_wise_softmax'
        # loss_type = 'iou'
        # loss_type = 'dice_soft'
        # loss_type = 'BCE+dice_soft'

        if run_id is None:
            run_id = time_stamp()

        params = to_dict(
            run_id=run_id,
            verbose=verbose,
            learning_rate=learning_rate,
            beta1=beta1,
            batch_size=batch_size,
            stage=stage,
            net_type=net_type,
            loss_type=loss_type,
            capacity=capacity,
            n_classes=n_classes,
            depth=depth,
            dropout_rate=dropout_rate,
            comment=comment
        )
        return params

    def prepare_set(self):

        helper = TGS_salt_DataHelper()
        train_set = helper.train_set
        # train_set = helper.get_non_empty_mask(train_set)
        train_set = helper.lr_flip(train_set)
        train_set, hold_out = helper.split_hold_out(train_set)
        kfold_sets = helper.k_fold_split(train_set)
        train_set_fold1, valid_set_fold1 = kfold_sets[1]

        print('train_set')
        print(train_set_fold1)
        print('valid_set')
        print(valid_set_fold1)

        train_x, train_y = train_set_fold1.full_batch()
        valid_x, valid_y = valid_set_fold1.full_batch()

        return train_x, train_y, valid_x, valid_y

    def train(self):
        def encode(x):
            return x / 255.

        def decode(x):
            return x * 255

        train_x, train_y, valid_x, valid_y = self.prepare_set()

        train_x_enc = encode(train_x)
        train_y_enc = encode(train_y)
        valid_x_enc = encode(valid_x)
        valid_y_enc = encode(valid_y)

        model = self.model
        pprint(model)

        run_id = model.run_id
        dc_callback = CollectDataCallback(valid_x_enc, valid_y_enc)
        callbacks = [
            dc_callback,
            Top_k_save(
                path_join(INSTANCE_PATH, run_id, 'test_score'),
                k=1,
                name='test_score',
                save_model=True
            ).trace_on(
                dc_callback,
                'test_score'
            ),
            LoggingCallback(dc_callback),
            TFSummaryCallback(dc_callback, run_id),
            PlotToolsCallback(dc_callback),

            # EarlyStop(
            #     20, min_best=False
            # ).trace_on(
            #     dc_callback, 'test_iou_score'
            # ),
            # ReduceLrOnPlateau(0.5, 5, 0.0001, min_best=False).trace_on(dc_callback, 'test_iou_score'),
            # TriangleLRScheduler(10, 0.01, 0.001),
        ]

        # model.init_adam_momentum()
        # model.update_learning_rate(0.01)
        model.update_learning_rate(0.01)
        # model.update_dropout_rate(1)
        pprint(model)

        epoch = 50
        model.train(
            train_x_enc, train_y_enc, epoch=epoch,
            epoch_callbacks=callbacks,
            # dataset_callback=aug_callback
        )

    def new_model(self):
        params = self.params()
        # import tensorflow as tf
        # from tensorflow.python import debug as tf_debug
        # sess = tf.Session()
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "demetoir-desktop:7777")
        # params.update({'sess': sess})
        self.model = SemanticSegmentation(**params)

        self.model.build(x=(101, 101, 1), y=(101, 101, 1))

    def load_baseline(self):
        path = './instance/TGS_salt/SS/baseline'
        self.load_model(path)

    def load_model(self, path):
        self.model = SemanticSegmentation().load_meta(path)
        self.model.build(x=(101, 101, 1), y=(101, 101, 1))
        self.model.restore(path)

    @staticmethod
    def scramble_column(*args, size=10):
        ret = []
        for i in range(0, len(args[0]), size):
            for j in range(len(args)):
                ret += [args[j][i:i + size]]

        return np.concatenate(ret, axis=0)

    def log_score(self):
        baseline = self.model

        # train_predict = baseline.predict(pipe.train_x)
        # train_predict = mask_label_encoder.from_label(train_predict)
        #
        # test_predict = baseline.predict(pipe.valid_x)
        # test_predict = mask_label_encoder.from_label(test_predict)
        #
        # train_loss = baseline.metric(pipe.train_x, pipe.train_y_encode)
        # test_loss = baseline.metric(pipe.valid_x, pipe.valid_y_encode)
        #
        # train_proba = baseline.predict_proba(pipe.train_x)
        # test_proba = baseline.predict_proba(pipe.valid_x)
        #
        # train_score = Metrics.TGS_salt_score(pipe.train_y, train_predict)
        # test_score = Metrics.TGS_salt_score(pipe.valid_y, test_predict)
        #
        # train_mask_rate = masks_rate(pipe.train_y)
        # test_mask_rate = masks_rate(pipe.valid_y)
        #
        # train_miou = Metrics.miou(pipe.train_y, train_predict)
        # test_miou = Metrics.miou(pipe.valid_y, test_predict)
        #
        # train_ious = Metrics.ious(pipe.train_y, train_predict)
        # test_ious = Metrics.ious(pipe.valid_y, test_predict)

        # print(
        #     f'train TGS score = {train_score}\n'
        #     f'test TGS score = {test_score}\n'
        #     f'train miou = {train_miou}\n'
        #     f'test miou = {test_miou}\n'
        #     f'train loss = {train_loss}\n'
        #     f'test loss = {test_loss}\n'
        # )

    def log_split_mask_rate_score(self):

        pipe = self.load_baseline()
        baseline = pipe.model

        helper = pipe.data_helper
        train_set = helper.train_set_non_empty_mask_with_depth_image_under_1p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_under_1 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_under_1 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_under_5p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_under_5 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_under_5 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_under_10p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_under_10 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_under_10 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_under_20p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_under_20 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_under_20 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_upper_1p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_upper_1 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_upper_1 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_upper_5p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_upper_5 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_upper_5 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_upper_10p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_upper_10 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_upper_10 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_upper_20p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_upper_20 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_upper_20 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        print(
            f'\n'
            f' TGS score\n'
            f'under 1 {score_under_1}\n'
            f'under 5 {score_under_5}\n'
            f'under 10 {score_under_10}\n'
            f'under 20 {score_under_20}\n'
            f'upper 1 {score_upper_1}\n'
            f'upper 5 {score_upper_5}\n'
            f'upper 10 {score_upper_10}\n'
            f'upper 20 {score_upper_20}\n'

            f'\n'
            f'\n'
            f' iou score\n'
            f'under 1 {iou_score_under_1}\n'
            f'under 5 {iou_score_under_5}\n'
            f'under 10 {iou_score_under_10}\n'
            f'under 20 {iou_score_under_20}\n'
            f'upper 1 {iou_score_upper_1}\n'
            f'upper 5 {iou_score_upper_5}\n'
            f'upper 10 {iou_score_upper_10}\n'
            f'upper 20 {iou_score_upper_20}\n'
            f'\n'
        )

    def plot_test_set_sorted_by_iou(self):
        pipe = self.load_baseline()
        baseline = pipe.model

        train_predict = baseline.predict(pipe.train_x)
        train_predict = mask_label_encoder.from_label(train_predict)

        test_predict = baseline.predict(pipe.valid_x)
        test_predict = mask_label_encoder.from_label(test_predict)

        train_loss = baseline.metric(pipe.train_x, pipe.train_y_encode)
        test_loss = baseline.metric(pipe.valid_x, pipe.valid_y_encode)

        train_proba = baseline.predict_proba(pipe.train_x)
        test_proba = baseline.predict_proba(pipe.valid_x)

        train_score = Metrics.TGS_salt_score(pipe.train_y, train_predict)
        test_score = Metrics.TGS_salt_score(pipe.valid_y, test_predict)

        train_mask_rate = masks_rate(pipe.train_y)
        test_mask_rate = masks_rate(pipe.valid_y)

        train_miou = Metrics.miou(pipe.train_y, train_predict)
        test_miou = Metrics.miou(pipe.valid_y, test_predict)

        train_ious = Metrics.iou(pipe.train_y, train_predict)
        test_ious = Metrics.iou(pipe.valid_y, test_predict)

        test_x = pipe.valid_x
        test_y = pipe.valid_y
        test_predict = test_predict
        test_mask_rate = test_mask_rate

        zipped = zip(test_x, test_y, test_predict, test_ious)
        sort = list(sorted(zipped, key=lambda a: a[3]))
        test_x, test_y, test_predict, test_ious = zip(*sort)
        print(test_ious)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_predict = np.array(test_predict)
        test_x_image = test_x[:, :, :, 0]
        test_x_image = test_x_image.reshape([-1, 101, 101, 1])
        test_y = test_y.reshape([-1, 101, 101, 1])
        test_predict = test_predict.reshape([-1, 101, 101, 1])

        tile_image = self.scramble_column(test_x_image, test_y, test_predict)
        plot.plot_image_tile(tile_image, title='all', path=f'./matplot/baseline/test_set_sorted_by_iou.png')

    def test_set_sorted_by_mask_rate(self):
        pipe = self.load_baseline()
        baseline = pipe.model

        train_predict = baseline.predict(pipe.train_x)
        train_predict = mask_label_encoder.from_label(train_predict)

        test_predict = baseline.predict(pipe.valid_x)
        test_predict = mask_label_encoder.from_label(test_predict)

        train_loss = baseline.metric(pipe.train_x, pipe.train_y_encode)
        test_loss = baseline.metric(pipe.valid_x, pipe.valid_y_encode)

        train_proba = baseline.predict_proba(pipe.train_x)
        test_proba = baseline.predict_proba(pipe.valid_x)

        train_score = Metrics.TGS_salt_score(pipe.train_y, train_predict)
        test_score = Metrics.TGS_salt_score(pipe.valid_y, test_predict)

        train_mask_rate = masks_rate(pipe.train_y)
        test_mask_rate = masks_rate(pipe.valid_y)

        train_miou = Metrics.miou(pipe.train_y, train_predict)
        test_miou = Metrics.miou(pipe.valid_y, test_predict)

        train_ious = Metrics.iou(pipe.train_y, train_predict)
        test_ious = Metrics.iou(pipe.valid_y, test_predict)

        test_x = pipe.valid_x
        test_y = pipe.valid_y
        test_predict = test_predict
        test_mask_rate = test_mask_rate

        zipped = zip(test_x, test_y, test_predict, test_mask_rate)
        sort = list(sorted(zipped, key=lambda a: a[3]))
        test_x, test_y, test_predict, test_mask_rate = zip(*sort)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_predict = np.array(test_predict)
        test_mask_rate = np.array(test_mask_rate)
        test_x_image = test_x[:, :, :, 0]
        test_x_image = test_x_image.reshape([-1, 101, 101, 1])
        test_y = test_y.reshape([-1, 101, 101, 1])
        test_predict = test_predict.reshape([-1, 101, 101, 1])

        # plot_path = f'./matplot/baseline'
        tile_image = self.scramble_column(test_x_image, test_y, test_predict)
        plot.plot_image_tile(tile_image, title='all', path=f'./matplot/baseline/test_set_sorted_by_mask_rate.png')
