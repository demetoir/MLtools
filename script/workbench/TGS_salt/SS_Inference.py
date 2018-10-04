from pprint import pprint

import numpy as np
from tqdm import tqdm

from script.data_handler.TGS_salt import mask_label_encoder
from script.model.sklearn_like_model.BaseModel import BaseEpochCallback, BaseDataCollector
from script.model.sklearn_like_model.SemanticSegmentation import SemanticSegmentation
from script.model.sklearn_like_model.TFSummary import TFSummaryScalar
from script.model.sklearn_like_model.callback.EarlyStop import EarlyStop
from script.model.sklearn_like_model.callback.Top_k_save import Top_k_save
from script.util.misc_util import time_stamp, path_join, to_dict
from script.workbench.TGS_salt.TGS_salt_inference import plot, TGS_salt_DataHelper, \
    TGS_salt_aug_callback, masks_rate, Metrics

SUMMARY_PATH = f'./tf_summary/TGS_salt/SS'
INSTANCE_PATH = f'./instance/TGS_salt/SS'
PLOT_PATH = f'./matplot/TGS_salt/SS'


class CollectDataCallback(BaseDataCollector):
    def __init__(self, test_x, test_y, **kwargs):
        super().__init__(**kwargs)
        self.test_x = test_x
        self.test_y = test_y

    def update_data(self, model, dataset, metric, epoch):
        self.train_loss = metric

        train_x, train_y = dataset.next_batch(500, update_cursor=False)
        self.train_x = train_x
        self.train_y = train_y

        train_predict = model.predict(self.train_x)
        self.train_predict = mask_label_encoder.from_label(train_predict)

        test_predict = model.predict(self.test_x)
        self.test_predict = mask_label_encoder.from_label(test_predict)

        self.train_score = Metrics.TGS_salt_score(self.train_y, self.train_predict)
        self.test_score = Metrics.TGS_salt_score(self.test_y, self.test_predict)

        idx = np.mean(self.test_y.reshape([len(self.test_y), -1]), axis=1) >= 0.01
        self.test_upper_1p_TGS_score = Metrics.TGS_salt_score(self.test_y[idx], self.test_predict[idx])
        self.test_upper_1p_iou_score = Metrics.miou(self.test_y[idx], self.test_predict[idx])

        idx = np.mean(self.train_y.reshape([len(self.train_y), -1]), axis=1) >= 0.01
        self.train_upper_1p_TGS_score = Metrics.TGS_salt_score(self.train_y[idx], self.train_predict[idx])
        self.train_upper_1p_iou_score = Metrics.miou(self.train_y[idx], self.train_predict[idx])

        self.train_iou_score = Metrics.miou(self.train_y, self.train_predict)
        self.test_iou_score = Metrics.miou(self.test_y, self.test_predict)

        self.train_predict_sample = self.train_predict[:20]


class LoggingCallback(BaseEpochCallback):
    def __init__(self, data_collection):
        super().__init__()
        self.data_collection = data_collection
        self.dc = self.data_collection

    def log_TGS_salt_metric(self, epoch):
        msg = f'\n'
        msg += f'e:{epoch}, '

        msg += f'TGS_salt_metric scpre\n'
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


class PlotToolsCallback(BaseEpochCallback):
    def __init__(self, data_collection, **kwargs):
        self.plot = plot
        self.data_collection = data_collection
        self.kwargs = kwargs

    def plot_mask_image(self, model, dataset, metric, epoch):
        run_id = model.run_id

        x, y = dataset.next_batch(20, update_cursor=False)
        x_image = x[:, :, :, 0]
        y = y.reshape([-1, 101, 101, 1]) * 254

        predict = model.predict(x)
        predict = mask_label_encoder.from_label(predict)
        proba = model.predict_proba(x)
        proba = proba[:, :, :, 1].reshape([-1, 101, 101, 1]) * 255
        x_image = x_image.reshape([-1, 101, 101, 1])
        predict = predict.reshape([-1, 101, 101, 1])

        def scramble_column(*args, size=10):
            ret = []
            for i in range(0, len(args[0]), size):
                for j in range(len(args)):
                    ret += [args[j][i:i + size]]

            return np.concatenate(ret, axis=0)

        np_tile = scramble_column(x_image, y, predict, proba)
        self.plot.plot_image_tile(
            np_tile,
            title=f'predict_epoch({epoch})',
            column=10,
            path=path_join(PLOT_PATH, run_id, f'predict_mask/({epoch}).png'))

    def plot_non_mask_rate_iou(self, model, dataset, metric, epoch):
        run_id = model.run_id
        test_y = self.data_collection.test_y
        test_predict = self.data_collection.test_predict

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


class SemanticSegmentation_pipeline:
    def __init__(self):
        self.data_helper = TGS_salt_DataHelper()
        self.plot = plot

    def set_train_set(self, train_set):
        self.train_set = train_set

    def set_dict(self, set_dict):
        self.holdout_set = None
        self.kfold_sets = None
        self.train_set = None

        self.train_x = None
        self.train_y = None
        self.valid_x = None
        self.valid_y = None

        self.test_x = None
        self.test_y = None

        self.holdout_y = None
        self.holdout_y = None

        for key, val in set_dict.items():
            setattr(self, key, val)

        self.train_y_encode = mask_label_encoder.to_label(self.train_y)
        # self.test_y_encode = mask_label_encoder.to_label(self.test_y)
        self.valid_y_encode = mask_label_encoder.to_label(self.valid_y)
        self.holdout_y_encode = mask_label_encoder.to_label(self.holdout_y)

    def params(
            self,
            run_id=None,
            verbose=10,
            learning_rate=0.01,
            learning_rate_decay_rate=0.99,
            learning_rate_decay_method=None,
            beta1=0.9,
            batch_size=100,
            stage=4,
            loss_type='pixel_wise_softmax',
            n_classes=2,
            net_type=None,
            capacity=64,
            depth=2,
            dropout_rate=0.5,
            comment=''
    ):
        # loss_type = 'pixel_wise_softmax'
        # loss_type = 'iou'
        # loss_type = 'dice_soft'
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

    def new_model(self, params):
        model = SemanticSegmentation(**params)
        model.build(Xs=self.train_x, Ys=self.train_y_encode)
        self.model = model

    def del_model(self):
        del self.model

    def load_model(self, path):
        model = SemanticSegmentation().load(path)
        print(model)
        model.build(Xs=self.train_x, Ys=self.train_y_encode)
        model.restore(path)
        self.model = model

        return model

    def train(self, n_epoch=10, augmentation=False):
        pprint(self.model)

        model = self.model
        run_id = model.run_id
        dc_callback = CollectDataCallback(self.valid_x, self.valid_y)
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

            EarlyStop(
                10, min_best=False
            ).trace_on(
                dc_callback, 'test_iou_score'
            ),
            # ReduceLrOnPlateau(0.95, 5, 0.005).trace_on(dc_callback, 'test_iou_score'),
            # TriangleLRScheduler(10, 0.002, 0.0002),
        ]

        batch_size = model.batch_size
        aug_callback = TGS_salt_aug_callback(self.train_x, self.train_y_encode, batch_size) \
            if augmentation else None

        # save_tf_summary_params(SUMMARY_PATH, model.params)
        model.init_adam_momentum()

        model.update_learning_rate(0.002)
        model.update_dropout_rate(0.5)
        pprint(self.model)
        model.train(
            self.train_x, self.train_y_encode, epoch=n_epoch,
            epoch_callbacks=callbacks
        )
