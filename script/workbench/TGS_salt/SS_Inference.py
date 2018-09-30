import numpy as np
from tqdm import tqdm

from script.data_handler.TGS_salt import mask_label_encoder
from script.model.sklearn_like_model.BaseModel import BaseEpochCallback
from script.model.sklearn_like_model.SemanticSegmentation import SemanticSegmentation
from script.model.sklearn_like_model.TFSummary import TFSummaryScalar
from script.model.sklearn_like_model.TransferLearning import TransferLearning
from script.model.sklearn_like_model.callback.EarlyStop import EarlyStop
from script.model.sklearn_like_model.callback.Top_k_save import Top_k_save
from script.model.sklearn_like_model.callback.TriangleLRScheduler import TriangleLRScheduler
from script.util.misc_util import time_stamp, path_join
from script.workbench.TGS_salt.TGS_salt_inference import plot, TGS_salt_metric, TGS_salt_DataHelper, to_dict, \
    TGS_salt_aug_callback, save_tf_summary_params, iou_metric, masks_rate
from script.workbench.TGS_salt.post_process_AE import post_process_AE
from script.workbench.TGS_salt.pretrain_Unet import pre_train_Unet

SUMMARY_PATH = f'./tf_summary/TGS_salt/SS'
INSTANCE_PATH = f'./instance/TGS_salt/SS'
PLOT_PATH = f'./matplot/TGS_salt/SS'


class BaseDataCollector(BaseEpochCallback):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def update_data(self, model, dataset, metric, epoch):
        raise NotImplementedError

    def __call__(self, model, dataset, metric, epoch):
        self.update_data(model, dataset, metric, epoch)


class CollectDataCallback(BaseDataCollector):
    def __init__(self, test_x, test_y, **kwargs):
        super().__init__(**kwargs)
        self.test_x = test_x
        self.test_y = test_y

    def update_data(self, model, dataset, metric, epoch):
        self.train_loss = metric

        train_x, train_y = dataset.next_batch(1000, update_cursor=False)
        self.train_x = train_x
        self.train_y = train_y
        # self.train_x = train_x.reshape([-1, 101, 101, 1])
        # self.train_y = train_y.reshape([-1, 101, 101, 1])

        train_predict = model.predict(self.train_x)
        self.train_predict = mask_label_encoder.from_label(train_predict)

        test_predict = model.predict(self.test_x)
        self.test_predict = mask_label_encoder.from_label(test_predict)

        self.train_score = TGS_salt_metric(self.train_y, self.train_predict)
        self.test_score = TGS_salt_metric(self.test_y, self.test_predict)

        def get_idxs(true):
            size = len(true)
            true = true.reshape([size, -1])
            true = np.mean(true, axis=1)

            empty_idxs = true == 0
            non_empty_idxs = true != 0
            return empty_idxs, non_empty_idxs

        # split train to empty mask and non empty mask
        train_empty_idxs, train_non_empty_idxs = get_idxs(self.train_y)

        self.train_empty_mask_predict = self.train_predict[train_empty_idxs]
        self.train_non_empty_mask_predict = self.train_predict[train_non_empty_idxs]

        self.train_empty_mask_y = self.train_y[train_empty_idxs]
        self.train_non_empty_mask_y = self.train_y[train_non_empty_idxs]

        self.train_empty_mask_score = TGS_salt_metric(self.train_empty_mask_y, self.train_empty_mask_predict)
        self.train_non_empty_score = TGS_salt_metric(self.train_non_empty_mask_y, self.train_non_empty_mask_predict)

        self.train_empty_mask_size = len(self.train_empty_mask_predict)
        self.train_non_empty_mask_size = len(self.train_non_empty_mask_predict)

        # split test to empty mask and non empty mask
        test_empty_idxs, test_non_empty_idxs = get_idxs(self.test_y)

        self.test_empty_mask_predict = self.test_predict[test_empty_idxs]
        self.test_non_empty_mask_predict = self.test_predict[test_non_empty_idxs]

        self.test_empty_mask_y = self.test_y[test_empty_idxs]
        self.test_non_empty_mask_y = self.test_y[test_non_empty_idxs]

        self.test_empty_mask_score = TGS_salt_metric(self.test_empty_mask_y, self.test_empty_mask_predict)
        self.test_non_empty_score = TGS_salt_metric(self.test_non_empty_mask_y, self.test_non_empty_mask_predict)

        self.test_empty_mask_size = len(self.test_empty_mask_predict)
        self.test_non_empty_mask_size = len(self.test_non_empty_mask_predict)

        iou = []
        for gt, predict in zip(self.train_y, self.train_predict):
            iou += [iou_metric(gt, predict)]
        self.train_iou_score = np.mean(iou)

        iou = []
        for gt, predict in zip(self.test_y, self.test_predict):
            iou += [iou_metric(gt, predict)]
        self.test_iou_score = np.mean(iou)

        self.train_predict_sample = self.train_predict[:20]


class LoggingCallback(BaseEpochCallback):
    def __init__(self, data_collection):
        super().__init__()
        self.data_collection = data_collection
        self.dc = self.data_collection

    def log_TGS_salt_metric(self, epoch):
        msg = f'\n'
        msg += f'e:{epoch}, '
        msg += f'TGS_salt_metric train score   = {self.dc.train_score}\n'
        msg += f'TGS_salt_metric test score    = {self.dc.test_score}\n'

        msg += f'empty mask score\n'
        msg += f'train empty mask score        = {self.dc.train_empty_mask_score},' \
               f' (total = {self.dc.train_empty_mask_size})\n'
        msg += f'test empty mask score         = {self.dc.test_empty_mask_score},' \
               f' (total = {self.dc.test_empty_mask_size})\n'
        msg += f'\n'

        msg += f'non empty mask score\n'
        msg += f'train non empty mask score    = {self.dc.train_non_empty_score}, ' \
               f'(total = {self.dc.train_non_empty_mask_size})\n'
        msg += f'test non empty mask score     = {self.dc.test_non_empty_score}, ' \
               f'(total = {self.dc.test_non_empty_mask_size})\n'
        msg += f'\n'

        msg += f'iou score\n'
        msg += f'train iou score                = {self.dc.train_iou_score}, ' \
               f'(total = {self.dc.train_non_empty_mask_size}\n'
        msg += f'test iou score                 = {self.dc.test_iou_score}, ' \
               f'(total = {self.dc.test_non_empty_mask_size})\n'
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
        self.summary_non_empty_train_score = TFSummaryScalar(train_path, 'non_empty_train_score')
        self.summary_non_empty_test_score = TFSummaryScalar(test_path, 'non_empty_test_score')
        self.summary_iou_train_score = TFSummaryScalar(train_path, 'train_iou')
        self.summary_iou_test_score = TFSummaryScalar(test_path, 'test_iou')

    def __call__(self, model, dataset, metric, epoch):
        sess = model.sess
        train_loss = self.data_collection.train_loss
        train_score = self.data_collection.train_score
        test_score = self.data_collection.test_score

        test_non_empty_score = self.data_collection.test_non_empty_score
        train_non_empty_score = self.data_collection.train_non_empty_score

        train_iou_score = self.data_collection.train_iou_score
        test_iou_score = self.data_collection.test_iou_score

        self.summary_train_loss.update(sess, train_loss, epoch)
        self.summary_train_acc.update(sess, train_score, epoch)
        self.summary_test_acc.update(sess, test_score, epoch)
        self.summary_non_empty_test_score.update(sess, test_non_empty_score, epoch)
        self.summary_non_empty_train_score.update(sess, train_non_empty_score, epoch)
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

        ys = np.array([iou_metric(true, predict) for true, predict in zip(test_y, test_predict)])

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


class Top_k_saveCallback(BaseEpochCallback):
    def __init__(self, data_collection, run_id):
        self.run_id = run_id
        self.data_collection = data_collection
        self.test_score_top_k_save = Top_k_save(
            path_join(INSTANCE_PATH, self.run_id, 'test_score'),
            k=1,
            name='test_score',
            save_model=True
        )
        self.non_empty_mask_test_score_top_k_save = Top_k_save(
            path_join(INSTANCE_PATH, self.run_id, 'non_empty_test_score'),
            k=1,
            name='non_empty_mask_test_score',
            save_model=False
        )

    def __call__(self, model, dataset, metric, epoch):
        test_score = self.data_collection.test_score
        test_non_empty_score = self.data_collection.test_non_empty_score

        self.test_score_top_k_save(model, dataset, test_score, epoch)
        self.non_empty_mask_test_score_top_k_save(
            model, dataset, test_non_empty_score, epoch
        )


class SemanticSegmentation_pipeline:
    def __init__(self):
        self.data_helper = TGS_salt_DataHelper()
        self.plot = plot
        # self.aug_callback = TGS_salt_aug_callback
        # self.epoch_callback = epoch_callback

        self.init_dataset()

    def init_dataset(self):
        # train_set = self.data_helper.train_set

        # train_set = self.data_helper.train_set_non_empty_mask
        # x_full, y_full = train_set.full_batch()
        # x_full = x_full.reshape([-1, 101, 101, 1])
        # y_full = y_full.reshape([-1, 101, 101, 1])

        train_set = self.data_helper.train_set_non_empty_mask_with_depth_image
        train_set.x_keys = ['x_with_depth']
        x_full, y_full = train_set.full_batch()
        print(x_full.shape)

        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            x_full, y_full, test_size=0.30, random_state=12345)

        train_y_encode = mask_label_encoder.to_label(train_y)
        test_y_encode = mask_label_encoder.to_label(test_y)

        self.x_full = x_full
        self.y_full = y_full
        self.train_x = train_x
        self.valid_x = test_x
        self.train_y = train_y
        self.valid_y = test_y
        self.train_y_encode = train_y_encode
        self.valid_y_encode = test_y_encode

        test_set = self.data_helper.test_set
        self.train_x_AE = test_set.full_batch(batch_keys=['image'])['image']
        size = 18000
        # size = 1000
        self.train_x_AE = self.train_x_AE[:size]
        # self.train_x_AE = self.train_x_AE.reshape([size, 101, 101, 1])

        # print(self.train_x_AE.shape)

    def params(self, run_id=None, verbose=10, learning_rate=0.01, learning_rate_decay_rate=0.99,
               learning_rate_decay_method=None, beta1=0.9, batch_size=100, stage=4,
               loss_type='pixel_wise_softmax', n_classes=2, net_type=None,
               capacity=64, depth=2, dropout_rate=0.5,
               comment=''):
        # loss_type = 'pixel_wise_softmax'
        # loss_type = 'iou'
        # loss_type = 'dice_soft'
        if run_id is None:
            run_id = time_stamp()

        # net_type = 'FusionNet'
        # net_type = 'UNet'
        # net_type = 'UNet_res_block'

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

    def train(self, params, n_epoch=10, augmentation=False, path=None, callbacks=None):
        if path:
            model = SemanticSegmentation().load(path)
            model.build(Xs=self.train_x, Ys=self.train_y_encode)
            model.restore(path)
        else:
            model = SemanticSegmentation(**params)
            model.build(Xs=self.train_x, Ys=self.train_y_encode)

        run_id = model.run_id
        if callbacks is None:
            dc_callback = CollectDataCallback(self.valid_x, self.valid_y)
            callbacks = [
                dc_callback,
                LoggingCallback(dc_callback),
                TFSummaryCallback(dc_callback, run_id),
                PlotToolsCallback(dc_callback),
                Top_k_saveCallback(dc_callback, run_id),
                # ReduceLrOnPlateau(0.9, 5, 0.0005),
                # TriangleLRScheduler(7, 0.001, 0.0005),
                EarlyStop(21),
            ]

        aug_callback = TGS_salt_aug_callback(self.train_x, self.train_y_encode, params['batch_size']) \
            if augmentation else None

        if params:
            save_tf_summary_params(SUMMARY_PATH, params)

        for i in range(8):
            model.update_learning_rate(0.002)
            model.train(
                self.train_x, self.train_y_encode, epoch=n_epoch,
                epoch_callbacks=callbacks
            )

    def pre_train(self, params, n_epoch, path=None):
        path = './instance/pretrain_AE_Unet'
        if path:
            pretrain_model = pre_train_Unet().load(path)
            pretrain_model.build(x=self.train_x_AE)
            pretrain_model.restore(path)
        else:
            pretrain_model = pre_train_Unet(**params)
            pretrain_model.build(x=self.train_x_AE)

        # epoch_callback = Epoch_callback(pretrain_model, self.valid_x, self.valid_y, params, self.train_x_AE)
        # dataset_callback = TGS_salt_aug_callback(self.train_x, self.train_y_encode, params['batch_size']) \
        #     if augmentation else None

        # pretrain_model.train_unsupervised(self.train_x_AE, epoch=50)
        path = './instance/pretrain_AE_Unet'
        # pretrain_model.save(path)

        model = SemanticSegmentation(**params)
        model.build(Xs=self.train_x, Ys=self.train_y_encode)
        scope = 'main_graph/FusionNetModule'

        model = TransferLearning(pretrain_model, scope).to(model, scope)

        epoch_callbacks = [
            # Logging_callback(model, self.valid_x, self.valid_y, params)
        ]
        model.train(self.train_x, self.train_y_encode, epoch=100, epoch_callbacks=epoch_callbacks)
        path = './instance/transfered_model'
        model.save(path)
        del model

    def load_baseline(self, path=None):
        path = './instance/TGS_salt/SS/baseline'
        # path = ".\\instance\\TGS_salt\\SS\\2018-09-27_14-18-36\\test_score\\top_1"
        model = None
        if path:
            model = SemanticSegmentation().load(path)
            model.build(Xs=self.train_x, Ys=self.train_y_encode)
            model.restore(path)

        return model

    def train_post_processing(self):
        ae = post_process_AE()

        predict = None
        x = predict
        y = None

        ae.build(x=x, y=y)
        ae.train(x, y, epoch=100)
