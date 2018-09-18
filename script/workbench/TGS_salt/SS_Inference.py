import numpy as np
from tqdm import tqdm

from script.data_handler.TGS_salt import mask_label_encoder
from script.model.sklearn_like_model.BaseModel import BaseEpochCallback
from script.model.sklearn_like_model.SemanticSegmentation import SemanticSegmentation
from script.model.sklearn_like_model.TFSummary import TFSummaryScalar
from script.model.sklearn_like_model.Top_k_save import Top_k_save
from script.util.misc_util import time_stamp, path_join
from script.workbench.TGS_salt.TGS_salt_inference import plot, TGS_salt_metric, data_helper, to_dict, \
    TGS_salt_aug_callback, save_tf_summary_params

SUMMARY_PATH = f'./tf_summary/TGS_salt/SS'
INSTANCE_PATH = f'./instance/TGS_salt/SS'
PLOT_PATH = f'./matplot/TGS_salt/SS'


class Epoch_callback(BaseEpochCallback):
    def __init__(self, model, test_x, test_y, params):
        super().__init__()
        self.model = model
        self.plot = plot
        self.test_x = test_x
        self.test_y = test_y
        self.params = params

        self.run_id = self.params['run_id']

        self.summary_train_loss = TFSummaryScalar(path_join(SUMMARY_PATH, self.run_id, 'train'), 'train_loss')
        self.summary_train_acc = TFSummaryScalar(path_join(SUMMARY_PATH, self.run_id, 'train'), 'train_acc')
        self.summary_test_acc = TFSummaryScalar(path_join(SUMMARY_PATH, self.run_id, 'test'), 'test_acc')

        self.top_k_save = Top_k_save(path_join(INSTANCE_PATH, self.run_id), k=3)

    def log_TGS_salt_metric(self, epoch):
        msg = f'\n'
        msg += f'e:{epoch}, '
        msg += f'TGS_salt_metric train score = {self.train_score}\n'
        msg += f'TGS_salt_metric test score = {self.test_score}\n'
        msg += f'empty mask score\n'
        msg += f'\n'
        msg += f'train empty mask score = {self.train_empty_mask_score}, (total = {self.train_empty_mask_size})\n'
        msg += f'test empty mask score = {self.test_empty_mask_score}, (total = {self.test_empty_mask_size})\n'
        msg += f'\n'
        msg += f'train non empty mask score = {self.train_non_empty_score},' \
               f'(total = {self.train_non_empty_mask_size}\n'
        msg += f'test non empty mask score = {self.test_non_empty_score}' \
               f'(total = {self.test_non_empty_mask_size})\n'
        msg += f'\n'

        tqdm.write(msg)

    def plot_mask_image(self, dataset, epoch):
        x, y = dataset.next_batch(20)
        x = x.reshape([-1, 101, 101, 1])
        y = y.reshape([-1, 101, 101, 1]) * 254
        predict = self.model.predict(x)
        proba = self.model.predict_proba(x)
        proba = proba[:, :, :, 1].reshape([-1, 101, 101, 1]) * 255
        predict = mask_label_encoder.from_label(predict)

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
            path=path_join(PLOT_PATH, self.run_id, f'predict_mask/({epoch}).png'))

    def plot_mask_rate_iou(self, dataset, epoch):
        test_x = self.test_x[:100]
        test_y = self.test_y[:100]
        train_x, train_y = dataset.next_batch(100)

        train_predict = self.model.predict(train_x)

        def iou(true, predict):
            if np.sum(true) == 0:
                if np.sum(predict) > 0:
                    return 0
                else:
                    return 1
            else:
                mask_true = true / 255
                mask_predict = predict / 255

                intersect = np.logical_and(mask_true, mask_predict)
                union = np.logical_or(mask_true, mask_predict)
                iou = np.sum(intersect) / np.sum(union)

                return iou
            pass

        def mask_rate(mask):
            pass

        x = mask_rate(train_y)
        y = iou(train_y, train_predict)
        dots = zip(x, y)
        self.plot.scatter_2d(
            dots,
            title=f'mask rate and iou',
            path=path_join(PLOT_PATH, self.run_id, f'mask_rate_iou/({epoch}).png'))

    def update_summary(self, sess, epoch):
        self.summary_train_loss.update(sess, self.train_loss, epoch)
        self.summary_train_acc.update(sess, self.train_score, epoch)
        self.summary_test_acc.update(sess, self.test_score, epoch)

    def update_data(self, sess, dataset, epoch):
        train_x, train_y = dataset.next_batch(1000)
        self.train_x = train_x.reshape([-1, 101, 101, 1])
        self.train_y = train_y.reshape([-1, 101, 101, 1])

        train_predict = self.model.predict(self.train_x)
        self.train_predict = mask_label_encoder.from_label(train_predict)

        test_predict = self.model.predict(self.test_x)
        self.test_predict = mask_label_encoder.from_label(test_predict)

        self.train_loss = self.model.metric(self.train_x, self.train_y)

        self.train_score = TGS_salt_metric(self.train_y, self.train_predict)
        self.test_score = TGS_salt_metric(self.test_y, self.test_predict)

        def get_idxs(true):
            size = len(true)
            true = true.reshape([size, -1])
            true = np.mean(true, axis=1)

            empty_idxs = true == 0
            non_empty_idxs = true != 0
            return empty_idxs, non_empty_idxs

        train_empty_idxs, train_non_empty_idxs = get_idxs(self.train_y)
        self.train_empty_mask_size = len(train_empty_idxs)
        self.train_non_empty_mask_size = len(train_non_empty_idxs)

        self.train_empty_mask_predict = self.train_predict[train_empty_idxs]
        self.train_non_empty_mask_predict = self.train_predict[train_non_empty_idxs]

        self.train_empty_mask_y = self.train_y[train_empty_idxs]
        self.train_non_empty_mask_y = self.train_y[train_non_empty_idxs]

        self.train_empty_mask_score = TGS_salt_metric(self.train_empty_mask_y, self.train_empty_mask_predict)
        self.train_non_empty_score = TGS_salt_metric(self.train_non_empty_mask_y, self.train_non_empty_mask_predict)

        test_empty_idxs, test_non_empty_idxs = get_idxs(self.test_y)
        self.test_empty_mask_size = len(test_empty_idxs)
        self.test_non_empty_mask_size = len(test_non_empty_idxs)

        self.test_empty_mask_predict = self.test_predict[test_empty_idxs]
        self.test_non_empty_mask_predict = self.test_predict[test_non_empty_idxs]

        self.test_empty_mask_y = self.test_y[test_empty_idxs]
        self.test_non_empty_mask_y = self.test_y[test_non_empty_idxs]

        self.test_empty_mask_score = TGS_salt_metric(self.test_empty_mask_y, self.test_empty_mask_predict)
        self.test_non_empty_score = TGS_salt_metric(self.test_non_empty_mask_y, self.test_non_empty_mask_predict)

    def __call__(self, sess, dataset, epoch, log=None):
        self.update_data(sess, dataset, epoch)

        self.plot_mask_image(dataset, epoch)
        self.log_TGS_salt_metric(epoch)
        self.update_summary(sess, epoch)
        self.top_k_save(self.test_score, self.model)


class SemanticSegmentation_pipeline:
    def __init__(self):
        self.data_helper = data_helper()
        self.plot = plot
        # self.aug_callback = TGS_salt_aug_callback
        # self.epoch_callback = epoch_callback

        self.init_dataset()

    def init_dataset(self):
        train_set = self.data_helper.train_set

        x_full, y_full = train_set.full_batch()
        x_full = x_full.reshape([-1, 101, 101, 1])
        y_full = y_full.reshape([-1, 101, 101, 1])

        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            x_full, y_full, test_size=0.33)

        train_y_encode = mask_label_encoder.to_label(train_y)
        test_y_encode = mask_label_encoder.to_label(test_y)

        self.x_full = x_full
        self.y_full = y_full
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.train_y_encode = train_y_encode
        self.test_y_encode = test_y_encode

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
            net_type='FusionNet',
            loss_type=loss_type,
            capacity=capacity,
            n_classes=n_classes,
            depth=depth,
            dropout_rate=dropout_rate,
            comment=comment
        )
        return params

    def train(self, params, n_epoch=10, augmentation=False, early_stop=True, patience=20, path=None):
        save_tf_summary_params(SUMMARY_PATH, params)

        if path:
            model = SemanticSegmentation().load(path)
        else:
            model = SemanticSegmentation(**params)

        epoch_callback = Epoch_callback(model, self.test_x, self.test_y, params)
        dataset_callback = TGS_salt_aug_callback(self.train_x, self.train_y_encode, params['batch_size']) \
            if augmentation else None

        model.train(self.train_x, self.train_y_encode, epoch=n_epoch, aug_callback=dataset_callback,
                    epoch_callback=epoch_callback, early_stop=early_stop, patience=patience,
                    iter_pbar=True)
