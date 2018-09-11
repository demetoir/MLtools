import numpy as np
from tqdm import tqdm

from script.data_handler.TGS_salt import mask_label_encoder
from script.model.sklearn_like_model.BaseModel import BaseEpochCallback
from script.model.sklearn_like_model.SemanticSegmentation import SemanticSegmentation
from script.model.sklearn_like_model.TFSummary import TFSummary
from script.model.sklearn_like_model.Top_k_save import Top_k_save
from script.util.misc_util import time_stamp, path_join
from script.workbench.TGS_salt.TGS_salt_inference import plot, TGS_salt_metric, data_helper, to_dict, \
    TGS_salt_aug_callback, save_tf_summary_params

SUMMARY_PATH = f'./tf_summary/TGS_salt/SS'
INSTANCE_PATH = f'./instance/TGS_salt/SS'
PLOT_PATH = f'./matplot/TGS_salt/SS'


class Unet_epoch_callback(BaseEpochCallback):
    def __init__(self, model, test_x, test_y, params):
        super().__init__()
        self.model = model
        self.plot = plot
        self.test_x = test_x
        self.test_y = test_y
        self.params = params

        self.run_id = self.params['run_id']

        self.summary_train_loss = TFSummary(path_join(SUMMARY_PATH, self.run_id, 'train'), 'train_loss')
        self.summary_train_acc = TFSummary(path_join(SUMMARY_PATH, self.run_id, 'train'), 'train_acc')
        self.summary_test_acc = TFSummary(path_join(SUMMARY_PATH, self.run_id, 'test'), 'test_acc')

        self.top_k_save = Top_k_save(path_join(INSTANCE_PATH, self.run_id))

    def log_TGS_salt_metric(self, dataset, epoch):
        x, y = dataset.next_batch(1000)
        x = x.reshape([-1, 101, 101, 1])
        y = y.reshape([-1, 101, 101, 1])
        predict_train = self.model.predict(x)
        predict_train = mask_label_encoder.from_label(predict_train)
        predict_test = self.model.predict(self.test_x)
        predict_test = mask_label_encoder.from_label(predict_test)
        self.train_loss = self.model.metric(x, y)
        self.train_score = TGS_salt_metric(y, predict_train)
        self.test_score = TGS_salt_metric(self.test_y, predict_test)
        tqdm.write(f'e:{epoch}, TGS_salt_metric train = {self.train_score}\n'
                   f'test = {self.test_score}\n')

    def plot_mask(self, dataset, epoch):
        x, y = dataset.next_batch(20)
        x = x.reshape([-1, 101, 101, 1])
        y = y.reshape([-1, 101, 101, 1]) * 254
        predict = self.model.predict(x)
        proba = self.model.predict_proba(x)
        proba = proba[:, :, :, 1].reshape([-1, 101, 101, 1]) * 255
        predict = mask_label_encoder.from_label(predict)

        def f(*args, size=10):
            ret = []
            for i in range(0, len(args[0]), size):
                for j in range(len(args)):
                    ret += [args[j][i:i + size]]

            return np.concatenate(ret, axis=0)

        np_tile = f(x, y, predict, proba)
        self.plot.plot_image_tile(
            np_tile,
            title=f'predict_epoch({epoch})',
            column=10,
            path=path_join(PLOT_PATH, self.run_id, f'predict_epoch({epoch}).png'))

    def update_summary(self, sess, epoch):
        self.summary_train_loss.update(sess, self.train_loss, epoch)
        self.summary_train_acc.update(sess, self.train_score, epoch)
        self.summary_test_acc.update(sess, self.test_score, epoch)

    def __call__(self, sess, dataset, epoch, log=None):
        self.plot_mask(dataset, epoch)
        self.log_TGS_salt_metric(dataset, epoch)
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
               loss_type='pixel_wise_softmax', n_classes=2,
               capacity=64, depth=1):
        # loss_type = 'pixel_wise_softmax'
        # loss_type = 'iou'
        # loss_type = 'dice_soft'
        if run_id is None:
            run_id = time_stamp()

        net_type = 'FusionNet'
        net_type = 'UNet'
        net_type = 'UNet_res_block'

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
            depth=depth
        )
        return params

    def train(self, params, n_epoch=10, augmentation=False, early_stop=True, patience=20, save_path=None):
        save_tf_summary_params(SUMMARY_PATH, params)

        model = SemanticSegmentation(**params)

        epoch_callback = Unet_epoch_callback(model, self.test_x, self.test_y, params)
        dataset_callback = TGS_salt_aug_callback if augmentation else None

        model.train(self.train_x, self.train_y_encode, epoch=n_epoch, aug_callback=dataset_callback,
                    epoch_callback=epoch_callback, early_stop=early_stop, patience=patience,
                    iter_pbar=True)

        # if save_path is None:
        #     save_path = f'./instance/TGS_salt/SS/{params_str}'
        # model.save(save_path)
