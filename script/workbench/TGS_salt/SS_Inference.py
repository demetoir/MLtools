import numpy
from tqdm import tqdm

from script.data_handler.TGS_salt import mask_label_encoder
from script.model.sklearn_like_model.BaseModel import BaseEpochCallback
from script.model.sklearn_like_model.SemanticSegmentation import SemanticSegmentation
from script.model.sklearn_like_model.TFSummary import TFSummary
from script.model.sklearn_like_model.Top_k_save import Top_k_save
from script.workbench.TGS_salt.TGS_salt_inference import plot, TGS_salt_metric, data_helper, to_dict, param_to_string, \
    TGS_salt_aug_callback


class Unet_epoch_callback(BaseEpochCallback):
    def __init__(self, model, test_x, test_y, params):
        super().__init__()
        self.model = model
        self.plot = plot
        self.test_x = test_x
        self.test_y = test_y
        self.params = params
        params_str = "_".join([f"{key}={val}" for key, val in self.params.items()])
        self.summary_train_loss = TFSummary(f'./tf_summary/TGS_salt/SS/{params_str}/train', 'train_loss')
        self.summary_train_acc = TFSummary(f'./tf_summary/TGS_salt/SS/{params_str}/train', 'train_acc')
        self.summary_test_acc = TFSummary(f'./tf_summary/TGS_salt/SS/{params_str}/test', 'test_acc')
        self.top_k_save = Top_k_save(f'instance/TGS_salt/SS/{params_str}')

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

        tile = f(x, y, predict, proba)
        self.plot.plot_image_tile(tile, title=f'predict_epoch({epoch})', column=10,
                                  path=f'./matplot/{self.model.id}/predict_epoch({epoch}).png')

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

    def param_FusionNet(self, run, verbose=10, learning_rate=0.01, learning_rate_decay_rate=0.99,
                        learning_rate_decay_method=None, beta1=0.9, batch_size=100, stage=4,
                        loss_type='pixel_wise_softmax', n_classes=2,
                        capacity=64, depth=1):
        # loss_type = 'pixel_wise_softmax'
        # loss_type = 'iou'
        # loss_type = 'dice_soft'
        params = to_dict(
            run=run,
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

    def param_UNet(self, run, verbose=10, learning_rate=0.01, learning_rate_decay_rate=0.99,
                   learning_rate_decay_method=None, beta1=0.9, batch_size=100, stage=4,
                   loss_type='pixel_wise_softmax', n_classes=2,
                   capacity=64, depth=1):
        # loss_type = 'pixel_wise_softmax'
        # loss_type = 'iou'
        # loss_type = 'dice_soft'
        params = to_dict(
            run=run,
            verbose=verbose,
            learning_rate=learning_rate,
            beta1=beta1,
            batch_size=batch_size,
            stage=stage,
            net_type='UNet',
            loss_type=loss_type,
            capacity=capacity,
            n_classes=n_classes,
            depth=depth
        )
        return params

    def param_Unet_res_block(self):
        # loss_type = 'pixel_wise_softmax'
        loss_type = 'iou'
        # loss_type = 'dice_soft'
        channel = 16
        level = 4
        learning_rate = 0.002
        batch_size = 256
        net_type = 'FusionNet'
        n_conv_blocks = 10
        params = to_dict(
            run=1,
            n_conv_blocks=n_conv_blocks,
            loss_type=loss_type,
            net_capacity=channel,
            level=level,
            learning_rate=learning_rate,
            batch_size=batch_size,
            net_type=net_type)
        return params

    def make_model(self, params):
        pass

    def _train(self):
        pass

    def train(self, params, n_epoch=10, augmentation=False, early_stop=True, patience=20, save_path=None):
        params_str = param_to_string(params)
        model = SemanticSegmentation(**params)

        epoch_callback = Unet_epoch_callback(model, self.test_x, self.test_y, params)
        dataset_callback = TGS_salt_aug_callback if augmentation else None

        model.train(self.train_x, self.train_y_encode, epoch=n_epoch, aug_callback=dataset_callback,
                    epoch_callback=epoch_callback, early_stop=early_stop, patience=patience,
                    iter_pbar=True)

        if save_path is None:
            save_path = f'./instance/TGS_salt/SS/{params_str}'
        model.save(save_path)
