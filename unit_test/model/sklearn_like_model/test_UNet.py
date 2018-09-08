import numpy as np
from pprint import pprint
from script.data_handler.ImgMaskAug import ActivatorMask, ImgMaskAug
from script.data_handler.TGS_salt import HEAD_PATH, collect_images
from script.model.sklearn_like_model.BaseModel import BaseDatasetCallback
from script.model.sklearn_like_model.UNet import UNet
from script.util.PlotTools import PlotTools
from script.util.misc_util import path_join
from imgaug import augmenters as iaa

plot = PlotTools(save=True, show=False)


class mask_label_encoder:
    @staticmethod
    def to_label(x):
        return np.array(x / 255, dtype=int)

    @staticmethod
    def from_label(x):
        return np.array(x * 255, dtype=float)


def test_Unet_toy_set():
    x = np.zeros([100, 128, 128, 1])
    y = np.ones([100, 128, 128, 1])
    y_gt = y

    y_encode = mask_label_encoder.to_label(y)
    print(x.shape)
    print(y_encode.shape)

    Unet = UNet(stage=4, batch_size=10)
    Unet.train(x, y_encode, epoch=100)

    score = Unet.score(x, y_encode)
    pprint(score)

    predict = Unet.predict(x)
    pprint(predict[0])
    pprint(predict.shape)

    proba = Unet.predict_proba(x)
    pprint(proba[0])
    pprint(proba.shape)

    metric = Unet.metric(x, y_encode)
    print(metric)

    predict = mask_label_encoder.from_label(predict)
    plot.plot_image_tile(np.concatenate([x, predict, y_gt], axis=0), title='predict', column=10)


def test_UNet():
    sample_IMAGE_PATH = path_join(HEAD_PATH, 'sample/images')
    sample_MASK_PATH = path_join(HEAD_PATH, 'sample/masks')

    sample_size = 7
    limit = None
    print(f'collect sample images')
    train_images, _, _ = collect_images(sample_IMAGE_PATH, limit=limit)
    train_images = train_images.reshape([-1, 101, 101])
    print(f'collect sample images')
    train_mask_images, _, _ = collect_images(sample_MASK_PATH, limit=limit)
    train_mask_images = train_mask_images.reshape([-1, 101, 101])
    x = train_images
    y = train_mask_images

    import cv2

    x = np.array([cv2.resize(a, (128, 128)) for a in x]).reshape([-1, 128, 128, 1])

    y = np.array([cv2.resize(a, (128, 128)) for a in y]).reshape([-1, 128, 128, 1])
    y_gt = y

    y_encode = mask_label_encoder.to_label(y)
    print(x.shape)
    print(y_encode.shape)

    Unet = UNet(stage=4, batch_size=7)
    Unet.train(x, y_encode, epoch=100, early_stop=True, patience=10)

    score = Unet.score(x, y_encode)
    pprint(score)

    predict = Unet.predict(x)
    pprint(predict[0])
    pprint(predict.shape)

    proba = Unet.predict_proba(x)
    pprint(proba[0])
    pprint(proba.shape)

    metric = Unet.metric(x, y_encode)
    print(metric)

    predict = mask_label_encoder.from_label(predict)
    plot.plot_image_tile(np.concatenate([x, predict, y_gt], axis=0), title='predict', column=sample_size)


def train_early_stop(self, x, y, n_epoch=200, patience=20, min_best=True):
    if min_best is False:
        raise NotImplementedError

    last_metric = np.Inf
    patience_count = 0
    for e in range(1, n_epoch + 1):
        self.train(x, y, epoch=1)
        metric = self.metric(x, y)
        print(f'e = {e}, metric = {metric}, best = {last_metric}')

        if last_metric > metric:
            print(f'improve {last_metric - metric}')
            last_metric = metric
            patience_count = 0
        else:
            patience_count += 1

        if patience_count == patience:
            print(f'early stop')
            break


def test_train_early_stop():
    class dummy_model:
        def train(self, x, y, epoch=4):
            pass

        def metric(self, x, y):
            import random
            return random.uniform(0, 10)

    x = None
    y = None
    model = dummy_model()
    train_early_stop(model, x, y, n_epoch=100)


class dataset_callback(BaseDatasetCallback):
    def __init__(self, x, y, batch_size):
        super().__init__(x, y, batch_size)

        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5, name="Flipper"),

        ])
        self.activator = ActivatorMask([])
        self.aug = ImgMaskAug(self.x, self.y, self.seq, self.activator, self.batch_size, n_jobs=1)

    @property
    def size(self):
        return len(self.x)

    def shuffle(self):
        pass

    def next_batch(self, batch_size, batch_keys=None, update_cursor=True, balanced_class=False, out_type='concat'):
        x, y = self.aug.get_batch()
        return x, y


def test_train_dataset_callback():
    sample_IMAGE_PATH = path_join(HEAD_PATH, 'sample/images')
    sample_MASK_PATH = path_join(HEAD_PATH, 'sample/masks')

    sample_size = 7
    limit = None
    print(f'collect sample images')
    train_images, _, _ = collect_images(sample_IMAGE_PATH, limit=limit)
    train_images = train_images.reshape([-1, 101, 101])
    print(f'collect sample images')
    train_mask_images, _, _ = collect_images(sample_MASK_PATH, limit=limit)
    train_mask_images = train_mask_images.reshape([-1, 101, 101])
    x = train_images
    y = train_mask_images

    import cv2

    x = np.array([cv2.resize(a, (128, 128)) for a in x]).reshape([-1, 128, 128, 1])
    y = np.array([cv2.resize(a, (128, 128)) for a in y]).reshape([-1, 128, 128, 1])
    y_gt = y

    y_encode = mask_label_encoder.to_label(y)
    print(x.shape)
    print(y_encode.shape)

    Unet = UNet(stage=4, batch_size=7)
    # Unet.train(x, y_encode, epoch=100)
    Unet.train(x, y_encode, epoch=1000, dataset_callback=dataset_callback)
    Unet.train(x, y_encode, epoch=1000, dataset_callback=dataset_callback)

    score = Unet.score(x, y_encode)
    pprint(score)

    predict = Unet.predict(x)
    pprint(predict[0])
    pprint(predict.shape)

    proba = Unet.predict_proba(x)
    pprint(proba[0])
    pprint(proba.shape)

    metric = Unet.metric(x, y_encode)
    print(metric)

    predict = mask_label_encoder.from_label(predict)
    plot.plot_image_tile(np.concatenate([x, predict, y_gt], axis=0), title='predict', column=sample_size)
