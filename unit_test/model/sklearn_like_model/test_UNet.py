from pprint import pprint

import numpy as np

from script.data_handler.TGS_salt import HEAD_PATH, collect_images
from script.model.sklearn_like_model.UNet import UNet
from script.util.PlotTools import PlotTools
from script.util.misc_util import path_join

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
    plot.plot_image_tile(np.concatenate([x, predict, y_gt], axis=0), title='predict', column=sample_size)
