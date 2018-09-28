import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa

from script.data_handler.ImgMaskAug import ActivatorMask
from script.data_handler.TGS_salt import load_sample_image, collect_images, TRAIN_MASK_PATH
from script.util.PlotTools import PlotTools
from script.workbench.TGS_salt.TGS_salt_inference import TGS_salt_metric, TGS_salt_DataHelper

plot = PlotTools(save=True, show=False)


def test_TGS_salt_aug_callback():
    # image + mask
    # from 101*101

    x, y = load_sample_image()
    x = x[:5]
    y = y[:5]
    # x = np.concatenate([x[:1] for i in range(5)])
    # y = np.concatenate([y[:1] for i in range(5)])

    import random
    ia.seed(random.randint(1, 10000))

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # bright add
    # contrast multiply
    seq = iaa.Sequential([
        iaa.OneOf([
            iaa.PiecewiseAffine((0.002, 0.1), name='PiecewiseAffine'),
            iaa.Affine(rotate=(-20, 20)),
            iaa.Affine(shear=(-45, 45)),
            iaa.Affine(translate_percent=(0, 0.3), mode='symmetric'),
            iaa.Affine(translate_percent=(0, 0.3), mode='wrap'),
            iaa.PerspectiveTransform((0.0, 0.3))
        ], name='affine'),
        iaa.Fliplr(0.5, name="horizontal flip"),
        iaa.Crop(percent=(0, 0.3), name='crop'),

        # image only
        iaa.OneOf([
            iaa.Add((-45, 45), name='bright'),
            iaa.Multiply((0.5, 1.5), name='contrast')]
        ),
        iaa.OneOf([
            iaa.AverageBlur((1, 5), name='AverageBlur'),
            # iaa.BilateralBlur(),
            iaa.GaussianBlur((0.1, 2), name='GaussianBlur'),
            iaa.MedianBlur((1, 7), name='MedianBlur'),
        ], name='blur'),

        # scale to  128 * 128
        iaa.Scale((128, 128), name='to 128 * 128'),
    ])
    activator = ActivatorMask(['bright', 'contrast', 'AverageBlur', 'GaussianBlur', 'MedianBlur'])
    hook_func = ia.HooksImages(activator=activator)

    n_iter = 5
    tile = []
    for idx in range(n_iter):
        print(idx)
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images(x)
        mask_aug = seq_det.augment_images(y, hooks=hook_func)
        tile += [image_aug]
        tile += [mask_aug]

    tile = np.concatenate(tile)
    plot.plot_image_tile(tile, title=f'test_image_aug', column=5, )


def test_TGS_salt_metric():
    print(f'collect train mask images')
    train_mask_images, _, _ = collect_images(TRAIN_MASK_PATH)
    train_mask_images = train_mask_images.reshape([-1, 101, 101])
    idx = 10
    size = 10
    images = train_mask_images[idx:idx + size]

    metric_images = []
    for a in images:
        for b in images:
            a = a.reshape([101, 101, 1])
            b = b.reshape([101, 101, 1])
            zero_channel = np.zeros([101, 101, 1])
            rgb_image = np.concatenate([a, b, zero_channel], axis=2)
            metric_images += [rgb_image]
    metric_images = np.array(metric_images)
    plot.plot_image_tile(metric_images, title='test_metric', column=size)

    metric_score = []
    for a in images:
        for b in images:
            metric_score += [TGS_salt_metric(a, b)]
    metric_score = np.array(metric_score).reshape([size, size])
    print(metric_score)


def test_TGS_salt_data_helper():
    helper = TGS_salt_DataHelper()
    non_empty_mask_train_set = helper.train_set_non_empty_mask
    empty_mask_train_set = helper.train_set_empty_mask

    img, mask = non_empty_mask_train_set.next_batch(10)
    tile = np.concatenate((img, mask), axis=0)
    tile = tile.reshape([20, 101, 101, 1])
    plot.plot_image_tile(tile, path='./matplot/non_empty_mask.png')

    img, mask = empty_mask_train_set.next_batch(10)
    tile = np.concatenate((img, mask), axis=0)
    tile = tile.reshape([20, 101, 101, 1])
    plot.plot_image_tile(tile, path='./matplot/empty_mask.png')
