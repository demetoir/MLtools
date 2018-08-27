# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from script.data_handler.TGS_salt import TRAIN_IMAGE_PATH, collect_images, TRAIN_MASK_PATH, RLE_mask_encoding, \
    TEST_IMAGE_PATH
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.util.Logger import pprint_logger, Logger
from script.util.deco import deco_timeit
from script.util.PlotTools import PlotTools

# print(built-in function) is not good for logging
from script.util.numpy_utils import np_img_gray_to_rgb

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame
Series = pd.Series

plot = PlotTools(save=True, show=False)


# plot = PlotTools(save=False, show=True)


def add_feature_mask_pixel_rate(mask, mask_value=255):
    return np.mean(mask) / 255


def add_feature_sliced_mask(mask):
    encoding = RLE_mask_encoding(mask.reshape([101, 101]).transpose())
    maks_rate = add_feature_mask_pixel_rate(mask)
    if 0 < encoding / 2 < 8 and 0.1 < maks_rate < 1:
        return True
    else:
        return False


def is_white_image(image):
    if np.mean(image) == 255:
        return True
    else:
        return False


def is_black_image(image):
    if np.mean(image) == 0:
        return True
    else:
        return False


def np_img_to_img_scatter(images, xy, panel_x=10000, panel_y=10000):
    img_x = images.shape[1]
    img_y = images.shape[2]

    if images.ndim == 3:
        panel = np.zeros([panel_x + img_x, panel_y + img_y], dtype=images.dtype)
    else:
        panel = np.zeros([panel_x, panel_y, 3], dtype=images.dtype)

    xs, ys = xy[:, 0], xy[:, 1]
    for image, x, y in zip(images, xs, ys):
        a = int(x * panel_x)
        b = int(y * panel_y)
        panel[a:a + img_x, b: b + img_y] = image

    return panel


def masking_images(image, mask, mask_rate=.8):
    image = np.array(image)
    if image.ndim != 3:
        raise ValueError('image ndim must 3')

    image[:, :, 0] = mask * mask_rate

    return image


class experiment:
    def tsne_cluster_image(self):
        # not good...
        from sklearn.preprocessing import MinMaxScaler

        print(f'collect train images')
        limit = 500
        images, _, ids = collect_images(TRAIN_IMAGE_PATH, limit=limit)

        print('fit transform')
        tsne = TSNE()
        tsne.fit(images.reshape([-1, 101 * 101]))
        vector = tsne.fit_transform(images.reshape([-1, 101 * 101]))

        scaler = MinMaxScaler()
        scaler.fit(vector)
        vector = scaler.transform(vector)
        print(vector, vector.shape)

        plot.scatter_2d(vector, title='tsne_cluster')

        cluster_image = np_img_to_img_scatter(images, vector, 5000, 5000)
        cluster_image = np_img_gray_to_rgb(cluster_image)
        plot.plot_image(cluster_image, title='tsne_cluster_images')

    def pca_cluster_image(self):
        # not good...
        from sklearn.preprocessing import MinMaxScaler

        limit = 500
        print(f'collect train images')
        images, _, ids = collect_images(TRAIN_IMAGE_PATH, limit=limit)

        pca = PCA(n_components=2)
        pca.fit(images.reshape([-1, 101 * 101]))
        vector = pca.transform(images.reshape([-1, 101 * 101]))

        scaler = MinMaxScaler()
        scaler.fit(vector)
        vector = scaler.transform(vector)
        print(vector, vector.shape)

        x = vector[:, 0]
        y = vector[:, 1]
        plot.scatter_2d(vector, title='pca_cluster')

        # images = np_img_gray_to_rgb(x, y)

        cluster_image = np_img_to_img_scatter(images, vector, 5000, 5000)
        cluster_image = np_img_gray_to_rgb(cluster_image)
        plot.plot_image(cluster_image, title='pca_cluster_images')

        # TODO
        pass

    def plot_train_image_with_mask(self):
        limit = None
        print(f'collect train images')
        train_images, _, _ = collect_images(TRAIN_IMAGE_PATH, limit=limit)

        print(f'collect train mask images')
        train_mask_images, _, _ = collect_images(TRAIN_MASK_PATH, limit=limit)
        train_mask_images = train_mask_images.reshape([-1, 101, 101])

        train_images = np_img_gray_to_rgb(train_images)

        masked_images = []
        for image, mask in zip(train_images, train_mask_images):
            masked_images += [masking_images(image, mask)]
        masked_images = np.array(masked_images)

        plot.plot_image_tile(masked_images[:500], title='masked_0', column=20)
        plot.plot_image_tile(masked_images[500:1000], title='masked_1', column=20)
        plot.plot_image_tile(masked_images[1000:1500], title='masked_2', column=20)
        plot.plot_image_tile(masked_images[1500:2000], title='masked_3', column=20)

    def plot_test_image(self):
        print(f'collect test images')
        test_images, _, _ = collect_images(TEST_IMAGE_PATH)
        test_images = np_img_gray_to_rgb(test_images)

        for i in range(0, len(test_images), 500):
            print(i)
            plot.plot_image_tile(test_images[i: i + 500], title=f'test_{i}', column=20)

    def sort_by_mask_area_size(self):
        limit = None
        print(f'collect train images')
        images, _, ids = collect_images(TRAIN_IMAGE_PATH, limit=limit)
        images = np_img_gray_to_rgb(images)

        print(f'collect train masks images')
        masks, _, _ = collect_images(TRAIN_MASK_PATH, limit=limit)
        masks = masks.reshape([-1, 101, 101])

        masked = []
        mean = []
        for image, mask, id in zip(images, masks, ids):
            a = np.sum(mask) / (101 * 101 * 1 * 255)
            print(id, a)
            if a > 0.6:
                masked += [masking_images(image, mask)]
                mean += [a]

        print(len(masked))
        print(np.mean(mean), np.std(mean))
        pprint(mean)
        masked = np.array(masked)
        plot.plot_image_tile(masked, title='mask_small_area')

    def filter_white_image(self):
        limit = None
        print(f'collect train images')
        images, _, ids = collect_images(TRAIN_IMAGE_PATH, limit=limit)
        images = np_img_gray_to_rgb(images)

        print(f'collect train masks images')
        masks, _, _ = collect_images(TRAIN_MASK_PATH, limit=limit)
        masks = masks.reshape([-1, 101, 101])

        masked = []
        mean = []
        for image, mask, id in zip(images, masks, ids):
            a = np.sum(image) / (101 * 101 * 3 * 255)
            print(id, a)
            if a > 0.85:
                masked += [masking_images(image, mask)]
                mean += [a]

        print(len(masked))
        print(np.mean(mean), np.std(mean))
        pprint(mean)
        masked = np.array(masked)
        plot.plot_image_tile(masked, title='almost_white')

    def filter_black_image(self):
        limit = None
        print(f'collect train images')
        images, _, ids = collect_images(TRAIN_IMAGE_PATH, limit=limit)
        images = np_img_gray_to_rgb(images)

        print(f'collect train masks images')
        masks, _, _ = collect_images(TRAIN_MASK_PATH, limit=limit)
        masks = masks.reshape([-1, 101, 101])

        masked = []
        mean = []
        for image, mask, id in zip(images, masks, ids):
            a = np.sum(image) / (101 * 101 * 3 * 255)
            print(id, a)
            if a < 0.20:
                masked += [masking_images(image, mask)]
                mean += [a]

        print(len(masked))
        print(np.mean(mean), np.std(mean))
        pprint(mean)
        masked = np.array(masked)
        plot.plot_image_tile(masked, title='almost_black')

    def chopped_mask_image(self):
        limit = None
        print(f'collect train images')
        images, _, ids = collect_images(TRAIN_IMAGE_PATH, limit=limit)
        images = np_img_gray_to_rgb(images)

        print(f'collect train masks images')
        masks, _, _ = collect_images(TRAIN_MASK_PATH, limit=limit)
        masks = masks.reshape([-1, 101, 101])

        masked = []
        for image, mask, id in zip(images, masks, ids):
            rle_mask = RLE_mask_encoding(mask.reshape([101, 101]).transpose())
            n_rle_mask = len(rle_mask)
            a = n_rle_mask
            mask_area = np.sum(mask) / (101 * 101 * 255)

            if 0 < a / 2 < 8 and 0.1 < mask_area < 0.99:
                print(id, a, rle_mask)
                masked += [masking_images(image, mask)]

        masked = np.array(masked)
        print(len(masked))
        plot.plot_image_tile(masked, title='chopped')


def empty_mask_clf():
    # TODO
    pass


def image_augmentation():
    # TODO
    # image + mask
    pass


class learning_rate_mixIn:
    pass


class FusionNet(BaseModel):
    # TODO
    pass


@deco_timeit
def main():
    # exp = experiment()
    # exp.tsne_cluster_image()
    # exp.pca_cluster_image()

    # plot_test_image()
    # chopped_mask_image()
    pass
