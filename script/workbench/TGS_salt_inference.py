from pprint import pprint

import numpy as np
from imgaug import augmenters as iaa
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from script.data_handler.ImgMaskAug import ActivatorMask, ImgMaskAug
from script.data_handler.TGS_salt import collect_images, TRAIN_MASK_PATH, TGS_salt, \
    mask_label_encoder, TRAIN_IMAGE_PATH, TEST_IMAGE_PATH, RLE_mask_encoding
from script.model.sklearn_like_model.BaseModel import BaseDatasetCallback, BaseEpochCallback
from script.model.sklearn_like_model.ImageClf import ImageClf
from script.model.sklearn_like_model.Top_k_save import Top_k_save
from script.model.sklearn_like_model.UNet import UNet
from script.util.PlotTools import PlotTools
from script.util.numpy_utils import np_img_to_img_scatter, np_img_gray_to_rgb

plot = PlotTools(save=True, show=False)


def TGS_salt_metric(mask_true, mask_predict):
    # TODO
    def _metric(mask_true, mask_predict):
        if np.sum(mask_true) == 0 and np.sum(mask_predict) > 0:
            # return 0
            return 1
        elif np.sum(mask_true) == 0 and np.sum(mask_predict) == 0:
            return 1
        else:
            threshold = np.arange(0.5, 1, 0.05)

            mask_true = mask_true / 255
            mask_predict = mask_predict / 255

            upper = np.logical_and(mask_true, mask_predict)
            lower = np.logical_or(mask_true, mask_predict)
            iou_score = np.sum(upper) / np.sum(lower)
            # print(iou_score)

            # print(threshold <= iou_score)
            score = np.sum(threshold <= iou_score) / 10.0
            return score

    if mask_true.shape != mask_predict.shape:
        raise ValueError(f'mask shape does not match, true={mask_true.shape}, predict={mask_predict}')

    if mask_true.ndim in (3, 4):
        ret = np.mean([_metric(m_true, m_predict) for m_true, m_predict in zip(mask_true, mask_predict)])
    else:
        ret = _metric(mask_true, mask_predict)

    return ret


class TGS_salt_aug_callback(BaseDatasetCallback):
    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def __init__(self, x, y, batch_size):
        super().__init__(x, y, batch_size)

        self.seq = iaa.Sequential([
            iaa.OneOf([
                iaa.PiecewiseAffine((0.002, 0.1), name='PiecewiseAffine'),
                iaa.Affine(rotate=(-20, 20)),
                iaa.Affine(shear=(-45, 45)),
                iaa.Affine(translate_percent=(0, 0.3), mode='symmetric'),
                iaa.Affine(translate_percent=(0, 0.3), mode='wrap'),
                # iaa.PerspectiveTransform((0.0, 0.3))
            ], name='affine'),
            iaa.Fliplr(0.5, name="horizontal flip"),
            # iaa.Crop(percent=(0, 0.3), name='crop'),

            # image only
            iaa.OneOf([
                iaa.Add((-45, 45), name='bright'),
                iaa.Multiply((0.5, 1.5), name='contrast')]
            ),
            iaa.OneOf([
                iaa.AverageBlur((1, 5), name='AverageBlur'),
                # iaa.BilateralBlur(),
                iaa.GaussianBlur((0.1, 2), name='GaussianBlur'),
                # iaa.MedianBlur((1, 7), name='MedianBlur'),
            ], name='blur'),

            # scale to  128 * 128
            # iaa.Scale((128, 128), name='to 128 * 128'),
        ])
        self.activator = ActivatorMask(['bright', 'contrast', 'AverageBlur', 'GaussianBlur', 'MedianBlur'])
        self.aug = ImgMaskAug(self.x, self.y, self.seq, self.activator, self.batch_size, n_jobs=4, q_size=4000)

    @property
    def size(self):
        return len(self.x)

    def shuffle(self):
        pass

    def next_batch(self, batch_size, batch_keys=None, update_cursor=True, balanced_class=False, out_type='concat'):
        x, y = self.aug.get_batch()

        # try:
        #     plot.plot_image_tile(np.concatenate([x, y]), title='aug')
        # except BaseException:
        #     pass
        return x[:batch_size], y[:batch_size]


class data_helper:
    def __init__(self, data_pack_path='./data/TGS_salt', sample_offset=10, sample_size=10):
        self.data_pack_path = data_pack_path
        self.sample_offset = sample_offset
        self.sample_size = sample_size

        self._data_pack = None
        self._train_set = None
        self._test_set = None
        self._sample_xs = None
        self._sample_ys = None

    @property
    def data_pack(self):
        if self._data_pack is None:
            self._data_pack = TGS_salt()
            self._data_pack.load(self.data_pack_path)

        return self._data_pack

    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = self.data_pack['train']

        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set = self.data_pack['test']

        return self._test_set

    @property
    def sample_xs(self):
        if self._sample_xs is None:
            x_full, _ = self.train_set.full_batch()
            sample_x = x_full[self.sample_offset:self.sample_offset + self.sample_size]
            self._sample_xs = sample_x

        return self._sample_xs

    @property
    def sample_ys(self):
        if self._sample_ys is None:
            _, ys_full = self.train_set.full_batch()
            self._sample_ys = ys_full[self.sample_offset:self.sample_offset + self.sample_size]

        return self._sample_ys

    @property
    def valid_set(self):
        # TODO
        return None


class Unet_pipeline:
    def __init__(self):
        self.data_helper = data_helper()
        self.plot = plot
        # self.aug_callback = TGS_salt_aug_callback
        # self.epoch_callback = epoch_callback

        self.aug_callback = None

        self.epoch_callback = None
        self.model = None

    def train(self, n_epoch=10, augmentation=False, early_stop=True, patience=20):
        train_set = self.data_helper.train_set

        x_full, y_full = train_set.full_batch()
        x_full = x_full.reshape([-1, 101, 101, 1])
        y_full = y_full.reshape([-1, 101, 101, 1])

        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            x_full, y_full, test_size=0.33)

        train_y_encode = mask_label_encoder.to_label(train_y)
        test_y_encode = mask_label_encoder.to_label(test_y)

        # loss_type = 'pixel_wise_softmax'
        loss_type = 'iou'
        # loss_type = 'dice_soft'
        channel = 32
        level = 4
        learning_rate = 0.01
        batch_size = 128
        self.model = UNet(stage=4, batch_size=batch_size,
                          Unet_level=level, Unet_n_channel=channel, loss_type=loss_type,
                          learning_rate=learning_rate)

        class callback(BaseEpochCallback):
            def __init__(self, model, plot, train_set):
                super().__init__()
                self.model = model
                self.plot = plot
                self.train_set = train_set

            def print_TGS_salt_metric(self):
                x, y = train_set.next_batch(100)
                x = x.reshape([-1, 101, 101, 1])
                y = y.reshape([-1, 101, 101, 1])
                predict = self.model.predict(x)
                predict = mask_label_encoder.from_label(predict)

                tqdm.write(f'TGS_salt_metric {TGS_salt_metric(y, predict)}')

            def plot_mask(self, epoch):
                x, y = train_set.next_batch(20)
                x = x.reshape([-1, 101, 101, 1])
                y = y.reshape([-1, 101, 101, 1])
                predict = self.model.predict(x)
                proba = self.model.predict_proba(x)
                proba = proba[:, :, :, 1].reshape([-1, 101, 101, 1]) * 255
                predict = mask_label_encoder.from_label(predict)

                tile = np.concatenate([x, predict, y], axis=0)
                self.plot.plot_image_tile(tile, title=f'predict_epoch({epoch})', column=10,
                                          path=f'./matplot/{self.model.id}/predict/predict_epoch({epoch}).png')

                tile = np.concatenate([x, proba, y], axis=0)
                self.plot.plot_image_tile(tile, title=f'predict_epoch({epoch})', column=10,
                                          path=f'./matplot/{self.model.id}/proba/proba_epoch({epoch}).png')

            def __call__(self, epoch, log=None):
                self.plot_mask(epoch)
                self.print_TGS_salt_metric()

        epoch_callback = callback
        dataset_callback = TGS_salt_aug_callback if augmentation else None
        self.model.train(train_x, train_y_encode, epoch=n_epoch, aug_callback=dataset_callback,
                         epoch_callback=epoch_callback, early_stop=early_stop, patience=patience,
                         iter_pbar=True)

        self.model.save()


class cnn_pipeline:
    def __init__(self):
        self.data_helper = data_helper()
        self.plot = plot
        self.data_helper.train_set.y_keys = ['empty_mask']

    def train(self, n_epoch, augmentation=False, early_stop=True, patience=20):
        train_set = self.data_helper.train_set
        train_x, train_y = train_set.full_batch()
        train_x = train_x.reshape([-1, 101, 101, 1])
        train_y = train_y.reshape([-1, 1])
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        enc.fit(train_y)

        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            train_x, train_y, test_size=0.33)

        train_y_onehot = enc.transform(train_y).toarray()
        test_y_onehot = enc.transform(test_y).toarray()

        print(np.mean(train_y))

        sample_size = 100
        sample_x = train_x[:sample_size]
        sample_y = train_y[:sample_size]
        sample_y_onehot = train_y_onehot[:sample_size]

        clf = ImageClf(net_type='InceptionV1')

        class Callback(BaseEpochCallback):
            def __init__(self, top_k_path):
                super().__init__()

                self.k = 5
                self.top_k = [np.Inf for _ in range(self.k)]
                self.top_k_save = Top_k_save(top_k_path)

            def __call__(self, epoch, log=None):
                from sklearn.metrics import confusion_matrix
                from sklearn.metrics import accuracy_score

                test_predict = clf.predict(test_x)
                print(test_predict.shape)
                print(test_y_onehot.shape)
                test_score = accuracy_score(test_y, test_predict)
                test_confusion = confusion_matrix(test_y, test_predict)

                sample_predict = clf.predict(sample_x)
                sample_score = accuracy_score(sample_y, sample_predict)
                sample_confusion = confusion_matrix(sample_y, sample_predict)
                log(f'e={epoch}, '
                    f'sample_score = {sample_score}, \n'
                    f'sample_confusion = {sample_confusion}, \n'
                    f'test_score = {test_score}, \n'
                    f'test_confusion ={test_confusion}\n')

                self.top_k_save(test_score, clf)

        callback = Callback('./instance/TGS_salt_cnn_top_k')
        clf.train(train_x, train_y_onehot, epoch=n_epoch, epoch_callback=callback,
                  batch_size=64, iter_pbar=True, early_stop=early_stop, patience=patience)

        clf.save('./instance/test')

        score = clf.score(sample_x, sample_y_onehot)
        test_score = clf.score(test_x, test_y_onehot)
        print(f'score = {score}, test= {test_score}')


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
        train_images = train_images.reshape([-1, 101, 101])

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
