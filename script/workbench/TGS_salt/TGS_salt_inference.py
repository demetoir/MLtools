from pprint import pprint
from imgaug import augmenters as iaa
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from script.data_handler.ImgMaskAug import ActivatorMask, ImgMaskAug
from script.data_handler.TGS_salt import collect_images, TRAIN_MASK_PATH, TGS_salt, \
    TRAIN_IMAGE_PATH, TEST_IMAGE_PATH, RLE_mask_encoding
from script.model.sklearn_like_model.BaseModel import BaseDatasetCallback
from script.model.sklearn_like_model.TFSummary import TFSummaryParams
from script.util.PlotTools import PlotTools
from script.util.misc_util import path_join, lazy_property
from script.util.numpy_utils import *
import tensorflow as tf


class Metrics:
    @staticmethod
    def miou(trues, predicts):
        return np.mean(Metrics.iou_vector(trues, predicts))

    @staticmethod
    def iou_vector(trues, predicts):
        return [
            Metrics.iou(gt, predict)
            for gt, predict in zip(trues, predicts)
        ]

    @staticmethod
    def iou(true, predict):
        true = true.astype(np.int32)
        predict = predict.astype(np.int32)

        # zero rate mask will include, 1
        intersection = np.logical_and(true, predict)
        union = np.logical_or(true, predict)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        return iou

    @staticmethod
    def TGS_salt_score(mask_true, mask_predict):
        def _metric(mask_true, mask_predict):
            iou_score = Metrics.iou(mask_true, mask_predict)
            threshold = np.arange(0.5, 1, 0.05)
            score = np.sum(threshold <= iou_score) / 10.0
            return score

        if mask_true.shape != mask_predict.shape:
            raise ValueError(f'mask shape does not match, true={mask_true.shape}, predict={mask_predict}')

        if mask_true.ndim in (3, 4):
            ret = np.mean([_metric(m_true, m_predict) for m_true, m_predict in zip(mask_true, mask_predict)])
        else:
            ret = _metric(mask_true, mask_predict)

        return ret

    @staticmethod
    def miou_non_empty(true, predict):
        non_empty = np.mean(true, axis=(1, 2, 3))
        idx = non_empty > 0
        return Metrics.miou(true[idx], predict[idx])


    @staticmethod
    def TGS_salt_score_non_empty(true, predict):
        non_empty = np.mean(true, axis=(1, 2, 3))
        idx = non_empty > 0
        return Metrics.TGS_salt_score(true[idx], predict[idx])

def masks_rate(masks):
    size = masks.shape[0]
    mask = masks.reshape([size, -1])
    return np.mean(mask, axis=1)


def save_tf_summary_params(path, params):
    with tf.Session() as sess:
        run_id = params['run_id']
        path = path_join(path, run_id)
        summary_params = TFSummaryParams(path, 'params')
        summary_params.update(sess, params)
        summary_params.flush()
        summary_params.close()
        print(f'TFSummaryParams save at {path}')


def is_empty_mask(mask):
    return np.mean(mask) == 0


def depth_to_image(depths):
    # normalize
    max_val = np.max(depths)
    min_val = np.min(depths)
    depths = (depths - min_val) / (max_val - min_val)

    # gen depth images
    base = [
        np.ones([1, 101, 101]) * depth * 255
        for depth in depths
    ]
    base = np.concatenate(base, axis=0)
    base = base.astype(np.uint8)
    return base


class TGS_salt_DataHelper:
    def __init__(self, data_pack_path='./data/TGS_salt', sample_offset=10, sample_size=10):
        self.data_pack_path = data_pack_path
        self.sample_offset = sample_offset
        self.sample_size = sample_size

        self._data_pack = None
        self._train_set = None
        self._test_set = None
        self._sample_xs = None
        self._sample_ys = None
        self._train_set_non_empty_mask = None
        self._train_set_empty_mask = None
        self._train_depth_image = None

    @lazy_property
    def data_pack(self):
        self._data_pack = TGS_salt()
        self._data_pack.load(self.data_pack_path)

        return self._data_pack

    @lazy_property
    def train_set(self):
        return self.data_pack['train']

    @lazy_property
    def test_set(self):
        return self.data_pack['test']

    @lazy_property
    def sample_xs(self):
        x_full, _ = self.train_set.full_batch()
        sample_x = x_full[self.sample_offset:self.sample_offset + self.sample_size]
        return sample_x

    @lazy_property
    def sample_ys(self):
        _, ys_full = self.train_set.full_batch()
        self._sample_ys = ys_full[self.sample_offset:self.sample_offset + self.sample_size]
        return self._sample_ys

    def get_non_empty_mask_idxs(self, dataset):
        ys = dataset.full_batch(['mask'])['mask']
        idxs = [
            i
            for i, y in enumerate(ys)
            if not is_empty_mask(y)
        ]
        return idxs

    def get_non_empty_mask(self, dataset):
        idxs = self.get_non_empty_mask_idxs(dataset)
        return dataset.query_by_idxs(idxs)

    def get_empty_mask_idxs(self, dataset):
        ys = dataset.full_batch(['mask'])['mask']
        idxs = [
            i
            for i, y in enumerate(ys)
            if is_empty_mask(y)
        ]
        return idxs

    def get_empty_mask(self, dataset):
        idxs = self.get_empty_mask_idxs(dataset)
        return dataset.query_by_idxs(idxs)

    def add_depth_image_channel(self, dataset):
        np_dict = dataset.full_batch(['image', 'depth_image'])
        x = np_dict['image']
        depth_image = np_dict['depth_image']
        x_with_depth = np.concatenate((x, depth_image), axis=3)
        dataset.add_data('x_with_depth', x_with_depth)

        return dataset

    def mask_rate_under_n_percent(self, dataset, n):
        mask_rate = dataset.full_batch(['mask_rate'])['mask_rate']

        idx = mask_rate < n
        return dataset.query_by_idxs(idx)

    def mask_rate_upper_n_percent(self, dataset, n):
        mask_rate = dataset.full_batch(['mask_rate'])['mask_rate']
        idx = mask_rate > n
        return dataset.query_by_idxs(idx)

    def lr_flip(self, dataset, x_key='image', y_key='mask'):
        flip_lr_set = dataset.copy()
        x, y = flip_lr_set.full_batch()
        x = np.fliplr(x)
        flip_lr_set.data[x_key] = x
        y = np.fliplr(y)
        flip_lr_set.data[y_key] = y
        dataset = dataset.merge(dataset, flip_lr_set)
        return dataset

    def split_hold_out(self, dataset, random_state=1234, ratio=(9, 1)):
        return dataset.split(ratio, shuffle=False, random_state=random_state)

    def k_fold_split(self, dataset, k=5, shuffle=False, random_state=1234):
        return dataset.k_fold_split(k, shuffle=shuffle, random_state=random_state)


class TGS_salt_aug_callback(BaseDatasetCallback):
    def __init__(self, x, y, batch_size, n_job=2, q_size=100):
        super().__init__(x, y, batch_size)

        self.seq = iaa.Sequential([
            # iaa.OneOf([
            #     iaa.PiecewiseAffine((0.002, 0.1), name='PiecewiseAffine'),
            #     iaa.Affine(rotate=(-20, 20)),
            #     iaa.Affine(shear=(-45, 45)),
            #     iaa.Affine(translate_percent=(0, 0.3), mode='symmetric'),
            #     iaa.Affine(translate_percent=(0, 0.3), mode='wrap'),
            #     # iaa.PerspectiveTransform((0.0, 0.3))
            # ], name='affine'),
            iaa.Fliplr(0.5, name="horizontal flip"),
            # iaa.Crop(percent=(0, 0.3), name='crop'),

            # image only
            # iaa.OneOf([
            #     iaa.Add((-45, 45), name='bright'),
            #     iaa.Multiply((0.5, 1.5), name='contrast')]
            # ),
            # iaa.OneOf([
            #     iaa.AverageBlur((1, 5), name='AverageBlur'),
            #     # iaa.BilateralBlur(),
            #     iaa.GaussianBlur((0.1, 2), name='GaussianBlur'),
            #     # iaa.MedianBlur((1, 7), name='MedianBlur'),
            # ], name='blur'),

            # scale to  128 * 128
            # iaa.Scale((128, 128), name='to 128 * 128'),
        ])
        self.activator = ActivatorMask(['bright', 'contrast', 'AverageBlur', 'GaussianBlur', 'MedianBlur'])
        self.aug = ImgMaskAug(self.x, self.y, self.seq, self.activator, self.batch_size, n_jobs=n_job, q_size=q_size)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

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
    @staticmethod
    def is_empty_mask(mask):
        return np.mean(mask) == 0

    @staticmethod
    def is_white_image(image):
        if np.mean(image) == 255:
            return True
        else:
            return False

    @staticmethod
    def is_black_image(image):
        if np.mean(image) == 0:
            return True
        else:
            return False

    @staticmethod
    def masking_images(image, mask, mask_rate=.8):
        image = np.array(image)
        if image.ndim != 3:
            raise ValueError('image ndim must 3')

        image[:, :, 0] = mask * mask_rate

        return image

    @staticmethod
    def masks_rate(masks):
        size = masks.shape[0]
        mask = masks.reshape([size, -1])
        return np.mean(mask, axis=1)


plot = PlotTools(save=True, show=False)


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
            masked_images += [data_helper.masking_images(image, mask)]
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
        for image, mask, id_ in zip(images, masks, ids):
            a = np.sum(mask) / (101 * 101 * 1 * 255)
            print(id_, a)
            if a > 0.6:
                masked += [data_helper.masking_images(image, mask)]
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
        for image, mask, id_ in zip(images, masks, ids):
            a = np.sum(image) / (101 * 101 * 3 * 255)
            print(id_, a)
            if a > 0.85:
                masked += [data_helper.masking_images(image, mask)]
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
        for image, mask, id_ in zip(images, masks, ids):
            a = np.sum(image) / (101 * 101 * 3 * 255)
            print(id_, a)
            if a < 0.20:
                masked += [data_helper.masking_images(image, mask)]
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
        for image, mask, id_ in zip(images, masks, ids):
            rle_mask = RLE_mask_encoding(mask.reshape([101, 101]).transpose())
            n_rle_mask = len(rle_mask)
            a = n_rle_mask
            mask_area = np.sum(mask) / (101 * 101 * 255)

            if 0 < a / 2 < 8 and 0.1 < mask_area < 0.99:
                print(id_, a, rle_mask)
                masked += [data_helper.masking_images(image, mask)]

        masked = np.array(masked)
        print(len(masked))
        plot.plot_image_tile(masked, title='chopped')
