from pprint import pprint
from imgaug import augmenters as iaa
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from script.data_handler.Base.BaseDataset import BaseDataset
from script.data_handler.ImgMaskAug import ActivatorMask, ImgMaskAug
from script.data_handler.TGS_salt import collect_images, TRAIN_MASK_PATH, TGS_salt, \
    TRAIN_IMAGE_PATH, TEST_IMAGE_PATH, RLE_mask_encoding, make_submission_csv
from script.model.sklearn_like_model.BaseModel import BaseDatasetCallback
from script.model.sklearn_like_model.TFSummary import TFSummaryParams
from script.util.PlotTools import PlotTools
from script.util.misc_util import path_join, lazy_property, load_pickle
from script.util.numpy_utils import *
import tensorflow as tf

from script.workbench.TGS_salt.post_process_AE import post_process_AE


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

    @staticmethod
    def get_non_empty_mask_idxs(dataset):
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

    @staticmethod
    def get_empty_mask_idxs(dataset):
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

    @staticmethod
    def add_depth_image_channel(dataset):
        np_dict = dataset.full_batch(['image', 'depth_image'])
        x = np_dict['image']
        depth_image = np_dict['depth_image']
        x_with_depth = np.concatenate((x, depth_image), axis=3)
        dataset.add_data('x_with_depth', x_with_depth)

        return dataset

    @staticmethod
    def mask_rate_under_n_percent(dataset, n):
        mask_rate = dataset.full_batch(['mask_rate'])['mask_rate']

        idx = mask_rate < n
        return dataset.query_by_idxs(idx)

    @staticmethod
    def mask_rate_upper_n_percent(dataset, n):
        mask_rate = dataset.full_batch(['mask_rate'])['mask_rate']
        idx = mask_rate > n
        return dataset.query_by_idxs(idx)

    @staticmethod
    def lr_flip(dataset, x_key='image', y_key='mask'):
        flip_lr_set = dataset.copy()
        x, y = flip_lr_set.full_batch()
        x = np.fliplr(x)
        flip_lr_set.data[x_key] = x
        y = np.fliplr(y)
        flip_lr_set.data[y_key] = y
        dataset = dataset.merge(dataset, flip_lr_set)
        return dataset

    @staticmethod
    def split_hold_out(dataset, random_state=1234, ratio=(9, 1)):
        return dataset.split(ratio, shuffle=False, random_state=random_state)

    @staticmethod
    def k_fold_split(dataset, k=5, shuffle=False, random_state=1234):
        return dataset.k_fold_split(k, shuffle=shuffle, random_state=random_state)

    @staticmethod
    def crop_dataset(dataset, size=(64, 64), k=30, with_edge=True):
        xs, ys = dataset.full_batch()

        w, h = size
        new_x = []
        new_y = []
        size = len(xs)
        # edge
        if with_edge:
            for i in range(size):
                x = xs[i]
                y = ys[i]
                new_x += [x[:w, :h, :].reshape([1, h, w, 1])]
                new_y += [y[:w, :h, :].reshape([1, h, w, 1])]

                new_x += [x[101 - w:101, :h, :].reshape([1, h, w, 1])]
                new_y += [y[101 - w:101, :h, :].reshape([1, h, w, 1])]

                new_x += [x[:w, 101 - h:101, :].reshape([1, h, w, 1])]
                new_y += [y[:w, 101 - h:101, :].reshape([1, h, w, 1])]

                new_x += [x[101 - w:101, 101 - h:101, :].reshape([1, h, w, 1])]
                new_y += [y[101 - w:101, 101 - h:101, :].reshape([1, h, w, 1])]

        # non_edge
        for i in range(size):
            for _ in range(k):
                x = xs[i]
                y = ys[i]
                a = np.random.randint(1, 101 - 64 - 1)
                b = np.random.randint(1, 101 - 64 - 1)
                new_x += [x[a:a + w, b:b + h, :].reshape([1, h, w, 1])]
                new_y += [y[a:a + w, b:b + h, :].reshape([1, h, w, 1])]

        new_x = np.concatenate(new_x)
        new_y = np.concatenate(new_y)
        print(new_x.shape)
        print(new_y.shape)

        return BaseDataset(x=new_x, y=new_y)

    @staticmethod
    def crop_dataset_stride(dataset, size=(64, 64), stride=10, with_edge=True):
        xs, ys = dataset.full_batch()

        w, h = size
        new_x = []
        new_y = []
        size = len(xs)
        # edge
        if with_edge:
            for i in range(size):
                x = xs[i]
                y = ys[i]
                new_x += [x[:w, :h, :].reshape([1, h, w, 1])]
                new_y += [y[:w, :h, :].reshape([1, h, w, 1])]

                new_x += [x[101 - w:101, :h, :].reshape([1, h, w, 1])]
                new_y += [y[101 - w:101, :h, :].reshape([1, h, w, 1])]

                new_x += [x[:w, 101 - h:101, :].reshape([1, h, w, 1])]
                new_y += [y[:w, 101 - h:101, :].reshape([1, h, w, 1])]

                new_x += [x[101 - w:101, 101 - h:101, :].reshape([1, h, w, 1])]
                new_y += [y[101 - w:101, 101 - h:101, :].reshape([1, h, w, 1])]

        # non_edge
        for i in range(size):
            for a in range(0, 101 - 64, stride):
                for b in range(0, 101 - 64, stride):
                    x = xs[i]
                    y = ys[i]
                    new_x += [x[a:a + w, b:b + h, :].reshape([1, h, w, 1])]
                    new_y += [y[a:a + w, b:b + h, :].reshape([1, h, w, 1])]

        new_x = np.concatenate(new_x)
        new_y = np.concatenate(new_y)
        print(new_x.shape)
        print(new_y.shape)

        return BaseDataset(x=new_x, y=new_y)


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


class post_processing:

    def fill_hole(self):
        # suck way
        import cv2
        import numpy as np

        # Read image
        im_in = cv2.imread("nickel.jpg", cv2.IMREAD_GRAYSCALE)

        # Threshold.
        # Set values equal to or above 220 to 0.
        # Set values below 220 to 255.

        th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)

        # Copy the thresholded image.
        im_floodfill = im_th.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv

        # Display images.
        cv2.imshow("Thresholded Image", im_th)
        cv2.imshow("Floodfilled Image", im_floodfill)
        cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
        cv2.imshow("Foreground", im_out)
        cv2.waitKey(0)

    def fill_hole2(self):
        # suck way
        train_predict, valid_predict = self.load_SS_predict()

        def fill_hole(xs):
            #
            def batch(x):
                import numpy as np
                import skimage.morphology, skimage.data

                labels = skimage.morphology.label(x)
                labelCount = np.bincount(labels.ravel())
                background = np.argmax(labelCount)
                x[labels != background] = 255
                return x

            return np.concatenate(
                [
                    batch(x)
                    for x in xs
                ],
                axis=0
            )

        a = fill_hole(train_predict)
        a = a.reshape([-1, 101, 101, 1])
        print(a.shape)
        print(a[:10])
        plot.plot_image_tile(a, path=f"./matplot/fill_hole_train.png")

    def AE(self):
        ae = post_process_AE(capacity=32, batch_size=32)
        pprint(ae)
        # set_dict = data_dict().post_AE_mask_only()
        set_dict = None
        train_x = set_dict['train_x']
        valid_x = set_dict['valid_x']
        # ae.load(f"./instance/post_process_AE")
        ae.build(x=train_x, y=train_x)

        # ae.restore(f"./instance/post_process_AE")

        def to_scale(x):
            return x / 255.

        def from_scale(x):
            return x * 255.

        train_x_scale = to_scale(train_x)
        valid_x_scale = to_scale(valid_x)

        train_metric = ae.metric(train_x_scale, train_x_scale)
        valid_metric = ae.metric(valid_x_scale, valid_x_scale)

        print(f'train metric = {train_metric}')
        print(f'test metric = {valid_metric}')

        # print(train_x_scale)
        for i in range(1):
            ae.train(train_x_scale, train_x_scale, epoch=1)
            train_metric = ae.metric(train_x_scale, train_x_scale)
            valid_metric = ae.metric(valid_x_scale, valid_x_scale)

            print(f'train metric = {train_metric}')
            print(f'test metric = {valid_metric}')

            recon = ae.recon(train_x_scale)
            recon = from_scale(recon)
            # print(recon)
            # recon = recon.reshape([-1, 101, 101, 1])
            plot.plot_image_tile(recon, path=f'./matplot/train/recon_{str(i*10)}.png')

            recon = ae.recon(valid_x_scale)
            recon = from_scale(recon)
            # print(recon)
            # recon = recon.reshape([-1, 101, 101, 1])
            plot.plot_image_tile(recon, path=f'./matplot/test/recon_{str(i*10)}.png')

        # ae.save(f"./instance/post_process_AE")

    @property
    def ae(self):
        if not getattr(self, '_ae', None):
            ae = post_process_AE(
                capacity=8, batch_size=64, learning_rate=0.01
            )

            ae.build(x=(101, 101, 1), y=(101, 101, 1))
            # ae.restore(f"./instance/post_process_AE")
            self._ae = ae

        return self._ae

    def post_processing(self, predict):
        ae = self.ae

        def to_scale(x):
            return x / 255.

        def from_scale(x):
            return x * 255.

        predict = to_scale(predict)
        pp_predict = ae.recon(predict)
        pp_predict = from_scale(pp_predict)
        return pp_predict

    def ae_score(self, x, y):

        def to_scale(x):
            return x / 255.

        def from_scale(x):
            return x * 255.

        x = to_scale(x)
        y = to_scale(y)
        x = self.ae.metric(x, y)
        return x

    @staticmethod
    def scramble_column(*args, size=10):
        ret = []
        for i in range(0, len(args[0]), size):
            for j in range(len(args)):
                ret += [args[j][i:i + size]]

        return np.concatenate(ret, axis=0)

    def log_post_process(self, y, predict, post_predict):

        before_tgs_score = Metrics.TGS_salt_score(y, predict)
        before_miou_score = Metrics.miou(y, predict)

        after_tgs_score = Metrics.TGS_salt_score(y, post_predict)
        after_miou_score = Metrics.miou(y, post_predict)

        print(
            f"before TGS score = {before_tgs_score}\n"
            f"before miou score = {before_miou_score}\n"
            f"after TGS score = {after_tgs_score}\n"
            f"after miou score = {after_miou_score}\n"
            f"diff = {after_tgs_score - before_tgs_score}\n"
            f"\n"
        )

    def load_SS_predict(self):
        # set_dict = data_dict().non_empty_with_depth()

        # train_x = set_dict['train_x']
        # train_y = set_dict['train_y']
        # valid_x = set_dict['valid_x']
        # valid_y = set_dict['valid_y']

        train_predict = load_pickle('./train_predict')
        valid_predict = load_pickle('./valid_predict')

        # SS_pipe = SS_baseline().load_baseline()
        # ss = SS_pipe.model
        # train_predict = ss.predict(train_x)
        # train_predict = mask_label_encoder.from_label(train_predict)
        # valid_predict = ss.predict(valid_x)
        # valid_predict = mask_label_encoder.from_label(valid_predict)
        # dump_pickle(train_predict, './train_predict')
        # dump_pickle(valid_predict, './valid_predict')
        # return

        # print(train_predict[0])
        # print(train_y[0])

        return train_predict, valid_predict

    def apply_pp(self):
        set_dict = None

        train_x = set_dict['train_x']
        train_y = set_dict['train_y']
        valid_x = set_dict['valid_x']
        valid_y = set_dict['valid_y']

        train_predict, valid_predict = self.load_SS_predict()

        for i in range(100):
            def to_scale(x):
                return x / 255.

            self.ae.train(to_scale(train_predict), to_scale(train_y))

            def normalize(x):
                threash_hold = 255 * 0.5
                new_x = np.zeros_like(x)
                new_x[x > threash_hold] = 255
                new_x[x <= threash_hold] = 0
                return new_x

            pp_train_predict = self.post_processing(to_scale(train_predict))
            pp_valid_predict = self.post_processing(to_scale(valid_predict))
            pre_normalize = pp_valid_predict
            pp_train_predict = normalize(pp_train_predict)
            pp_valid_predict = normalize(pp_valid_predict)

            # train_tile = self.scramble_column(train_y[:30], train_predict[:30], pp_train_predict[:30])
            valid_tile = self.scramble_column(
                valid_y[i * 30:i * 30 + 30],
                valid_predict[i * 30:i * 30 + 30],
                pp_valid_predict[i * 30:i * 30 + 30],
                pre_normalize[i * 30:i * 30 + 30])
            # plot.plot_image_tile(train_tile, path=f'./matplot/train_tile_{i}.png')
            plot.plot_image_tile(valid_tile, path=f'./matplot/valid_tile_{i}.png')

            print('train')
            self.log_post_process(train_y, train_predict, pp_train_predict)

            print('valid')
            self.log_post_process(valid_y, valid_predict, pp_valid_predict)

    def TTA(self, predict):
        pass


def crop_empty_mask_rate(crop_set):
    x, y = crop_set.full_batch()

    size = len(x)
    c = 0
    for i in y:
        if np.sum(i) == 0:
            c += 1
    print(c / size, size)


def merge_predict(model, x, true):
    def crop(x):
        up_left = None
        up_right = None
        down_left = None
        down_right = None
        return up_left, up_right, down_left, down_right


def merge_clf_SS():
    empty_mask_predict = np.load(f'./empty_mask_predict.np.npy')
    mask_predict = np.load(f'test_merge.np.npy')

    # print(mask_predict)
    threshold = -0.1067679754257063
    # threshold = 0
    mask_predict = mask_predict > threshold
    mask_predict = mask_predict.astype(np.float32)

    print(empty_mask_predict.shape)
    print(mask_predict.shape)

    # print(empty_mask_predict)
    # print(mask_predict)
    # mask_predict[empty_mask_predict == 1] *= 0
    # plot.plot_image_tile(mask_predict*255, path='./predict.png')

    # for idx in empty_mask_predict == 0:
    #     a = np.zeros([101, 101, 1])
    #     print(a.shape)
    #     a[0, :, :] += 1
    #     mask_predict[idx] = a
    # plot.plot_image_tile(mask_predict * 255, path='./predict.png')

    np.save('test_predict_merge', mask_predict)

    helper = TGS_salt_DataHelper()
    test_set = helper.test_set
    print(test_set)
    ids = test_set.full_batch(['id'])['id']
    print(ids)

    make_submission_csv(ids, mask_predict)
    pass
