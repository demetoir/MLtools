import numpy as np
from pprint import pprint
from imgaug import augmenters as iaa
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from script.data_handler.ImgMaskAug import ActivatorMask, ImgMaskAug
from script.data_handler.TGS_salt import collect_images, TRAIN_MASK_PATH, TGS_salt, \
    mask_label_encoder, TRAIN_IMAGE_PATH, TEST_IMAGE_PATH, RLE_mask_encoding
from script.model.sklearn_like_model.BaseModel import BaseDatasetCallback, BaseEpochCallback
from script.model.sklearn_like_model.ImageClf import ImageClf
from script.model.sklearn_like_model.TFSummary import TFSummary
from script.model.sklearn_like_model.Top_k_save import Top_k_save
from script.model.sklearn_like_model.SemanticSegmentation import SemanticSegmentation
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


def to_dict(**kwargs):
    return kwargs


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

    def to_string(self, params):
        return "_".join([f"{key}={val}" for key, val in params.items()])

    def train(self, params, n_epoch=10, augmentation=False, early_stop=True, patience=20, save_path=None):
        params_str = self.to_string(params)
        model = SemanticSegmentation(**params)

        epoch_callback = Unet_epoch_callback(model, self.test_x, self.test_y, params)
        dataset_callback = TGS_salt_aug_callback if augmentation else None

        # epoch_callback = None
        # dataset_callback = None

        model.train(self.train_x, self.train_y_encode, epoch=n_epoch, aug_callback=dataset_callback,
                    epoch_callback=epoch_callback, early_stop=early_stop, patience=patience,
                    iter_pbar=True)

        if save_path is None:
            save_path = f'./instance/TGS_salt/SS/{params_str}'
        model.save(save_path)


class cnn_EpochCallback(BaseEpochCallback):
    def __init__(self, model, test_x, test_y, sample_x, sample_y):
        super().__init__()
        self.model = model
        self.test_x = test_x
        self.test_y = test_y
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.k = 5
        self.top_k = [np.Inf for _ in range(self.k)]
        self.top_k_save = Top_k_save('./instance/TGS_salt_cnn_top_k')

    def __call__(self, sess, dataset, epoch, log=None):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score

        test_predict = self.model.predict(self.test_x)
        test_score = accuracy_score(self.test_y, test_predict)
        test_confusion = confusion_matrix(self.test_y, test_predict)

        sample_predict = self.model.predict(self.sample_x)
        sample_score = accuracy_score(self.sample_y, sample_predict)
        sample_confusion = confusion_matrix(self.sample_y, sample_predict)
        log(f'e={epoch}, '
            f'sample_score = {sample_score}, \n'
            f'sample_confusion = {sample_confusion}, \n'
            f'test_score = {test_score}, \n'
            f'test_confusion ={test_confusion}\n')

        # self.top_k_save(test_score, clf)


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

        net_capacity = 8
        net_type = 'InceptionV1'
        clf = ImageClf(net_type=net_type, net_capacity=net_capacity)
        Epoch_callback = cnn_EpochCallback(clf, test_x, test_y, sample_x, sample_y)
        clf.train(train_x, train_y_onehot, epoch=n_epoch, epoch_callback=Epoch_callback,
                  batch_size=16, iter_pbar=True,
                  dataset_callback=None, early_stop=early_stop, patience=patience)

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
