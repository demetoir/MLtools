import os
import cv2
import numpy as np
from glob import glob
import pandas as pd
from script.data_handler.Base.BaseDataset import BaseDataset
from script.data_handler.Base.BaseDatasetPack import BaseDatasetPack
from script.util.image_utils import PIL_img_from_file, PIL_img_to_np_img
from script.util.misc_util import path_join, load_pickle, dump_pickle

HEAD_PATH = './data/TGS_salt'
DEPTHS_CSV_PATH = path_join(HEAD_PATH, 'depths.csv')
TRAIN_CSV_PATH = path_join(HEAD_PATH, 'train.csv')
SUBMISSION_CSV_PATH = path_join(HEAD_PATH, 'sample_submission.csv')
TRAIN_IMAGE_PATH = path_join(HEAD_PATH, 'train/images')
TRAIN_MASK_PATH = path_join(HEAD_PATH, 'train/masks')
TEST_IMAGE_PATH = path_join(HEAD_PATH, 'test/images')
TRAIN_PKL_PATH = path_join(HEAD_PATH, 'train.pkl')
TEST_PKL_PATH = path_join(HEAD_PATH, 'test.pkl')
MERGE_CSV_PATH = path_join(HEAD_PATH, 'merge.csv')

size_h = 101
size_w = 101
size_c = 3


def collect_images(path, limit=None):
    img_paths = glob(path_join(path, '*'))
    if limit is not None:
        img_paths = img_paths[:limit]

    images = [PIL_img_to_np_img(PIL_img_from_file(path)) for path in sorted(img_paths)]
    images_names = [os.path.split(path)[1] for path in img_paths]
    ids = [name.split('.')[0] for name in images_names]

    # drop channel except one
    size = len(img_paths)
    images = np.array(images).reshape([size, size_h, size_w, -1])
    images = images[:, :, :, 0]
    images = np.reshape(images, [size, 101, 101, 1])

    names = np.array(images_names)
    ids = np.array(ids)

    return images, names, ids


def add_feature_mask_pixel_rate(mask, mask_value=255):
    return np.mean(mask) / mask_value


def get_feature_mask_rate(masks):
    return np.array([add_feature_mask_pixel_rate(mask) for mask in masks])


def get_feature_empty_mask(masks):
    return np.array([np.mean(mask) == 0 for mask in masks])


def get_feature_weired_mask(masks):
    def f(mask):
        encoding = RLE_mask_encoding(mask.reshape([101, 101]).transpose())
        encoding_len = len(encoding)
        mask_rate = add_feature_mask_pixel_rate(mask)

        if 0 < encoding_len / 2 < 8 and 0.1 < mask_rate < 1:
            return 1
        else:
            return 0

    return np.array([f(mask) for mask in masks])


def depth_to_image(depths):
    # normalize
    max_val = np.max(depths)
    min_val = np.min(depths)
    depths = (depths - min_val) / (max_val - min_val)

    # gen depth images
    base = [
        np.ones([1, 101, 101, 1]) * depth * 255
        for depth in depths
    ]
    base = np.concatenate(base, axis=0)
    base = base.astype(np.uint8)
    return base


def make_data_pkl():
    print(f'collect train images')
    train_images, train_image_names, train_ids = collect_images(TRAIN_IMAGE_PATH)

    print(f'collect train mask images')
    train_mask, train_mask_names, train_mask_ids = collect_images(TRAIN_MASK_PATH)

    print(f'collect test images')
    test_images, test_image_names, test_ids = collect_images(TEST_IMAGE_PATH)

    print(f'collect csv files')
    df_depths = pd.read_csv(DEPTHS_CSV_PATH)
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_train.fillna('none', inplace=True)

    df_merge = pd.merge(left=df_depths, right=df_train, how='outer', left_on='id', right_on='id')
    df_merge.to_csv(MERGE_CSV_PATH, index=False)

    print(f'collect train depth')
    train_depths = df_merge[df_merge['rle_mask'].notna()]
    train_depths = pd.DataFrame(train_depths).sort_values('id')
    train_depths = train_depths.reset_index(drop=True)
    train_depths = train_depths['z']

    print(f'collect test depth')
    test_depths = df_merge[df_merge['rle_mask'].isna()]
    test_depths = pd.DataFrame(test_depths).sort_values('id')
    test_depths = test_depths.reset_index(drop=True)
    test_depths = test_depths['z']

    print(f'collect train mask rate')
    train_mask_rate = get_feature_mask_rate(train_mask)

    print(f'collect train empty mask')
    train_empty_mask = get_feature_empty_mask(train_mask)

    print(f'collect train weired mask')
    train_weired_mask = get_feature_weired_mask(train_mask)

    print(f'collect train depth_image')
    train_depths_image = depth_to_image(train_depths)

    print(f'collect test depth_image')
    test_depths_image = depth_to_image(test_depths)

    print('dump train pickle')
    train_pkl = {
        'image': train_images,
        'mask': train_mask,
        'id': train_ids,
        'depths': train_depths,
        'mask_rate': train_mask_rate,
        'empty_mask': train_empty_mask,
        'is_weired_mask': train_weired_mask,
        'depth_image': train_depths_image,
    }
    dump_pickle(train_pkl, TRAIN_PKL_PATH)

    print('dump test pickle')
    test_pkl = {
        'image': test_images,
        'id': test_ids,
        'depths': test_depths,
        'depth_image': test_depths_image
    }
    dump_pickle(test_pkl, TEST_PKL_PATH)


def _RLE_mask_encoding(np_arr):
    h, w = np_arr.shape
    np_arr = np.reshape(np_arr, [-1])

    encode = np.argwhere(np.diff(np_arr)) + 2
    encode = encode.reshape([-1])

    if np_arr[0] == 255:
        encode = np.concatenate([[1], encode])

    if np_arr[-1] == 255:
        encode = np.concatenate([encode, [h * w + 1]])

    encode[1::2] = encode[1::2] - encode[::2]

    return encode


def RLE_mask_encoding(np_arr):
    if np_arr.ndim == 3:
        return [_RLE_mask_encoding(np_arr) for np_arr in np_arr]
    else:
        return _RLE_mask_encoding(np_arr)


def make_submission_csv(ids, masks):
    # TODO test
    masks = np.transpose(masks, (0, 2, 1, 3))
    masks = np.reshape(masks, [-1, 101, 101])

    rle_masks = RLE_mask_encoding(masks)
    df = pd.DataFrame({'id': ids, 'rle_mask': rle_masks})
    df = df[['id', 'rle_mask']]
    df_sample = pd.read_csv(SUBMISSION_CSV_PATH)
    df_submission = pd.merge(left=df, right=df_sample, how='inner', left_on='id', right_on='id')
    return df_submission


def load_sample_image():
    sample_IMAGE_PATH = path_join(HEAD_PATH, 'sample/images')
    sample_MASK_PATH = path_join(HEAD_PATH, 'sample/masks')

    sample_size = 7
    limit = None
    print(f'collect sample images')
    train_images, _, _ = collect_images(sample_IMAGE_PATH, limit=limit)
    train_images = train_images.reshape([-1, 101, 101, 1])
    print(f'collect sample images')
    train_mask_images, _, _ = collect_images(sample_MASK_PATH, limit=limit)
    train_mask_images = train_mask_images.reshape([-1, 101, 101, 1])
    x = train_images
    y = train_mask_images

    return x, y


def to_128(x):
    x = np.array([cv2.resize(a, (128, 128)) for a in x]).reshape([-1, 128, 128, 1])
    return x


def to_101(x):
    x = np.array([cv2.resize(a, (128, 128)) for a in x]).reshape([-1, 128, 128, 1])
    return x


class mask_label_encoder:
    @staticmethod
    def to_label(x):
        return np.array(x / 255, dtype=int)

    @staticmethod
    def from_label(x):
        return np.array(x * 255, dtype=float)


class train_set(BaseDataset):

    def load(self, path):
        pkl_path = path_join(path, 'train.pkl')
        if not os.path.exists(pkl_path) or not self.caching:
            make_data_pkl()

        pkl = load_pickle(pkl_path)

        self.add_data('image', pkl['image'])
        self.add_data('id', pkl['id'])
        self.add_data('mask', pkl['mask'])
        self.add_data('depth', pkl['depths'])
        self.add_data('mask_rate', pkl['mask_rate'])
        self.add_data('empty_mask', pkl['empty_mask'])
        self.add_data('is_weired_mask', pkl['is_weired_mask'])
        self.add_data('depth_image', pkl['depth_image'])

        # self.x_keys = ['image', 'depth']
        self.x_keys = ['image']
        self.y_keys = ['mask']


class test_set(BaseDataset):
    def load(self, path):
        pkl_path = path_join(path, 'test.pkl')
        if not os.path.exists(pkl_path) or not self.caching:
            make_data_pkl()

        pkl = load_pickle(pkl_path)

        self.add_data('image', pkl['image'])
        self.add_data('id', pkl['id'])
        self.add_data('depth', pkl['depths'])
        self.add_data('depth_image', pkl['depth_image'])
        # self.x_keys = ['image', 'depth']
        self.x_keys = ['image']


class TGS_salt(BaseDatasetPack):

    def __init__(self, caching=True, verbose=0, **kwargs):
        super().__init__(caching, verbose, **kwargs)
        self.pack['train'] = train_set(caching, verbose, caching=self.caching)
        self.pack['test'] = test_set(caching, verbose, caching=self.caching)
