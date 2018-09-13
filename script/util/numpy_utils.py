"""numpy util

* np_img(single numpy image ) is numpy array with shape WHC
* np_imgs(multiple numpy images) is numpy array with shape NWHC
"""
from PIL import Image
from skimage import color, io
import math
import numpy as np
from scipy.stats import rankdata
from script.util.misc_util import setup_file

NpArr = np.array


def np_img_float32_to_uint8(np_img):
    """type cast numpy image from float to uint8

    :type np_img: numpy.array
    :param np_img: numpy array image

    :return: uint8 type numpy image(HWC shape)
    """
    return np.uint8(np_img * 255)


def np_img_uint8_to_float32(np_img):
    """type cast numpy image from uint8 to float32

    :type np_img: numpy.array
    :param np_img: numpy array image

    :return: float32 type numpy image(HWC shape)
    """
    return np.float32(np_img) / 255.0


def np_img_rgb_to_gray(np_img):
    """convert numpy image from rgb to gary scale

    :type np_img: numpy.array
    :param np_img: numpy array image

    :return: gray scale numpy image(HWC shape)
    """
    return np.uint8(color.rgb2gray(np_img) * 255)


def np_img_gray_to_rgb(np_img):
    """convert numpy image from gray scale to rgb

    :type np_img: numpy.array
    :param np_img: numpy array image

    :return: rgb numpy image(HWC shape)
    """
    return color.gray2rgb(np_img)


def np_img_to_PIL_img(np_img):
    """convert numpy image to PIL Image

    :type np_img: numpy.array
    :param np_img: numpy image

    :return: PIL Image
    """
    return Image.fromarray(np_img)


def np_img_load_file(path):
    """load numpy image from file

    :type path: str
    :param path:path to load image file

    :return: numpy image(HWC shape)
    """
    return io.imread(path)


def np_img_to_tile(np_imgs, column_size=10, padding=3, padding_value=0):
    """convert multiple numpy images to one grid tile image

    :type np_imgs: numpy.array
    :type column_size: int
    :param np_imgs: numpy images
    numpy image shape expect NHWC
    :param column_size: column size of grid tile image
    :param padding_value: value to padding area

    :return: gird tiled numpy image(HWC shape)
    """

    def np_imgs_check_shape(shape):
        if len(shape) == 3:
            return 'NHW'
        elif len(shape) == 4:
            return 'NHWC'
        else:
            raise TypeError("shape %s is not np_img type" % str(shape))

    if np_imgs_check_shape(np_imgs.shape) == 'NHW':
        np_imgs = np_img_gray_to_rgb(np_imgs)

        padder = ((0, 0), (padding, padding), (padding, padding))
        const_val = ((padding_value, padding_value), (padding_value, padding_value), (padding_value, padding_value))
    else:
        padder = ((0, 0), (padding, padding), (padding, padding), (0, 0))
        const_val = (
            (padding_value, padding_value), (padding_value, padding_value), (padding_value, padding_value),
            (padding_value, padding_value))

    np_imgs = np.pad(np_imgs, padder, 'constant', constant_values=const_val)

    n, h, w, c = np_imgs.shape
    row_size = int(math.ceil(float(np_imgs.shape[0]) / float(column_size)))
    tile_width = w * column_size
    tile_height = h * row_size

    img_idx = 0
    tile = np.ones((tile_height, tile_width, 3), dtype=np.uint8)
    for y in range(0, tile_height, h):
        for x in range(0, tile_width, w):
            if img_idx >= n:
                break

            tile[y:y + h, x:x + w, :] = np_imgs[img_idx]

            img_idx += 1

        if img_idx >= n:
            break

    return tile


def np_imgs_NCWH_to_NHWC(imgs):
    """convert numpy images shape NCWH to NHWC

    :type imgs: numpy.array
    :param imgs: numpy images

    :return: numpy images(NHWC shape)
    """
    return np.transpose(imgs, [0, 2, 3, 1])


def np_index_to_onehot(x, n=None, dtype=float):
    """1-hot encode x with the max value n (computed from data if n is None).

    :param x: numpy array
    :param n: max value
    :param dtype:data type

    :return: numpy array
    """
    x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    return np.eye(n, dtype=dtype)[x]


def np_onehot_to_index(x, axis=1):
    return np.argmax(x, axis=axis)


NP_ARRAY_TYPE_INDEX = "index"
NP_ARRAY_TYPE_ONEHOT = "onehot"
NP_ARRAY_TYPE_INVALID = 'invalid'
INDEX_TO_ONEHOT = (NP_ARRAY_TYPE_INDEX, NP_ARRAY_TYPE_ONEHOT)
ONEHOT_TO_INDEX = (NP_ARRAY_TYPE_ONEHOT, NP_ARRAY_TYPE_INDEX)
NO_CONVERT = (
    (NP_ARRAY_TYPE_ONEHOT, NP_ARRAY_TYPE_ONEHOT),
    (NP_ARRAY_TYPE_INDEX, NP_ARRAY_TYPE_INDEX)
)


def get_np_arr_type(np_arr):
    shape = np_arr.shape
    if len(shape) == 1:
        _type = NP_ARRAY_TYPE_INDEX
    elif len(shape) == 2:
        _type = NP_ARRAY_TYPE_ONEHOT
    else:
        _type = NP_ARRAY_TYPE_INVALID

    return _type


def reformat_np_arr(np_arr, to_np_arr_type, from_np_arr_type=None, n=None):
    if not is_np_arr(np_arr):
        np_arr = np.array(np_arr)

    if from_np_arr_type is None:
        from_np_arr_type = get_np_arr_type(np_arr)

    convert_type = (from_np_arr_type, to_np_arr_type)

    if convert_type == ONEHOT_TO_INDEX:
        np_arr = np_onehot_to_index(np_arr)
    elif convert_type == INDEX_TO_ONEHOT:
        np_arr = np_index_to_onehot(np_arr, n)
    elif convert_type in NO_CONVERT:
        pass
    else:
        raise TypeError("np_arr reformat Error from '%s' type to '%s' type" % (from_np_arr_type, to_np_arr_type))

    return np_arr


def np_stat_dict(a):
    a = np.array(a)
    return {
        'min': np.round(a.min(), decimals=4),
        'mean': np.round(a.mean(), decimals=4),
        'max': np.round(a.max(), decimals=4),
        'std': np.round(a.std(), decimals=4),
        'count': len(a),
    }


def is_np_arr(x):
    return isinstance(x, np.ndarray)


def np_minmax_normalize(np_x: NpArr, min_=None, max_=None) -> NpArr:
    if min_ is None:
        min_ = np.min(np_x)

    if max_ is None:
        max_ = np.max(np_x)

    return (np_x - min_) / (max_ - min_)


def np_frequency_equal_bins(np_x: NpArr, n_bins: int) -> NpArr:
    np_x_size = np_x.shape[0]
    bin_size = np_x_size // n_bins

    rank = rankdata(np_x, method='dense')
    rank_count = np.bincount(rank)
    accumulate_sum = np.cumsum(rank_count)
    bins = [
        np_x[
            np.argwhere(
                rank == np.searchsorted(accumulate_sum, i, side='right'))
        ][0]
        for i in range(bin_size, np_x_size, bin_size)
    ]

    # for i in range(bin_size, np_x_size, bin_size):
    #     value = np.searchsorted(accumulate_sum, i, side='right')
    #     idx = np.argwhere(rank == value)
    #     bins += [np_x[idx]]

    bins = np.reshape(bins, newshape=[-1])
    bins[-1] = NpArr([np.max(np_x) + 1])
    bins = np.concatenate([
        NpArr([np.min(np_x) - 1]),
        bins,
    ])
    return bins


def np_width_equal_bins(np_x: NpArr, width: int, ) -> NpArr:
    min_ = np.min(np_x) - 1
    max_ = np.max(np_x) + 1
    bins = np.concatenate([np.arange(min_, max_, width), [max_]])
    return bins


def np_image_save(np_img, path):
    """save np_img file in visualizer path

    :type np_img: numpy.Array
    :type path: str
    :param np_img: np_img to save
    :param path: path to save image file
    default None
    if file_name is None, file name of np_img will be 'output_count.png'
    """

    pil_img = np_img_to_PIL_img(np_img)
    setup_file(path)
    with open(path, 'wb') as fp:
        pil_img.save(fp)


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
