from script.data_handler.TGS_salt import collect_images, make_data_pkl, TGS_salt, TRAIN_MASK_PATH, RLE_mask_encoding
import numpy as np

from script.util.deco import deco_timeit


def test_is_from_grayscale():
    """TGS salt dataset is from grayscale"""
    path = './data/TGS_salt/test/images'
    images, names, ids = collect_images(path)
    for i in range(len(images)):
        print(i)
        image = images[i]
        c1 = image[:, :, 0]
        c2 = image[:, :, 1]
        c3 = image[:, :, 2]
        if not np.array_equal(c1, c2) or not np.array_equal(c2, c3) or not np.array_equal(c3, c1):
            print('not equal')
            break

    path = './data/TGS_salt/train/images'
    images, names, ids = collect_images(path)
    for i in range(len(images)):
        print(i)
        image = images[i]
        c1 = image[:, :, 0]
        c2 = image[:, :, 1]
        c3 = image[:, :, 2]
        if not np.array_equal(c1, c2) or not np.array_equal(c2, c3) or not np.array_equal(c3, c1):
            print('not equal')
            break


def test_make_data_pkl():
    make_data_pkl()


def test_load_dataset():
    datapack = TGS_salt()
    datapack.load('./data/TGS_salt')



def test_RL_encoding():
    images, _, _ = collect_images(TRAIN_MASK_PATH)
    images = np.transpose(images, (0, 2, 1, 3))
    images = np.reshape(images, [-1, 101, 101])

    @deco_timeit
    def encoding_timeit(images):
        return RLE_mask_encoding(images)

    encoded = encoding_timeit(images)

