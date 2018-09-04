import numpy as np
from script.data_handler.ImgMaskAug import ActivatorMask, ImgMaskAug
from script.data_handler.TGS_salt import load_sample_image
from script.util.PlotTools import PlotTools
from imgaug import augmenters as iaa

plot = PlotTools(save=True, show=False)


def test_imgaug():
    x, y = load_sample_image()

    seq = iaa.Sequential([
        iaa.Fliplr(1, name="Flipper"),
        iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
        iaa.Dropout(0.02, name="Dropout"),
        iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="MyLittleNoise"),
        iaa.AdditiveGaussianNoise(loc=32, scale=0.0001 * 255, name="SomeOtherNoise"),
        iaa.Affine(translate_px={"x": (-40, 40)}, name="Affine")
    ])
    activator = ActivatorMask(["GaussianBlur", "Dropout", "MyLittleNoise"])

    n_iter = 10
    batch_size = 5
    with ImgMaskAug(x, y, seq, activator, batch_size, n_jobs=1) as aug:
        for idx in range(n_iter):
            print(idx)
            image, mask = aug.get_batch()
            a = np.concatenate([image, mask], axis=0)
            plot.plot_image_tile(np.concatenate([x[:5], y[:5], a], axis=0), column=batch_size,
                                 title=f'test_image_aug_{idx}')
