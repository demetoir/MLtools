
import numpy as np

from script.data_handler.TGS_salt import collect_images, TRAIN_MASK_PATH
from script.util.PlotTools import PlotTools

plot = PlotTools(save=True, show=False)


def metric(mask_true, mask_predict):
    # TODO
    def _metric(mask_true, mask_predict):
        if np.sum(mask_true) == 0 and np.sum(mask_predict) > 0:
            return 0
        elif np.sum(mask_true) == 0 and np.sum(mask_predict) == 0:
            return 1
        else:
            threshold = np.arange(0.5, 1, 0.05)

            mask_true = mask_true / 255
            mask_predict = mask_predict / 255

            upper = np.logical_and(mask_true, mask_predict)
            lower = np.logical_or(mask_true, mask_predict)
            iou_score = np.sum(upper) / np.sum(lower)
            print(iou_score)

            print(threshold <= iou_score)
            score = np.sum(threshold <= iou_score) / 10.0
            return score

    if mask_true.shape != mask_predict.shape:
        raise ValueError(f'mask shape does not match, true={mask_true.shape}, predict={mask_predict}')

    if mask_true.ndim == 3:
        ret = np.mean([_metric(m_true, m_predict) for m_true, m_predict in zip(mask_true, mask_predict)])
    else:
        ret = _metric(mask_true, mask_predict)

    return ret


def test_metric():
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
            metric_score += [metric(a, b)]
    metric_score = np.array(metric_score).reshape([size, size])
    print(metric_score)






