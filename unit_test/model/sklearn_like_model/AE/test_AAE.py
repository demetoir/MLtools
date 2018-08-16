from pprint import pprint
import numpy as np
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.AE.AAE import AAE
from script.sklearn_like_toolkit.param_search.param_grid import param_grid_full
from script.util.PlotTools import PlotTools
from script.util.misc_util import params_to_dict
from script.util.numpy_utils import np_img_float32_to_uint8
from script.util.pandas_util import DF

class_ = AAE
data_pack = DatasetPackLoader().load_dataset("MNIST")
train_set = data_pack['train']
full_Xs, full_Ys = train_set.full_batch()
sample_Xs = full_Xs[:2]
sample_Ys = full_Ys[:2]


def AAE_total_execute(model):
    model.train(full_Xs, full_Ys, epoch=1)

    code = model.code(sample_Xs)
    print("code {code}".format(code=code))

    recon = model.recon(sample_Xs, sample_Ys)
    print("recon {recon}".format(recon=recon))

    loss = model.metric(sample_Xs, sample_Ys)
    print("loss {}".format(loss))

    # generate(zs, Ys)

    proba = model.predict_proba(sample_Xs)
    print("proba {}".format(proba))

    predict = model.predict(sample_Xs)
    print("predict {}".format(predict))

    score = model.score(sample_Xs, sample_Ys)
    print("score {}".format(score))

    path = model.save()

    model = class_()
    model.load(path)
    print('model reloaded')

    code = model.code(sample_Xs)
    print("code {code}".format(code=code))

    recon = model.recon(sample_Xs, sample_Ys)
    print("recon {recon}".format(recon=recon))

    loss = model.metric(sample_Xs, sample_Ys)
    print("loss {}".format(loss))

    # generate(self, zs, Ys)

    proba = model.predict_proba(sample_Xs)
    print("proba {}".format(proba))

    predict = model.predict(sample_Xs)
    print("predict {}".format(predict))

    score = model.score(sample_Xs, sample_Ys)
    print("score {}".format(score))


def test_AAE():
    model = class_()
    AAE_total_execute(model)


def test_AAE_with_noise():
    class_ = AAE
    data_pack = DatasetPackLoader().load_dataset("MNIST")
    dataset = data_pack['train']
    Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
    sample_X = Xs[:2]
    sample_Y = Ys[:2]

    model = class_(with_noise=True)
    AAE_total_execute(model)


def test_AAE_latent_space():
    x = train_set.Ys_index_label
    idxs_labels = []
    for i in range(10):
        idxs_label = np.where(x == i)
        idxs_labels += [(full_Xs[idxs_label], full_Ys[idxs_label])]

    plot = PlotTools()

    model = AAE
    save_path = './CVAE.pkl'

    BOOLs = [True, False]
    linear_stack_activation = ['sigmoid', 'tanh', 'relu', 'lrelu', 'elu']
    param_grid = params_to_dict(
        latent_code_size=[2],
        learning_rate=[0.01],
        encoder_net_shapes=[(512, 256)],
        decoder_net_shapes=[(256, 512)],
        batch_size=[100],
        # KL_D_rate=[0.01],
        encoder_kwargs=params_to_dict(
            linear_stack_bn=[False],
            linear_stack_activation=['relu'],
            tail_bn=[False],
            tail_activation=['none'],
        ),
        decoder_kwargs=params_to_dict(
            linear_stack_bn=[False],
            linear_stack_activation=['relu'],
            tail_bn=[False],
            tail_activation=['sigmoid'],
        )
    )
    param_grid = param_grid_full(param_grid)

    df = DF({
        'params': list(param_grid)
    })
    df.to_csv('./params.csv')

    for param_idx, params in enumerate(param_grid):
        pprint(param_idx, params)

        ae = model(**params)
        for i in range(100):
            ae.train(full_Xs, full_Ys, epoch=1)
            # ae.save(save_path)
            metric = ae.metric(full_Xs, full_Ys)
            if any([np.isnan(val) for val in metric.values()]):
                print(f'param_idx:{param_idx}, metric is {metric}')
                break
            print(metric)

            codes = []
            for x, y in idxs_labels:
                code = ae.code(x)
                codes += [code]

            plot.scatter_2d(*codes, title=f'param_idx_{param_idx}/aae_latent_space_epoch_{i}.png')
            # for label, code in enumerate(codes):
            #     plot.scatter_2d(code, title=f'param_idx_{param_idx}/aae_latent_space_epoch_{i}_label+{label}.png')

            recon = ae.recon(sample_Xs, sample_Ys)
            gen = ae.generate(full_Ys[:30])
            code_walk = np.concatenate([ae.augmentation(sample_Xs, sample_Ys) for _ in range(5)], axis=0)
            recon_sharpen = ae.recon_sharpen(sample_Xs, sample_Ys)
            np_img = np.concatenate([sample_Xs, recon, recon_sharpen, gen, code_walk])
            np_img = np_img_float32_to_uint8(np_img)

            file_name = f'./matplot/param_idx_{param_idx}/aae_img_epoch_{i}.png'
            plot.plot_image_tile(np_img, file_name, column=5)
            # sample_imgs = Xs_gen

        del ae
