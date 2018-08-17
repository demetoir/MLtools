import numpy as np
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.AE.CVAE import CVAE
from script.sklearn_like_toolkit.param_search.param_grid import param_grid_full
from script.util.PlotTools import PlotTools
from script.util.misc_util import params_to_dict
from script.util.numpy_utils import np_img_float32_to_uint8, np_img_to_tile, np_image_save
from script.workbench.bench_code import print, DF, pprint

class_ = CVAE
data_pack = DatasetPackLoader().load_dataset("MNIST")
train_set = data_pack['train']
full_Xs, full_Ys = train_set.full_batch(['Xs', 'Ys'])
sample_Xs = full_Xs[:2]
sample_Ys = full_Ys[:2]


def CVAE_total_execute(model):
    model.train(full_Xs, full_Ys, epoch=1)

    code = model.code(sample_Xs, sample_Ys)
    print("code {code}".format(code=code))

    recon = model.recon(sample_Xs, sample_Ys)
    print("recon {recon}".format(recon=recon))

    loss = model.metric(sample_Xs, sample_Ys)
    print("loss {:.4}".format(np.mean(loss)))

    path = model.save()

    class_ = model.__class__
    del model

    model = class_().load(path)
    print('model reloaded')

    code = model.code(sample_Xs, sample_Ys)
    print("code {code}".format(code=code))

    recon = model.recon(sample_Xs, sample_Ys)
    print("recon {recon}".format(recon=recon))

    loss = model.metric(sample_Xs, sample_Ys)
    print("loss {:.4}".format(np.mean(loss)))

    gen = model.generate(sample_Ys)
    print(gen)


def test_CVAE():
    model = class_()
    CVAE_total_execute(model)


def test_CVAE_with_noise():
    model = CVAE(with_noise=True)
    CVAE_total_execute(model)


def test_CVAE_latent_space():
    x = train_set.Ys_index_label
    idxs_labels = []
    for i in range(10):
        idxs_label = np.where(x == i)
        idxs_labels += [(full_Xs[idxs_label], full_Ys[idxs_label])]

    plot = PlotTools()

    model = CVAE
    params = {
        'loss_type':          'VAE',
        'learning_rate':      0.01,
        'latent_code_size':   2,
        'encoder_net_shapes': (512, 256, 128, 64, 32),
        'encoder_kwargs':     {
            'tail_bn':                 False,
            'tail_activation':         'none',
            'linear_stack_bn':         False,
            'linear_stack_activation': 'lrelu',
        },
        'decoder_net_shapes': (32, 64, 128, 256, 512),
        'decoder_kwargs':     {
            'tail_bn':                 True,
            'tail_activation':         'sigmoid',
            'linear_stack_bn':         True,
            'linear_stack_activation': 'relu'
        },
        'batch_size':         256,
        # 'KL_D_rate': 0.01
    }

    ae = model(**params)
    n_iter = 50
    for i in range(n_iter):
        ae.train(full_Xs, full_Ys, epoch=1)
        metric = ae.metric(full_Xs, full_Ys)
        # ae.save(save_path)
        if np.isnan(metric):
            print(f'metric is {metric}')
            break
        print(metric)

        codes = []
        for x, y in idxs_labels:
            code = ae.code(x, y)
            codes += [code]

        plot.scatter_2d(*codes, title=f'CVAE_latent_space_epoch_{i}.png')
        for label, code in enumerate(codes):
            plot.scatter_2d(code, title=f'vae_latent_space_epoch_{i}_label+{label}.png')

    del ae


def test_CVAE_latent_space_grid_search():
    x = train_set.Ys_index_label
    idxs_labels = []
    for i in range(10):
        idxs_label = np.where(x == i)
        idxs_labels += [(full_Xs[idxs_label], full_Ys[idxs_label])]

    plot = PlotTools()

    model = CVAE
    save_path = './CVAE.pkl'

    BOOLs = [True, False]
    linear_stack_activation = ['sigmoid', 'tanh', 'relu', 'lrelu', 'elu']
    param_grid = params_to_dict(

        latent_code_size=[2],
        learning_rate=[0.005],
        encoder_net_shapes=[(512, 256, 128, 64, 32)],
        decoder_net_shapes=[(32, 64, 128, 256, 512)],
        batch_size=[256],
        # KL_D_rate=[0.01],
        loss_type=['VAE'],
        encoder_kwargs=params_to_dict(
            linear_stack_bn=[False],
            linear_stack_activation=['relu', 'lrelu', 'elu'],
            tail_bn=[False],
            tail_activation=['lrelu', 'none'],
        ),
        decoder_kwargs=params_to_dict(
            linear_stack_bn=[True],
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
        for i in range(10):
            ae.train(full_Xs, full_Ys, epoch=1)
            # ae.save(save_path)
            metric = ae.metric(full_Xs, full_Ys)
            if np.isnan(metric):
                print(f'param_idx:{param_idx}, metric is {metric}')
                break
            print(metric)

            codes = []
            for x, y in idxs_labels:
                code = ae.code(x, y)
                codes += [code]

            plot.scatter_2d(*codes, title=f'param_idx_{param_idx}/vae_latent_space_epoch_{i}.png')
            # for label, code in enumerate(codes):
            #     plot.scatter_2d(code, title=f'param_idx_{param_idx}/vae_latent_space_epoch_{i}_label+{label}.png')

            recon = ae.recon(sample_Xs, sample_Ys)
            gen = ae.generate(sample_Ys)
            np_img = np.concatenate([sample_Xs, recon, gen])
            np_img = np_img_float32_to_uint8(np_img)

            # sample_imgs = Xs_gen
            file_name = f'./matplot/param_idx_{param_idx}/vae_img_epoch_{i}.png'
            tile = np_img_to_tile(np_img, column_size=5)
            np_image_save(tile, file_name)

        del ae
