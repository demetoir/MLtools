import numpy as np
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.data_handler.MNIST import MNIST
from script.model.sklearn_like_model.AE.VAE import VAE
from script.sklearn_like_toolkit.param_grid import param_grid_full
from script.util.PlotTools import PlotTools
from script.util.misc_util import params_to_dict
from script.workbench.bench_code import DF, pprint, print

class_ = VAE
data_pack = DatasetPackLoader().load_dataset("MNIST")
dataset = data_pack['train']
Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
sample_X = Xs[:2]


def VAE_total_execute(model):
    model.train(Xs, epoch=1)

    metric = model.metric(sample_X)
    print(metric)

    code = model.code(sample_X)
    print(code)

    recon = model.recon(sample_X)
    print(recon)

    path = model.save()

    model = class_()
    model.load(path)

    model.train(Xs, epoch=1)

    metric = model.metric(sample_X)
    print(metric)

    metric = model.metric(sample_X)
    print(metric)

    code = model.code(sample_X)
    print(code)

    recon = model.recon(sample_X)
    print(recon)

    model.save()


def test_VAE():
    model = class_()
    VAE_total_execute(model)


def test_VAE_with_noise():
    model = class_(with_noise=True)
    VAE_total_execute(model)


def test_VAE_latent_space_grid_search(n_iter=6):
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\MNIST"""
    dataset_pack = MNIST().load(dataset_path)
    dataset_pack.shuffle()
    train_set = dataset_pack['train']
    full_Xs, full_Ys = train_set.full_batch()

    x = train_set.Ys_index_label
    idxs_labels = []
    for i in range(10):
        idxs_label = np.where(x == i)
        idxs_labels += [(full_Xs[idxs_label], full_Ys[idxs_label])]

    plot = PlotTools()

    model = VAE
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
        for i in range(50):
            ae.train(full_Xs, epoch=1)
            # ae.save(save_path)
            metric = ae.metric(full_Xs)
            if np.isnan(metric):
                print(f'param_idx:{param_idx}, metric is {metric}')
                break
            print(metric)

            codes = []
            for x, y in idxs_labels:
                code = ae.code(x)
                codes += [code]

            plot.scatter_2d(*codes, title=f'param_idx_{param_idx}_vae_latent_space_epoch_{i}.png')

        del ae


def test_VAE_latent_space(n_iter=100):
    dataset_path = """C:\\Users\\demetoir_desktop\\PycharmProjects\\MLtools\\data\\MNIST"""
    dataset_pack = MNIST().load(dataset_path)
    dataset_pack.shuffle()
    train_set = dataset_pack['train']
    full_Xs, full_Ys = train_set.full_batch()

    x = train_set.Ys_index_label
    idxs_labels = []
    for i in range(10):
        idxs_label = np.where(x == i)
        idxs_labels += [(full_Xs[idxs_label], full_Ys[idxs_label])]

    plot = PlotTools()

    model = VAE
    params = {
        'loss_type':          'VAE',
        'learning_rate':      0.01,
        'latent_code_size':   2,
        'encoder_net_shapes': (512, 256, 128, 64, 32),
        'encoder_kwargs':     {
            'tail_bn':                 False,
            'tail_activation':         'lrelu',
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
    for i in range(n_iter):
        ae.train(full_Xs, epoch=1)
        # ae.save(save_path)
        metric = ae.metric(full_Xs)
        if np.isnan(metric):
            print(f'metric is {metric}')
            break
        print(metric)

        codes = [ae.code(x) for x, y in idxs_labels]
        plot.scatter_2d(*codes, title=f'vae_latent_space_epoch_{i}.png')

    del ae
