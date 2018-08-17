from pprint import pprint
from script.sklearn_like_toolkit.param_search.param_grid import param_grid_random, param_grid_full
from script.util.misc_util import params_to_dict
from script.util.tensor_ops import activation_names


def test_param_grid_random():
    BOOLs = [True, False]
    encoder_kwargs = params_to_dict(
        linear_stack_bn=BOOLs, linear_stack_activation=activation_names,
        tail_bn=BOOLs, tail_activation=activation_names,
    )
    decoder_kwargs = params_to_dict(
        linear_stack_bn=BOOLs, linear_stack_activation=activation_names,
        tail_bn=BOOLs, tail_activation=activation_names,
    )

    param_grid = params_to_dict(
        latent_code_size=[2],
        learning_rate=[0.01],
        encoder_net_shapes=[(512, 256, 128, 64, 32)],
        decoder_net_shapes=[(32, 64, 128, 256, 512)],
        batch_size=[256],
        KL_D_rate=[0.01],
        # loss_type=VAE.loss_names,
        loss_type=['VAE'],
        encoder_kwargs=encoder_kwargs,
        # decoder_kwargs=decoder_kwargs
    )

    param_grid = param_grid_random(param_grid, 50)

    for grid in param_grid:
        pprint(grid)
    pprint(len(param_grid))


def test_param_grid_full():
    BOOLs = [True, False]
    encoder_kwargs = params_to_dict(
        linear_stack_bn=BOOLs, linear_stack_activation=activation_names,
        tail_bn=BOOLs, tail_activation=activation_names,
    )
    decoder_kwargs = params_to_dict(
        linear_stack_bn=BOOLs, linear_stack_activation=activation_names,
        tail_bn=BOOLs, tail_activation=activation_names,
    )

    param_grid = params_to_dict(
        latent_code_size=[2],
        learning_rate=[0.01],
        encoder_net_shapes=[(512, 256, 128, 64, 32)],
        decoder_net_shapes=[(32, 64, 128, 256, 512)],
        batch_size=[256],
        KL_D_rate=[0.01],
        loss_type=['VAE'],
        encoder_kwargs=encoder_kwargs,
        # decoder_kwargs=decoder_kwargs
    )

    param_grid = param_grid_full(param_grid)

    for grid in param_grid:
        pprint(grid)
    pprint(len(param_grid))
