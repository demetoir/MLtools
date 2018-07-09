import warnings
from catboost import CatBoostClassifier as _CatBoostClassifier
from catboost import CatBoostRegressor as _CatBoostRegressor
from script.sklearn_like_toolkit.base.BaseWrapperClf import BaseWrapperClf
from script.sklearn_like_toolkit.base.MixIn import meta_BaseWrapperClf_with_ABC, meta_BaseWrapperReg_with_ABC
from script.sklearn_like_toolkit.base.BaseWrapperReg import BaseWrapperReg
from hyperopt import hp


class CatBoostClf(BaseWrapperClf, _CatBoostClassifier, metaclass=meta_BaseWrapperClf_with_ABC):
    HyperOpt_space = {
        'iterations': 2 + hp.randint('iterations', 10),
        'depth': 4 + hp.randint('depth', 11),
        'random_strength': hp.choice('random_strength', [1, 2, 4, 0.5, ]),
        'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
        'learning_rate': hp.loguniform('learning_rate', -6, 0),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0, 1),
    }
    tuning_grid = {
        'iterations': [2, 4, 8, ],
        'depth': [i for i in range(4, 10 + 1, 2)],
        # 'random_strength': [1, 2, 4, 0.5, ],
        'bagging_temperature': [i / 100.0 for i in range(1, 10 + 1, 3)],
        'learning_rate': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'l2_leaf_reg': [i / 10.0 for i in range(1, 10 + 1, 3)],

    }
    remain_param = {
        'use_best_model': [True, False],
        'eval_metric': [],
        'od_type': None,
        'od_pval': None,
        'od_wait': None
    }

    def __init__(
            self, iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None, model_size_reg=None, rsm=None,
            loss_function='Logloss', border_count=None, feature_border_type=None, fold_permutation_block_size=None,
            od_pval=None, od_wait=None, od_type=None, nan_mode=None, counter_calc_method=None,
            leaf_estimation_iterations=None, leaf_estimation_method=None, thread_count=None, random_seed=None,
            use_best_model=None, verbose=None, silent=None, logging_level=None, metric_period=None,
            ctr_leaf_count_limit=None, store_all_simple_ctr=None, max_ctr_complexity=None, has_time=None,
            allow_const_label=None, classes_count=None, class_weights=None, one_hot_max_size=None,
            random_strength=None, name=None, ignored_features=None, train_dir=None, custom_loss=None,
            custom_metric=None, eval_metric=None, bagging_temperature=None, save_snapshot=None, snapshot_file=None,
            fold_len_multiplier=None, used_ram_limit=None, gpu_ram_part=None, pinned_memory_size=None,
            allow_writing_files=None, final_ctr_computation_mode=None, approx_on_full_history=None,
            boosting_type=None, simple_ctr=None, combinations_ctr=None, per_feature_ctr=None, ctr_description=None,
            task_type=None, device_config=None, devices=None, bootstrap_type=None, subsample=None, max_depth=None,
            n_estimators=None, num_boost_round=None, num_trees=None, colsample_bylevel=None, random_state=None,
            reg_lambda=None, objective=None, eta=None, max_bin=None, scale_pos_weight=None,
            gpu_cat_features_storage=None, data_partition=None, metadata=None):
        # logging_level = 'Silent'
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
        silent = True
        BaseWrapperClf.__init__(self)

        _CatBoostClassifier.__init__(
            self, iterations, learning_rate, depth, l2_leaf_reg, model_size_reg, rsm, loss_function,
            border_count, feature_border_type, fold_permutation_block_size, od_pval, od_wait, od_type,
            nan_mode, counter_calc_method, leaf_estimation_iterations, leaf_estimation_method,
            thread_count, random_seed, use_best_model, verbose, silent, logging_level, metric_period,
            ctr_leaf_count_limit, store_all_simple_ctr, max_ctr_complexity, has_time, allow_const_label,
            classes_count, class_weights, one_hot_max_size, random_strength, name, ignored_features,
            train_dir, custom_loss, custom_metric, eval_metric, bagging_temperature, save_snapshot,
            snapshot_file, fold_len_multiplier, used_ram_limit, gpu_ram_part, pinned_memory_size,
            allow_writing_files, final_ctr_computation_mode, approx_on_full_history, boosting_type,
            simple_ctr, combinations_ctr, per_feature_ctr, ctr_description, task_type, device_config,
            devices, bootstrap_type, subsample, max_depth, n_estimators, num_boost_round, num_trees,
            colsample_bylevel, random_state, reg_lambda, objective, eta, max_bin, scale_pos_weight,
            gpu_cat_features_storage, data_partition, metadata)


class CatBoostReg(_CatBoostRegressor, BaseWrapperReg, metaclass=meta_BaseWrapperReg_with_ABC):
    # HyperOpt_space = {
    #     'iterations': hp.randint('iterations', 2, 10),
    #     'depth': hp.randint('depth', 4, 11),
    #     'random_strength': hp.choice('random_strength', [1, 2, 4, 0.5, ]),
    #     'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
    #     'learning_rate': hp.uniform('learning_rate', 0, 1),
    #     'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0, 1),
    # }

    tuning_grid = {
        # 'iterations': [2, 4, 8, ],
        # 'depth': [i for i in range(4, 10 + 1, 2)],
        # # 'random_strength': [1, 2, 4, 0.5, ],
        # 'bagging_temperature': [i / 100.0 for i in range(1, 10 + 1, 3)],
        # 'learning_rate': [i / 10.0 for i in range(1, 10 + 1, 3)],
        # 'l2_leaf_reg': [i / 10.0 for i in range(1, 10 + 1, 3)],

    }
    remain_param = {
        'use_best_model': [True, False],
        'eval_metric': [],
        'od_type': None,
        'od_pval': None,
        'od_wait': None
    }

    def __init__(
            self, iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None, model_size_reg=None, rsm=None,
            loss_function='RMSE', border_count=None, feature_border_type=None, fold_permutation_block_size=None,
            od_pval=None, od_wait=None, od_type=None, nan_mode=None, counter_calc_method=None,
            leaf_estimation_iterations=None, leaf_estimation_method=None, thread_count=None, random_seed=None,
            use_best_model=None, verbose=None, silent=None, logging_level=None, metric_period=None,
            ctr_leaf_count_limit=None, store_all_simple_ctr=None, max_ctr_complexity=None, has_time=None,
            allow_const_label=None, one_hot_max_size=None, random_strength=None, name=None, ignored_features=None,
            train_dir=None, custom_metric=None, eval_metric=None, bagging_temperature=None, save_snapshot=None,
            snapshot_file=None, fold_len_multiplier=None, used_ram_limit=None, gpu_ram_part=None,
            pinned_memory_size=None, allow_writing_files=None, final_ctr_computation_mode=None,
            approx_on_full_history=None, boosting_type=None, simple_ctr=None, combinations_ctr=None,
            per_feature_ctr=None, ctr_description=None, task_type=None, device_config=None, devices=None,
            bootstrap_type=None, subsample=None, max_depth=None, n_estimators=None, num_boost_round=None,
            num_trees=None, colsample_bylevel=None, random_state=None, reg_lambda=None, objective=None, eta=None,
            max_bin=None, gpu_cat_features_storage=None, data_partition=None, metadata=None):
        silent = True
        _CatBoostRegressor.__init__(
            self, iterations, learning_rate, depth, l2_leaf_reg, model_size_reg, rsm, loss_function,
            border_count, feature_border_type, fold_permutation_block_size, od_pval, od_wait, od_type,
            nan_mode, counter_calc_method, leaf_estimation_iterations, leaf_estimation_method,
            thread_count, random_seed, use_best_model, verbose, silent, logging_level, metric_period,
            ctr_leaf_count_limit, store_all_simple_ctr, max_ctr_complexity, has_time, allow_const_label,
            one_hot_max_size, random_strength, name, ignored_features, train_dir, custom_metric,
            eval_metric, bagging_temperature, save_snapshot, snapshot_file, fold_len_multiplier,
            used_ram_limit, gpu_ram_part, pinned_memory_size, allow_writing_files,
            final_ctr_computation_mode, approx_on_full_history, boosting_type, simple_ctr,
            combinations_ctr, per_feature_ctr, ctr_description, task_type, device_config, devices,
            bootstrap_type, subsample, max_depth, n_estimators, num_boost_round, num_trees,
            colsample_bylevel, random_state, reg_lambda, objective, eta, max_bin, gpu_cat_features_storage,
            data_partition, metadata)
        BaseWrapperReg.__init__(self)
