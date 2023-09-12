"""
Main file to run autoML with HOP and NAS techniques
"""
import sys
import argparse
import os
import pandas as pd

from autogluon.common import space
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularPredictor
from utils.common import create_dir, set_seed, get_logger
from dataset_creator import load_optional_partial_csv
from model_pruner_search import network_layers_sparsity, get_prunable, get_model_sizes

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")

HYPERPARAMETERS_TUNE_ARGS = {
    'num_trials': None,
    'scheduler': 'local',
    'searcher': 'bayes',
}

HYPERPARAMETERS = {
    'NN_TORCH': {
        # https://github.com/autogluon/autogluon/blob/80c0ac6ec52248bcecd51841307a123c99736a59/tabular/src/autogluon/tabular/models/tabular_nn/hyperparameters/parameters.py#L25
        # using embed_min_categories there should not be embedding
        "proc.embed_min_categories": 1000,
        "embedding_size_factor": 1.0,
        "embed_exponent": 0.56,
        "max_embedding_dim": 100,
        'use_ngram_features': False,
        "proc.skew_threshold": 100,
        "proc.max_category_levels": 10000,
        "proc.impute_strategy": "median",
        "weight_decay": 1e-6,
        "y_range_extend": 0,
        "y_range": space.Categorical(0, 1),
        'use_batchnorm': space.Categorical(True, False),
        "optimizer": space.Categorical("adam", "sgd"),
        'activation': space.Categorical('relu', 'tanh'),
        'learning_rate': space.Real(1e-5, 1e-1, default=5e-4, log=True),
        'dropout_prob': space.Real(0.0, 0.8, default=0.2),
        'num_layers': space.Int(1, 100, default=3),
        'hidden_size': space.Int(1, 100, default=16),
        'batch_size': space.Int(1, 512, default=256),
        'max_batch_size': 512,
        'loss_function': "auto",
        'num_epochs': None,
        'epochs_wo_improve': None
    }
}


def main_dump(args):
    global logger
    set_seed()

    predictor = TabularPredictor.load(
        args.directory, verbosity=0, require_version_match=True)

    assert isinstance(predictor, TabularPredictor)

    predictor.fit_summary(verbosity=2)
    predictor.leaderboard(extra_info=True, silent=True).to_csv(
        os.path.join(predictor.path, 'leaderboard.csv'))

    pd.DataFrame(
        {"active": [1 if x in predictor.features(feature_stage='transformed') else 0 for x in predictor.original_features]},
        index=pd.Index(predictor.original_features, name="ID")).to_csv(
        os.path.join(predictor.path, 'features.csv'))

    model = predictor._trainer.load_model(args.model_name if args.model_name !=
                                          "best_model" else predictor.get_model_best())

    logger.info("Saving model info")
    model.save_info()
    layers = get_prunable(model.model)
    glob_sparsity, local_sparsities = network_layers_sparsity(layers)
    sizes = get_model_sizes(model.model)
    tmp = pd.DataFrame({
        **{k: v for k, v in zip(["param_size", "param_size_pruned_neurons", "param_size_pruned_connections"], sizes)},
        "global_sparsity": glob_sparsity,
        **{k: v for k, v in zip([f"sparsityOf_{i}_{str(x)}" for i, x in enumerate(layers)], local_sparsities)}
    }, index=pd.Index([0], name="ID"))
    tmp.to_csv(
        os.path.join(model.path, 'model_stats.csv'))

    logger.info(f"Writing architecture of best model")
    with open(os.path.join(model.path, 'model.log'), 'w') as f:
        f.write(str(model.model))

    df = pd.DataFrame(index=pd.Index([], name="ID"))
    for f in ["train", "validation", "finetune", "test"]:
        logger.info(f"Testing model {model.name} on {f} set")
        X = load_optional_partial_csv(os.path.join(args.directory, os.pardir, f"{f}.csv"), index_col="ID")
        Y = X.pop("Label")
        df = pd.concat(
            (df, pd.DataFrame(predictor.evaluate_predictions(
                Y, pd.Series(model.predict(X)),
                auxiliary_metrics=True, silent=True), index=pd.Index([f], name="ID"))))
    df.to_csv(
        os.path.join(model.path, 'leaderboard.csv'))

    if args.model_name == "best_model":
        os.chdir(predictor._trainer.path)
        if not os.path.exists(args.model_name):
            logger.info("Creating symlink")
            os.symlink(model.name, 'best_model', target_is_directory=True)


def main_search(args):
    """
    Main method to parse arguments and run the autoML whether locally
    or in a remote cluster.
    """
    global HYPERPARAMETERS, HYPERPARAMETERS_TUNE_ARGS
    set_seed()

    if args.cpu == -1:
        args.cpu = 'auto'
    if args.gpu == -1:
        args.gpu = 'auto'

    os.environ['RAY_ADDRESS'] = 'local'
    os.environ.pop('AG_DISTRIBUTED_MODE', None)

    HYPERPARAMETERS_TUNE_ARGS["num_trials"] = args.attempts
    HYPERPARAMETERS['NN_TORCH']['num_epochs'] = args.epochs
    HYPERPARAMETERS['NN_TORCH']['epochs_wo_improve'] = args.patience

    train_data = load_optional_partial_csv(os.path.join(args.directory, "train.csv"), index_col="ID")
    validation_data = load_optional_partial_csv(os.path.join(args.directory, "validation.csv"), index_col="ID")

    create_dir(os.path.join(args.directory, 'automl_search'), overwrite=False)

    predictor = TabularPredictor(
        label='Label', eval_metric=args.metric,
        problem_type='binary',
        verbosity=2, log_to_file=False,
        path=os.path.join(args.directory, 'automl_search'))
    predictor.fit(
        train_data,
        tuning_data=validation_data,
        fit_weighted_ensemble=False,
        time_limit=args.time_limit,
        num_cpus=args.cpu,
        num_gpus=args.gpu,
        hyperparameter_tune_kwargs=HYPERPARAMETERS_TUNE_ARGS,
        hyperparameters=HYPERPARAMETERS,
        feature_generator=AutoMLPipelineFeatureGenerator(),
        feature_prune_kwargs=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(dest='action', help="Main action")

    dumper = subparsers.add_parser('dump', help='Actions for dumping results')
    dumper.add_argument(
        'directory', help='path to the dataset directory', type=str)
    dumper.add_argument(
        '-m', '--model_name', help='name of the model to be dump', type=str, default="best_model")

    searcher = subparsers.add_parser('search', help='Actions for Searching models')
    searcher.add_argument(
        'directory', help='path to the dataset directory', type=str)
    searcher.add_argument(
        '-e', '--epochs', help='training epochs', type=int, default=1000)
    searcher.add_argument(
        '-p', '--patience', help='patience', type=int, default=20)
    searcher.add_argument(
        '-c', '--cpu', help='number of cpu cores to assign', type=int, default=round(0.7 * os.cpu_count()))
    searcher.add_argument(
        '-g', '--gpu', help='number of gpu, if any available', type=int, default=0)
    searcher.add_argument(
        '-a', '--attempts', help='HPO trials', type=int, default=1000)
    searcher.add_argument(
        '-t', '--time-limit', help='limit time for the running', type=int, default=60 * 60 * 1)
    searcher.add_argument(
        '-m', '--metric', help='metric to be plot', type=str, default='accuracy')
    args = parser.parse_args()
    args.directory = os.path.realpath(os.path.normpath(args.directory))
    getattr(sys.modules[__name__], f"main_{args.action}")(args)
