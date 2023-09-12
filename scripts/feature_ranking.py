import argparse
import os
from copy import deepcopy
import random
import sys
from functools import partial
from importlib import import_module
from multiprocessing.pool import ThreadPool

import feature_ranking_algorithms
import pandas as pd
from autogluon.tabular import TabularPredictor
from feature_ranking_algorithms.custom import test_model_on_subset
from utils.common import create_dir, get_logger, set_seed
from dataset_creator import load_optional_partial_csv

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def draw_best(scores, top=True):
    sor = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    comp = sor[0][1] if top else sor[-1][1]
    return random.choice([x[0] for x in sor if x[1] == comp])


def main_rank(args):
    global logger
    set_seed()

    algorithm = getattr(import_module(f"feature_ranking_algorithms.{args.algorithm}"), f"rank_{args.strategy}")

    logger.info("Loading predictor")
    predictor = TabularPredictor.load(
        os.path.join(args.directory, os.pardir, os.pardir, os.pardir),
        verbosity=0, require_version_match=True)
    assert isinstance(predictor, TabularPredictor)

    logger.info("Loading data")

    X = load_optional_partial_csv(os.path.join(args.directory, os.pardir, os.pardir,
                                               os.pardir, os.pardir, "finetune.csv"), index_col="ID")
    y = X.pop("Label")

    model_name = os.path.join(
        os.path.basename(os.path.realpath(os.path.join(args.directory, os.pardir))),
        os.path.basename(os.path.realpath(args.directory)))

    logger.info(f"Persist model {model_name=}")
    predictor.persist_models(models=[model_name])

    store_path = os.path.join(predictor._trainer.load_model(
        model_name=model_name).path, f"feature_ranking_{args.strategy}_{args.algorithm}")
    create_dir(store_path, overwrite=False)

    features_names = predictor.features(feature_stage='transformed')
    features = deepcopy(features_names) if args.strategy == "sequential_backward_elimination" else []
    tb_tested = []
    dff = pd.DataFrame(columns=features_names, index=pd.Index([], name="ID"))
    for i in range(len(features_names), 0, -1):
        logger.info(f"Ranking {i=} models when using {len(features)=} features")
        scores = algorithm(predictor, X, y, model_name, current_features=features,
                           n_cpus=args.cpus, target_metric=args.metric, logger=logger, time_limit=args.time_limit)

        dff.loc[len(scores)] = scores

        if args.strategy == "sequential_forward_selection":
            worst_or_best = draw_best(scores, top=True)
            features.append(worst_or_best)
            logger.info(f"Adding {worst_or_best=} to the current best features")

        tb_tested.append(deepcopy(features))

        if args.strategy == "sequential_backward_elimination":
            worst_or_best = draw_best(scores, top=False)
            features.remove(worst_or_best)
            logger.info(f"Removing {worst_or_best=} from the current features")
        break
    dff.to_csv(os.path.join(
        store_path, f'feature_importance.csv'))

    scores = []
    prev_rounded = 0
    logger.info(f"Launching {len(tb_tested)=} test tasks")
    with ThreadPool(processes=args.cpus) as pool:
        for i, x in enumerate(
            pool.imap(partial(test_model_on_subset, predictor, X, y, model_name),
                      tb_tested)):
            scores.append(pd.DataFrame(x, index=pd.Index([len(tb_tested[i])], name="ID")))
            perc = round(i * 100 / len(tb_tested))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info(f"Testing at {perc=}%")

    res = pd.concat(scores).sort_index(ascending=False)
    top = res.iloc[0][args.metric]
    res["degradation_full_baseline"] = (top - res[args.metric]) / top
    res.to_csv(os.path.join(store_path, 'leaderboard.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'directory', help='path to the dataset directory', type=str)
    parser.add_argument('algorithm', help='algorithm to be used', type=str,
                        choices=[x.replace(".py", "")
                                 for x in os.listdir(set(feature_ranking_algorithms.__path__).pop())
                                 if not x.startswith("_")])
    parser.add_argument('strategy', help='algorithm search strategy', type=str, choices=[
        "sequential_backward_elimination", "sequential_forward_selection"])
    parser.add_argument(
        '-c', '--cpus', help='number of cpu cores to assign', type=int, default=round(0.7 * os.cpu_count()))
    parser.add_argument(
        '-t', '--time-limit', help='time limit for autogluon search', type=int, default=None)
    parser.add_argument(
        '-m', '--metric', help='metric to be used', type=str, default='accuracy')
    args = parser.parse_args()
    args.directory = os.path.realpath(os.path.normpath(args.directory))
    getattr(sys.modules[__name__], f"main_{args.action}")(args)
