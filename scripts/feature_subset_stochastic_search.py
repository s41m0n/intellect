import argparse
import ast
import functools
import math
import os
import sys
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from dataset_creator import load_optional_partial_csv
from feature_ranking import test_model_on_subset
from utils.common import create_dir, get_logger, set_seed

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def handler(trials, all_features, store_path, size, best_list, is_weighted):
    global logger
    n_curr_features = round(len(all_features) * size) or 1

    all_ranks = load_optional_partial_csv(
        os.path.join(
            store_path, "feature_importance.csv"),
        index_col="ID")

    # all_ranks.loc[n_curr_features].sort_values(ascending=False)
    file_rank = all_ranks.loc[len(all_features)].sort_values(ascending=False)
    other_rank = all_ranks.loc[len(all_features)].sort_values(ascending=False)

    weights = None

    all_combs = {}
    n_all_combs = math.comb(len(all_features), n_curr_features)
    logger.info(
        f"All possible combinations are {n_all_combs=}"
        f" as {n_curr_features=} and {len(all_features)=}")
    best_list = sorted(best_list, reverse=True)
    for i, b in enumerate(best_list):
        if len(all_combs) == n_all_combs or len(all_combs) == trials:
            break
        n_best = round(b * n_curr_features)
        constant = []
        left_features = all_features
        n_curr_features_tmp = n_curr_features
        if n_best:
            constant = file_rank.index[:n_best].values.tolist()
            left_features = [x for x in all_features if x not in constant]
            n_curr_features_tmp -= len(constant)
        other_rank_tmp = other_rank.drop(index=constant)

        if is_weighted:
            tmp = np.array([x for x in range(len(other_rank_tmp), 0, -1)]) + sys.float_info.epsilon
            weights = (tmp / tmp.sum())

        pre_len = len(all_combs)
        current_max_combination_with_best = math.comb(len(left_features), n_curr_features_tmp)
        n_left_attempt = min(
            max(
                round((trials - pre_len) / (len(best_list) - i)),
                1),
            current_max_combination_with_best)

        cnt = 0
        logger.info(f"{b=} {n_best=} {current_max_combination_with_best=} {pre_len=}"
                    f" {n_left_attempt=} {n_curr_features_tmp=}")
        while len(all_combs) < pre_len + n_left_attempt:
            news = tuple(sorted(np.random.choice(
                left_features, n_curr_features_tmp, p=weights, replace=False).tolist() + constant))
            v = len(all_combs)
            all_combs.setdefault(news, b)
            cnt = cnt + 1 if v == len(all_combs) else 0
            if cnt == 1000:
                break

    return pd.DataFrame(
        {"best": [v for v in all_combs.values()],
         "features": [str(tuple(x)) for x in all_combs]},
        index=pd.Index(range(len(all_combs)),
                       name="ID"))


def main_search(args):
    global logger
    set_seed()

    logger.info("Loading the predictor")
    predictor = TabularPredictor.load(
        os.path.join(args.directory, os.pardir, os.pardir, os.pardir, os.pardir),
        verbosity=0, require_version_match=True)
    assert isinstance(predictor, TabularPredictor)

    model_name = os.path.join(
        os.path.basename(os.path.realpath(os.path.join(args.directory, os.pardir, os.pardir))),
        os.path.basename(os.path.realpath(os.path.join(args.directory, os.pardir))))
    logger.info(f"Persist model {model_name=}")
    predictor.persist_models(models=[model_name])

    logger.info("Loading the data")

    X = load_optional_partial_csv(
        os.path.join(args.directory, os.pardir, os.pardir,
                     os.pardir, os.pardir, os.pardir, "finetune.csv"),
        index_col="ID")
    y = X.pop("Label")

    features = predictor.features(feature_stage='transformed')

    store_path = os.path.join(
        args.directory, f"feature_subset_stochastic_search_{args.search_type}",
        f"feature_subsets_{args.subset_size:.2f}s")
    create_dir(store_path, overwrite=False)

    logger.info("Computing Combinations")
    df: pd.DataFrame = handler(args.attempts, features, args.directory, args.subset_size,
                               args.best_list, args.search_type == 'weighted')
    df.to_csv(os.path.join(store_path, "subsets.csv"))

    subset_list = [ast.literal_eval(x) for x in df["features"].tolist()]
    prev_rounded = 0
    with ThreadPool(processes=args.cpu) as pool:
        results = []
        logger.info("Launching test jobs")
        for i, x in enumerate(
            pool.imap(
                functools.partial(test_model_on_subset, predictor, X, y, model_name),
                subset_list)):
            results.append(pd.DataFrame(x, index=pd.Index([i], name="ID")))
            perc = round(i * 100 / len(subset_list))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info(f"Testing subsets at {perc=}%")
    results = pd.concat(results)
    results.sort_values(by=args.metric, inplace=True, ascending=False)

    tmp = load_optional_partial_csv(
        os.path.join(
            args.directory, "leaderboard.csv"),
        index_col="ID").loc[round(len(features) * args.subset_size) or 1]
    base_metric_at_rank = tmp.loc[args.metric]
    base_metric = base_metric_at_rank / (1 - tmp["degradation_full_baseline"])
    results["degradation_full_baseline"] = (base_metric - results[args.metric]) / base_metric
    results["degradation_at_rank"] = (base_metric_at_rank - results[args.metric]) / base_metric_at_rank
    results.to_csv(os.path.join(store_path, "leaderboard.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'directory', help='path to the dataset directory', type=str)
    parser.add_argument(
        'subset_size', help='size of the subset to look for', type=float)
    parser.add_argument(
        'search_type', help='weighted or random search', type=str, choices=["random", "weighted"])
    parser.add_argument('-b', '--best-list', help='list of best percentages to try', nargs="+",
                        type=float, default=np.round(np.linspace(0., 1., 21, endpoint=True), 2))
    parser.add_argument(
        '-c', '--cpu', help='number of cpu cores to assign', type=int, default=round(0.7 * os.cpu_count()))
    parser.add_argument(
        '-a', '--attempts', help='maximum trials', type=int, default=1000)
    parser.add_argument(
        '-m', '--metric', help='metric to be plot', type=str, default='accuracy')
    args = parser.parse_args()
    args.directory = os.path.realpath(os.path.normpath(args.directory))
    getattr(sys.modules[__name__], f"main_{args.action}")(args)
