"""
Script to create a new network given a sample one, but pruning
few layers/neurons.
https://towardsdatascience.com/pruning-neural-networks-1bb3ab5791f9
"""
import argparse
import ast
from logging import WARNING
import functools
import inspect
import os
import pickle
import sys
from copy import deepcopy
from importlib import import_module
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import \
    TabularNeuralNetTorchModel
from dataset_creator import load_optional_partial_csv
from feature_ranking_algorithms.custom import test_model_on_subset
from network_pruning_algorithms import globally, locally
from torch import nn
from utils.common import create_dir, get_logger, set_seed

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def get_prunable(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def get_model_sizes(model: nn.Module):
    status = {}
    for name, param in model.named_parameters():
        total_size_if_not_pruned = param.nelement() * param.element_size()
        dims = list(param.size())
        n_elem_for_single = dims[0] if len(dims) == 1 else dims[1]

        tensor_counting_many_zeros = param.count_nonzero(dim=-1)
        total_size_if_pruned = tensor_counting_many_zeros.count_nonzero().item() * param.element_size() * \
            n_elem_for_single

        n_elem_for_single_links = dims[0] if len(dims) == 1 else dims[0]
        tensor_counting_many_zeros_links = param.count_nonzero(dim=0)
        total_size_if_pruned_links = tensor_counting_many_zeros_links.count_nonzero().item() * param.element_size() * \
            n_elem_for_single_links

        status[name] = [total_size_if_not_pruned, total_size_if_pruned, total_size_if_pruned_links]
    return tuple([round(sum(v[i] for v in status.values())) for i in range(3)])


def network_layers_sparsity(refs: nn.Module):
    total = float(100. * sum(torch.sum(x.weight == 0)
                  for x in refs) / sum(x.weight.nelement() for x in refs)) +\
        float(100. * sum(torch.sum(x.bias == 0)
                         for x in refs) / sum(x.bias.nelement() for x in refs))
    single = [float(100. * torch.sum(x.weight == 0) / x.weight.nelement()) +
              float(100. * torch.sum(x.bias == 0) / x.bias.nelement()) for x in refs]
    return total, single


def handler(predictor, algorithm, original_model, X, y, amount, subset_features=[]):
    global logger
    if isinstance(amount, list):
        subset_features = ast.literal_eval(amount[1])
        amount = amount[0]
    copied = deepcopy(original_model)
    layers = get_prunable(copied.model)
    copied, layers = algorithm(copied, layers, amount, X=X, subset_features=subset_features)
    sizes = get_model_sizes(copied.model)
    glob_sparsity, local_sparsity = network_layers_sparsity(layers)
    test_results = test_model_on_subset(predictor, X, y, copied, group=None)
    return copied, amount, sizes, glob_sparsity, layers, local_sparsity, test_results


def _pickle_load_model(path, index):
    with open(os.path.join(path, f"{index}.pkl"), 'rb') as fp:
        return pickle.load(fp)


def _pickle_dump_model(args):
    path, model = args
    with open(path, 'wb') as fp:
        return pickle.dump(model, fp)


def handle_load_features(nrows, filepath):
    tops = load_optional_partial_csv(os.path.join(filepath, "leaderboard.csv"), index_col="ID",
                                     dest=list(range(nrows + 1)))
    sets_info = load_optional_partial_csv(os.path.join(filepath, "feature_sets.csv"),
                                          index_col="ID").loc[tops.index.values]
    return sets_info.index.values, sets_info["features"].tolist()


def main_test(args):
    global logger
    set_seed()

    logger.info("Loading Predictor")
    predictor_path = os.path.realpath(os.path.join(
        args.directory, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir))
    predictor = TabularPredictor.load(predictor_path,
                                      verbosity=0, require_version_match=True)
    assert isinstance(predictor, TabularPredictor)

    logger.info("Loading Data")

    X = load_optional_partial_csv(
        os.path.realpath(os.path.join(args.directory, os.pardir, os.pardir, os.pardir,
                                      os.pardir, os.pardir, os.pardir, os.pardir, "finetune.csv")),
        index_col="ID")
    y = X.pop("Label")

    logger.info("Loading Subsets")
    subsets_file = load_optional_partial_csv(
        os.path.join(args.directory, "subsets.csv"),
        index_col="ID")
    subset_leaderboard = load_optional_partial_csv(
        os.path.join(args.directory, "leaderboard.csv"),
        index_col="ID")

    if "_for_subset" in args.algorithm:
        prune_dir = os.path.join(args.directory, args.algorithm)
    else:
        prune_dir = os.path.join(args.directory, os.pardir, os.pardir, os.pardir,
                                 "prune_search", args.algorithm)
    valid_subsets_in_file = subsets_file.loc[subset_leaderboard[subset_leaderboard["degradation_at_rank"]
                                                                <= args.degradation].index.values[:args.top_subsets]]

    logger.info("Loading Models' stats")
    models_file = load_optional_partial_csv(
        os.path.join(prune_dir, "models_stats.csv"),
        index_col="ID")
    models_leaderboard = load_optional_partial_csv(
        os.path.join(prune_dir, "leaderboard.csv"),
        index_col="ID")
    valid_models_in_file = models_file
    if args.algorithm.startswith("locally_"):
        valid_models_in_file = models_file.loc[models_leaderboard[models_leaderboard["degradation_full_baseline"]
                                                                  <= args.degradation].index.values][:args.top_subsets]

    if "_for_subset" in args.algorithm:
        combs = iter(
            (prune_dir, i,
             valid_subsets_in_file.loc[int(valid_models_in_file.loc[i]["subset_ID"])]["features"])
            for i in valid_models_in_file.index.values)
        ids = [(i, int(valid_models_in_file.loc[i]["subset_ID"])) for i in valid_models_in_file.index.values]
    else:
        combs = iter((prune_dir, i, f)
                     for i in valid_models_in_file.index.values for f in valid_subsets_in_file["features"])
        ids = [(i, ii) for i in valid_models_in_file.index.values for ii in valid_subsets_in_file.index.values]
    results = []
    prev_rounded = 0
    logger.info(f"Testing {len(ids)=} combinations of subset-model asyncronously")
    with ThreadPool(args.cpu) as pool:
        for i, res in enumerate(
            pool.imap(functools.partial(semi_handler, predictor, X, y),
                      combs)):
            results.append(
                pd.DataFrame(
                    {"model_ID": ids[i][0],
                     "subset_ID": ids[i][1],
                     **res},
                    index=pd.Index([i],
                                   name="ID")))
            perc = round(i * 100 / len(ids))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info(f"Testing combinations at {perc=}%")
    results = pd.concat(results).sort_values(by=args.metric, ascending=False)
    base_metric = load_optional_partial_csv(
        os.path.join(args.directory, os.pardir, os.pardir, os.pardir, "leaderboard.csv"),
        index_col="ID").loc["finetune"][args.metric]
    results["degradation_full_baseline"] = (base_metric - results[args.metric]) / base_metric

    size = float(os.path.basename(args.directory).replace("feature_subsets_", "").replace("s", ""))
    base_metric_at_rank = load_optional_partial_csv(
        os.path.join(args.directory, os.pardir, os.pardir, "leaderboard.csv"), index_col="ID").loc[
        round(size * len(predictor.features(feature_stage="transformed"))) or 1][args.metric]
    results["degradation_at_rank"] = (
        base_metric_at_rank - results[args.metric]) / base_metric_at_rank

    tmp = [subset_leaderboard.loc[x]["accuracy"] for x in results["subset_ID"].values]
    results["degradation_at_subset"] = (tmp - results["accuracy"]) / tmp

    tmp = [models_leaderboard.loc[x]["accuracy"] for x in results["model_ID"].values]
    results["degradation_at_model"] = (tmp - results["accuracy"]) / tmp

    results.to_csv(
        os.path.join(args.directory, f"leaderboard_{args.algorithm}.csv"))


def semi_handler(predictor, X, y, args):
    (path, index, features) = args
    model = _pickle_load_model(path, index)
    res = test_model_on_subset(predictor, X, y, model, group=ast.literal_eval(features))
    del model
    return res


def main_search(args):
    global logger
    set_seed()

    logger.info("Loading Predictor")
    predictor_path = os.path.realpath(os.path.join(args.directory, os.pardir, os.pardir, os.pardir))
    predictor = TabularPredictor.load(predictor_path,
                                      verbosity=0, require_version_match=True)
    assert isinstance(predictor, TabularPredictor)

    model_name = os.path.join(
        os.path.basename(os.path.realpath(os.path.join(args.directory, os.pardir))),
        os.path.basename(os.path.realpath(os.path.join(args.directory))))

    logger.info(f"Loading {model_name=} Model")
    original_model: TabularNeuralNetTorchModel = predictor._trainer.load_model(model_name)

    logger.info("Loading Data")

    X = load_optional_partial_csv(
        os.path.realpath(os.path.join(args.directory, os.pardir, os.pardir, os.pardir,
                                      os.pardir, "finetune.csv")),
        index_col="ID")
    y = X.pop("Label")

    from network_pruning_algorithms.globally import _globally_neurons

    store_path = os.path.join(original_model.path, "prune_search", args.algorithm)
    subsets: pd.DataFrame = None

    if args.subsets:
        if "_for_subset" not in args.algorithm:
            raise ValueError("Used subset search when providing a wrong algorithm")
        store_path = os.path.join(args.subsets, args.algorithm)
        logger.info("Loading provided subsets")
        subsets = load_optional_partial_csv(
            os.path.join(args.subsets, "leaderboard.csv"),
            index_col="ID")
        valid_subsets = subsets[subsets["degradation_at_rank"] <= args.degradation].index.values[:args.top_subsets]
        subsets = load_optional_partial_csv(
            os.path.join(args.subsets, "subsets.csv"),
            index_col="ID").loc[valid_subsets]

    create_dir(store_path, overwrite=False)
    mod = import_module(f"network_pruning_algorithms.{args.algorithm.split('_')[0]}")
    locally.logger.setLevel(WARNING)
    globally.logger.setLevel(WARNING)
    algorithm = getattr(mod, args.algorithm)

    args.prune_amount = sorted(args.prune_amount, reverse=True)

    if args.algorithm.startswith("locally"):
        if 0. not in args.prune_amount:
            args.prune_amount = args.prune_amount + [0.]
        if 1. not in args.prune_amount:
            args.prune_amount = [1.] + args.prune_amount
        n_layers = len(get_prunable(original_model.model))
        all_combs = {}
        wanted = min(args.attempts, len(args.prune_amount)**n_layers)
        cnt = 0
        logger.info(f"Computing combination {wanted=} {n_layers=}")
        probs = np.linspace(1, len(args.prune_amount), len(args.prune_amount))
        probs = probs / probs.sum()
        while len(all_combs) < wanted:
            news = tuple(np.random.choice(
                args.prune_amount, n_layers, p=probs, replace=True).tolist())
            v = len(all_combs)
            all_combs.setdefault(news, 0)
            cnt = cnt + 1 if v == len(all_combs) else 0
            if cnt == 1000:
                break
        args.prune_amount = list(all_combs.keys())
    elif args.algorithm.startswith("globally"):
        if 0. in args.prune_amount:
            args.prune_amount.remove(0.)
        if 1. in args.prune_amount:
            args.prune_amount.remove(1.)
        args.prune_amount = sorted(np.random.choice(args.prune_amount, min(
            len(args.prune_amount), args.attempts), replace=False), reverse=True)

    if args.subsets:
        logger.info("Associating Prune amount to each subset")
        args.prune_amount = args.prune_amount[:min(args.attempts, round(args.attempts / args.top_subsets))]
        prev_len = len(args.prune_amount)
        args.prune_amount = [[k, v] for k in args.prune_amount for v in subsets["features"].to_list()]
        subsets = [i for _ in range(prev_len) for i in subsets.index.values]

    models = []
    stats = []
    results = []
    prev_rounded = 0
    with ThreadPool(processes=args.cpu) as pool:
        logger.info(f"Running {len(args.prune_amount)=} pruning tasks")
        for i, (model, amount, sizes, glob_sparsity, layers, local_sparsity, test_res) in enumerate(
            pool.imap(functools.partial(handler, predictor, algorithm, original_model, X, y),
                      args.prune_amount)):
            models.append(model)
            stats.append(pd.DataFrame({
                **({"subset_ID": subsets[i]} if subsets else {}),
                "amount": str(amount),
                **{k: v for k, v in zip(["param_size", "param_size_pruned_neurons", "param_size_pruned_connections"], sizes)},
                "global_sparsity": glob_sparsity,
                **{k: v for k, v in zip([f"sparsityOf_{i}_{str(x)}" for i, x in enumerate(layers)], local_sparsity)}
            }, index=pd.Index([i], name="ID")))

            results.append(pd.DataFrame(test_res, index=pd.Index([i], name="ID")))
            perc = round(i * 100 / len(args.prune_amount))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info(f"Pruning Search spaces at {perc=}%")

        logger.info("Saving models stats")
        pd.concat([v for v in stats]).to_csv(os.path.join(store_path, "models_stats.csv"))

        results = pd.concat(results).sort_values(
            by=args.metric, ascending=False)

        base_metric = load_optional_partial_csv(
            os.path.join(args.directory, "leaderboard.csv"), index_col="ID").loc["finetune"][args.metric]
        results["degradation_full_baseline"] = (base_metric - results[args.metric]) / base_metric
        results.to_csv(
            os.path.join(store_path, "leaderboard.csv"))

        valid_rows = results.index.values
        if args.algorithm.startswith("locally"):
            valid_rows = results.loc[results["degradation_full_baseline"] <= args.degradation].index.values
        paths = [os.path.join(store_path, f"{i}.pkl") for i in valid_rows]
        keys = [models[i] for i in valid_rows]
        prev_rounded = 0
        logger.info(f"Running dump of {len(valid_rows)=} models")
        for i, _ in enumerate(
                pool.imap(_pickle_dump_model, zip(paths, keys))):
            perc = round(i * 100 / len(keys))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info(f"Dumping at {perc=}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    methods = [x[0] for x in inspect.getmembers(globally, inspect.isfunction) + inspect.getmembers(
        locally, inspect.isfunction) if x[0].startswith("globally_") or x[0].startswith("locally_")]

    subparsers = parser.add_subparsers(dest='action', help="Main action")

    searcher = subparsers.add_parser('search', help='Actions for searching')
    searcher.add_argument(
        'directory', help='path to a result of feature_subset_stochastic_search', type=str)
    searcher.add_argument('algorithm', help='prune method name', type=str, choices=methods)
    searcher.add_argument(
        '-s', '--subsets', help='search activation using these subsets', type=str, nargs="?", default=None)
    searcher.add_argument(
        '-c', '--cpu', help='number of cpu cores to assign', type=int, default=round(0.7 * os.cpu_count()))
    searcher.add_argument(
        '-p', '--prune-amount', help='amount to be pruned', nargs="+", type=float,
        default=np.round(np.linspace(0., 1., 21, endpoint=True), 2).tolist())
    searcher.add_argument(
        '-a', '--attempts', help='trials in case of non-global search', type=int, default=1000)
    searcher.add_argument(
        '-t', '--top_subsets', help='trials in case of non-global search', type=int, default=100)
    searcher.add_argument(
        '-d', '--degradation', help='degradation accepted', type=float, default=0.10)
    searcher.add_argument(
        '-m', '--metric', help='metric to be plot', type=str, default='accuracy')

    tester = subparsers.add_parser('test', help='Actions for testing')
    tester.add_argument(
        'directory', help='path to a result of feature_subset_stochastic_search', type=str)
    tester.add_argument('algorithm', help='prune method name', type=str, choices=methods)
    tester.add_argument(
        '-c', '--cpu', help='number of cpu cores to assign', type=int, default=round(0.7 * os.cpu_count()))
    tester.add_argument(
        '-d', '--degradation', help='degradation accepted', type=float, default=0.10)
    tester.add_argument(
        '-t', '--top_subsets', help='trials in case of non-global search', type=int, default=100)
    tester.add_argument(
        '-m', '--metric', help='metric to be plot', type=str, default='accuracy')

    args = parser.parse_args()
    args.directory = os.path.realpath(os.path.normpath(args.directory))

    getattr(sys.modules[__name__], f"main_{args.action}")(args)
