# https://towardsdatascience.com/feature-subset-selection-6de1f05822b0
import functools
import os
from typing import List

import pandas as pd
from autogluon.tabular import TabularPredictor
from multiprocess.pool import ThreadPool
from utils.common import get_logger

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def rank_sequential_backward_elimination(
        predictor: TabularPredictor, X: pd.DataFrame, y: pd.Series, model_name, current_features: List[str] = None,
        time_limit=None, only_last=False, **_):
    X = X.copy(deep=True)

    if current_features is not None:
        X[X.columns.difference(current_features)] = 0.0
    X["Label"] = y
    logger.info("Computing scores")
    scores = predictor.feature_importance(
        X, features=current_features[-1:] if only_last else current_features, model=model_name,
        time_limit=time_limit, include_confidence_band=True, num_shuffle_sets=10)
    scores.index.name = "ID"
    return scores["importance"].to_dict()


def rank_sequential_forward_selection(
        predictor: TabularPredictor, X: pd.DataFrame, y: pd.Series, model_name, current_features: List[str] = None,
        time_limit=None, n_cpus=None, **_):
    remaining = [x for x in predictor.features(feature_stage='transformed') if x not in current_features]

    scores = {}
    prev_rounded = 0
    with ThreadPool(processes=n_cpus) as pool:
        for x in pool.imap(
                functools.partial(
                    rank_sequential_backward_elimination, predictor, X, y, model_name, time_limit=time_limit,
                    only_last=True),
                [current_features + [f] for f in remaining]):
            scores.update(x)
            perc = round(len(scores) * 100 / len(remaining))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info(f"Computing scores at {perc=}%")
    return scores
