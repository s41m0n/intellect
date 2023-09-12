import functools
import os
from multiprocessing.pool import ThreadPool
from typing import List
from autogluon.tabular import TabularPredictor

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils.common import get_logger

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def rank_sequential_backward_elimination(
        predictor: TabularPredictor, X: pd.DataFrame, y: pd.Series, __, current_features: List[str] = None,
        only_last=False, n_cpus=None, **___):
    X = X.copy(deep=True)

    if current_features is None:
        current_features = predictor.features(feature_stage='transformed')

    X[X.columns.difference(current_features)] = 0.0

    positions = [i for i, x in enumerate(predictor.features(feature_stage='transformed')) if x in current_features]

    rf = RandomForestClassifier(n_estimators=100, max_depth=100, max_features='sqrt', criterion='gini', n_jobs=n_cpus)
    rf.fit(X, y)

    if only_last:
        return {current_features[-1]: rf.feature_importances_[-1]}
    return dict(zip(current_features, [rf.feature_importances_[i] for i in positions]))


def rank_sequential_forward_selection(
        predictor: TabularPredictor, X: pd.DataFrame, y: pd.Series, __, current_features: List[str] = None, n_cpus=None,
        **___):
    remaining = [x for x in predictor.features(feature_stage='transformed').values if x not in current_features]

    scores = {}
    prev_rounded = 0
    with ThreadPool(processes=n_cpus) as pool:
        for i, x in enumerate(pool.imap(functools.partial(rank_sequential_backward_elimination, X, y, only_last=True),
                                        [current_features + [f] for f in remaining])):
            scores[remaining[i]] = x
            perc = round(i * 100 / len(remaining))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info(f"Computing scores at {perc=}%")
    return scores
