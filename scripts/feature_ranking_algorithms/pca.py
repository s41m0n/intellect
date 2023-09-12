# calibrate_decision_threshold
import functools
import os
from multiprocessing.pool import ThreadPool
from typing import List

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.decomposition import PCA
from utils.common import get_logger

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def rank_sequential_backward_elimination(
        predictor: TabularPredictor, X: pd.DataFrame, __, ___, current_features: List[str] = None, only_last=False, **
        ____):
    X = X.copy(deep=True)

    if current_features is None:
        current_features = predictor.features(feature_stage='transformed')

    positions = [i for i, x in enumerate(predictor.features(feature_stage='transformed')) if x in current_features]

    logger.info("Computing scores")

    pca = PCA()
    pca.fit(X)

    logger.info("Adjusting PCA scores")

    out = [pca.components_[i] * np.sqrt(pca.explained_variance_ratio_[i])
           for i in range(len(pca.explained_variance_ratio_))]
    df = pd.DataFrame(out, index=pd.Index(predictor.features(feature_stage='transformed'), name="ID"))
    if only_last:
        imp = pd.DataFrame(df.pow(2).sum().iloc[positions].values[-1],
                           index=pd.Index([current_features[-1]], name="ID"), columns=["importance"])
    else:
        imp = pd.DataFrame(df.pow(2).sum().iloc[positions].values, index=pd.Index(
            current_features, name="ID"), columns=["importance"])
    return imp


def rank_sequential_forward_selection(
        predictor: TabularPredictor, X: pd.DataFrame, __, ___, current_features: List[str] = None, n_cpus=None, **____):
    remaining = [x for x in predictor.features(feature_stage='transformed') if x not in current_features]

    scores = []
    prev_rounded = 0
    with ThreadPool(processes=n_cpus) as pool:
        for x in pool.imap(
                functools.partial(
                    rank_sequential_backward_elimination, None, X, None, None, n_cpus=None, only_last=True),
                [current_features + [f] for f in remaining]):
            scores.append(x)
            perc = round(len(scores) * 100 / len(remaining))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info(f"Computing scores at {perc=}%")
    res = pd.concat(scores)
    return res
