# https://towardsdatascience.com/feature-subset-selection-6de1f05822b0
import functools
import os
import warnings
import time
from typing import List

import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import \
    TabularNeuralNetTorchModel
from multiprocess.pool import ThreadPool

from utils.common import get_logger
import torch

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def test_model_on_subset(predictor: TabularPredictor, X: pd.DataFrame, y: pd.Series, model_name,
                         group: List[str] = None):
    # torch.set_num_threads(1)
    if isinstance(model_name, tuple):
        model_name, group = model_name
    if group is not None:
        X = X.copy(deep=True)
        group = list(group)
        X[X.columns.difference(group)] = 0.0

    start = time.time()
    if isinstance(model_name, str):
        ypred = predictor.predict(X, model=model_name)
    elif isinstance(model_name, TabularNeuralNetTorchModel):
        ypred = model_name.predict(X)
    else:
        raise RuntimeError(f"Uknown {model_name} type")
    end = time.time()
    if group is not None:
        del X
    ypred = pd.Series(ypred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            score = predictor.evaluate_predictions(y, ypred, silent=True, auxiliary_metrics=True)
            score["prediction_time"] = end - start
            return score
        except ValueError:
            backup = predictor.evaluate_predictions(y, ypred, silent=True, auxiliary_metrics=False)
            backup["prediction_time"] = end - start
            return backup


def rank_sequential_backward_elimination(
        predictor: TabularPredictor, X: pd.DataFrame, y: pd.Series, model_name,
        current_features: List[str] = None, n_cpus=None, target_metric=None, **_):
    if current_features is None:
        current_features = predictor.features(feature_stage='transformed')

    baseline = test_model_on_subset(
        predictor, X, y, model_name, group=current_features)[target_metric]

    scores = {}
    prev_rounded = 0
    with ThreadPool(processes=n_cpus) as pool:
        for i, x in enumerate(pool.imap(functools.partial(test_model_on_subset, predictor, X, y, model_name),
                                        [[x for x in current_features if x != f] for f in current_features])):
            scores[current_features[i]] = baseline - x[target_metric]
            perc = round(i * 100 / len(current_features))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info(f"Computing scores at {perc=}%")
    return scores


def rank_sequential_forward_selection(
        predictor: TabularPredictor, X: pd.DataFrame, y: pd.Series, model_name,
        current_features: List[str] = None, n_cpus=None, target_metric=None, **_):
    remaining = [x for x in predictor.features(feature_stage='transformed') if x not in current_features]

    baseline = test_model_on_subset(
        predictor, X, y, model_name, group=current_features)[target_metric]

    scores = []
    prev_rounded = 0
    with ThreadPool(processes=n_cpus) as pool:
        for x in pool.imap(functools.partial(test_model_on_subset, predictor, X, y, model_name),
                           [current_features + [f] for f in remaining]):
            scores.append(x[target_metric] - baseline)
            perc = round(len(scores) * 100 / len(remaining))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info(f"Computing scores at {perc=}%")

    df = pd.DataFrame(scores, index=pd.Index(remaining, name="ID"),
                      columns=['importance'])
    return df
