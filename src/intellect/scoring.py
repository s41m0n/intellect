"""
Module containing utility functions for scoring.
"""
from numbers import Number
from typing import Callable

import numpy as np
import pandas as pd
from river import metrics
from sklearn.metrics import accuracy_score

from .dataset import Dataset
from .model.base import BaseModel


def knowledge_loss_gain_score(df: pd.DataFrame, func_weights: tuple[float] = None) -> pd.Series:
    """Function to measure with different indicators the knowledge loss/gain with respect to
    a baseline scenario, also provided in the dataframe.

    Args:
        df (pd.DataFrame): dataframe with data to be compared
        func_weights (tuple[float], optional): list of weights to assign to each
            indicator. Defaults to None.

    Returns:
        pd.Series: a series containing all the indicators and the final evaluation function score.
    """
    df = df[df.columns.difference(['Global'])]

    if func_weights is None:
        func_weights = [0.25] * 4

    target = 'During Test' if 'During Test' in df.index else 'Test Before'
    tmp = df.loc[target]
    seen = tmp[~tmp.isna()].index.values
    unseen = tmp[tmp.isna()].index.values

    seen_proportions = pd.Series([1/len(seen)]*len(seen), index=seen)
    unseen_proportions = pd.Series([1/len(unseen)]*len(unseen), index=unseen)

    diff = df.loc['Validation After'] - df.loc['Validation Before']
    diff_seen = diff[seen]
    diff_unseen = diff[unseen]

    loss_seen = diff_seen[diff_seen < 0]
    loss_seen = (loss_seen*seen_proportions).sum()
    gain_seen = diff_seen[diff_seen > 0]
    gain_seen = (gain_seen*seen_proportions).sum()

    loss_unseen = gain_unseen = 0
    if unseen.size:
        loss_unseen = diff_unseen[diff_unseen < 0]
        loss_unseen = (loss_unseen*unseen_proportions).sum()
        gain_unseen = diff_unseen[diff_unseen > 0]
        gain_unseen = (gain_unseen*unseen_proportions).sum()
    else:
        to_add = func_weights[2] / 2 + func_weights[3] / 2
        func_weights[2] = func_weights[3] = 0.
        func_weights[0] += to_add
        func_weights[1] += to_add

    factors = [loss_seen, gain_seen, loss_unseen, gain_unseen]

    return pd.Series([sum(factors[i] * func_weights[i] for i in range(len(factors)))] + factors,
                     index=['Func', 'Loss Seen', 'Gain Seen', 'Loss Unseen', 'Gain Unseen'])

def safe_division(dividend: Number, divisor: Number) -> float:
    """Perform safe division between two numbers, returning
    0 in case of infinite division.

    Args:
        dividend (Number): the nominator
        divisor (Number): the denominator

    Returns:
        Number: the result of the division
    """
    if divisor == 0:
        return 0
    return dividend / divisor


def compute_metric_percategory_on_datasets(
        model: BaseModel, scorer: Callable, datasets: list[Dataset], names: list[str]) -> pd.DataFrame:
    """Function to compute the score metric on each provided dataset.

    Args:
        model (BaseModel): the model used for prediction
        scorer (Callable): the evaluation metric
        datasets (list[Dataset]): the list of datasets to be tested
        names (list[str]): the name of the tested datasets

    Returns:
        pd.DataFrame: a pandas DataFrame containing the scores for each category (columns) and dataset (rows)
    """
    df_percategory = pd.DataFrame()
    for ds, name in zip(datasets, names):
        y_pred_tmp = model.predict(ds.X)
        tmp_percategory = compute_metric_percategory(ds.y.values, y_pred_tmp, ds._y, scorer=scorer)
        df_percategory = pd.concat((df_percategory, pd.DataFrame(tmp_percategory, index=[name])))
    return df_percategory


def compute_metric_percategory(
        ytrue: list, ypred: list, labels: pd.Series,
        scorer: Callable = accuracy_score) -> dict[str, float]:
    """Function to compute the metric foreach category available.

    Args:
        ytrue (list): array containing the true labels
        ypred (list): array containing the predicted labels
        labels (pd.Series): categories assigned to each entry
        scorer (Callable, optional): evaluation metric. Defaults to accuracy_score.

    Returns:
        dict[str, float]: dictionary with the score for each category.
    """
    ret = {}
    ytrue = _ensure_type(ytrue)
    ypred = _ensure_type(ypred)
    labels = _ensure_type(labels)

    ret['Global'] = scorer(ytrue, ypred)
    for k in np.unique(labels):
        indexes = np.where(labels == k)[0]
        ret[k] = scorer(ytrue[indexes], ypred[indexes])
    return ret


def compute_metric_incremental(
        ytrue: list, ypred: list, metric: metrics.base = metrics.Accuracy()) -> list[float]:
    """Function to compute the metric incrementally point after point.

    Args:
        ytrue (list): the list of true labels
        ypred (list): the list of predicted labels
        metric (metrics.base, optional): the incremental metric to be used. Defaults to metrics.Accuracy().

    Returns:
        list[float]: list of the metric computed after each point.
    """
    ytrue = _ensure_type(ytrue)
    ypred = _ensure_type(ypred)

    metric = metric.clone()
    m = []
    for i, v in enumerate(ytrue):
        metric.update(v, ypred[i])
        m.append(metric.get())
    return m

def _ensure_type(x):
    if hasattr(x, 'numpy'):
        return x.numpy()
    if hasattr(x, 'to_numpy'):
        return x.to_numpy()
    return x
