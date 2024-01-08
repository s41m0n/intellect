from numbers import Number
from typing import Callable

import pandas as pd
from river import metrics
from sklearn.metrics import accuracy_score

from .dataset import Dataset
from .model.base import BaseModel


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
        tmp_percategory = compute_metric_percategory(ds.y.values, y_pred_tmp, ds._y, scorer=scorer, also_global=True)
        df_percategory = pd.concat((df_percategory, pd.DataFrame(tmp_percategory, index=[name])))
    return df_percategory


def compute_metric_percategory(
        ytrue: list, ypred: list, labels: pd.Series, scorer: Callable = accuracy_score, also_global: bool = True) -> dict[str, float]:
    """Function to compute the metric foreach category available.

    Args:
        ytrue (list): array containing the true labels
        ypred (list): array containing the predicted labels
        labels (pd.Series): categories assigned to each entry
        scorer (Callable, optional): evaluation metric. Defaults to accuracy_score.
        also_global (bool, optional): add the overall metric on the whole data. Defaults to True.

    Returns:
        dict[str, float]: dictionary with the score for each category.
    """
    ret = {}
    v_perc = dict(labels.value_counts(normalize=True).items())
    if also_global:
        ret["Global"] = scorer(ytrue, ypred)
    for k in labels.value_counts().keys():
        indexes = labels[labels == k].index.values
        ret[(k, round(v_perc.get(k, 100), 2))] = scorer(ytrue[indexes], ypred[indexes])
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
    import torch
    if isinstance(ytrue, torch.Tensor):
        ytrue = ytrue.numpy()
    if isinstance(ypred, torch.Tensor):
        ypred = ypred.numpy()

    metric = metric.clone()
    m = []
    for i in range(len(ytrue)):
        metric.update(ytrue[i], ypred[i])
        m.append(metric.get())
    return m
