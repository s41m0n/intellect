from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from .dataset import Dataset
from .model.base import BaseModel


def rank_metric_zero(
        model: BaseModel, data: Dataset, current_features: list[str] = None, metric: Callable = accuracy_score) -> dict[str, float]:
    """Function to compute feature importance by measuring the model accuracy deltas 
    when using the feature or not. 

    Args:
        model (BaseModel): model to be used for inference
        data (Dataset): dataset of interest
        current_features (list[str], optional): list of features to preserve, while zeroing the others. Defaults to None.
        metric (str, optional): evaluation metric. Defaults to "accuracy".

    Returns:
        dict[str, float]: dictionary with the score associated to each feature
    """
    if current_features is not None:
        data = data.filter_features(current_features)
    else:
        current_features = data.features
    scores = {}
    baseline = metric(data.y, model.predict(data.X))
    for x in current_features:
        tmp = data.X[x]
        data.X[x] = 0.
        scores[x] = baseline - metric(data.y, model.predict(data.X))
        data.X[x] = tmp
    return scores


def rank_metric_permutation_sklearn(
        model: BaseModel, data: Dataset, current_features: list[str] = None, metric: str = "accuracy", **kwargs) -> dict[str, float]:
    """Feature to compute feature importance using sklearn permutation method with the provided metric 

    Args:
        model (BaseModel): model to be used for inference
        data (Dataset): dataset of interest
        current_features (list[str], optional): list of features to preserve, while zeroing the others. Defaults to None.
        metric (str, optional): evaluation metric. Defaults to "accuracy".

    Returns:
        dict[str, float]: dictionary with the score associated to each feature
    """
    if current_features is not None:
        data = data.filter_features(current_features)
    else:
        current_features = data.features
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]
    importances = permutation_importance(
        model, data.X, data.y, scoring=metric, **kwargs).importances_mean
    return dict(zip(current_features, [importances[i] for i in positions]))


def rank_random_forest(
        model: BaseModel, data: Dataset, current_features: list[str] = None, estimators: int = 100, depth: int = 7,
        features: str = "sqrt", criterion="gini") -> dict[str, float]:
    """Function to compute feature importance by using an external random forest classifier. In case the model
    provided is a RandomForestClassifier, use the model itself.

    Args:
        model (BaseModel): model to be used
        data (Dataset): dataset of interest
        current_features (list[str], optional): list of features to preserve, while zeroing the others. Defaults to None.
        estimators (int, optional): number of trees. Defaults to 100.
        depth (int, optional): depth of the forest. Defaults to 7.
        features (str, optional): features used in the forest. Defaults to "sqrt".
        criterion (str, optional): criterion for measuring the quality of a split. Defaults to "gini".

    Returns:
        dict[str, float]: dictionary with the score associated to each feature
    """
    if current_features is not None:
        data = data.filter_features(current_features)
    else:
        current_features = data.features
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]
    if isinstance(model, RandomForestClassifier):
        rf = model
    else:
        rf = RandomForestClassifier(n_estimators=estimators, max_depth=depth,
                                    max_features=features, criterion=criterion)
        rf.fit(data.X.values, data.y.values)

    return dict(zip(current_features, [rf.feature_importances_[i] for i in positions]))


def rank_principal_component_analysis(model: BaseModel, data: Dataset, current_features=None) -> dict[str, float]:
    """Function to compute feature importance by using Principal Component Analysis.

    Args:
        model (BaseModel): model to be used
        data (Dataset): dataset of interest
        current_features (list[str], optional): list of features to preserve, while zeroing the others. Defaults to None.

    Returns:
        dict[str, float]: dictionary with the score associated to each feature
    """
    if current_features is not None:
        data = data.filter_features(current_features)
    else:
        current_features = data.features
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]

    if isinstance(model, PCA):
        pca = model
    else:
        pca = PCA()
        pca.fit(data.X.values, data.y.values)

    out = np.array([pca.components_[i] * np.sqrt(pca.explained_variance_ratio_[i])
                    for i in range(len(pca.explained_variance_ratio_))])

    values = np.power(out, 2).sum(axis=1)

    return {k: v for k, v in zip(values[positions], current_features)}


def sequential_backward_elimination(
        model: BaseModel, data: Dataset, rank_algorithm: Callable = None, fixed_rank: dict[str, float] = None,
        metric: Callable = accuracy_score, **kwargs) -> tuple[dict[str, float], list[float], list[str]]:
    """Function to perform Sequential Backward Elimination with the provided info.

    Args:
        model (BaseModel): the model to be used
        data (Dataset): the dataset of interest
        rank_algorithm (Callable): rank algorithm to be used. It can be either one of the above methods
            or one custom defined in the model directory. Provided in case at each step the rank should
            be computed. Default to None.
        fixed_rank (dict[str, float], optional): fixed rank to be used instead of computing the recursive
            one (rank_algorithm). Defaults to None.
        metric (Callable, optional): evaluation metric. Defaults to accuracy_score.

    Raises:
        ValueError: when at least one among fixed_rank and rank_algorithm is not specified

    Returns:
        tuple[dict[str, float], list[float], list[str]]: tuple with ranks, list of scores, and list
            of removed features at each step.
    """
    if fixed_rank is None and rank_algorithm is None:
        raise ValueError("One among fixed_rank and rank_algorithm must be different than None")

    data = data.clone()
    current_features = list(fixed_rank.keys()) if fixed_rank else data.X.columns.values.tolist()
    results = []
    results_metric = []
    removed_names = []
    while current_features:
        results_metric.append(metric(data.y.values, model.predict(data.X.values)))
        if fixed_rank is None:
            scores = rank_algorithm(
                model, data, current_features=current_features, **kwargs)
        else:
            scores = {k: v for k, v in fixed_rank.items() if k in current_features}

        results.append(scores)
        worst_feature, _ = min(scores.items(), key=lambda x: x[1])

        removed_names.append(worst_feature)

        current_features.remove(worst_feature)

    return results, results_metric, removed_names
