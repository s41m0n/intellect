import sys
from copy import deepcopy

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from .model import BaseModel, MyDataset


def rank_metric_zero(model: BaseModel, data: MyDataset, current_features, metric=accuracy_score):
    scores = {}
    baseline = metric(data.y, model.predict(data.X))
    for x in current_features:
        tmp = data.X[x]
        data.X[x] = 0.
        scores[x] = baseline - metric(data.y, model.predict(data.X))
        data.X[x] = tmp
    return scores


def rank_metric_permutation_sklearn(
        model: BaseModel, data: MyDataset, current_features, metric=accuracy_score, n_repeats=5):
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]
    importances = permutation_importance(
        model, data.X, data.y, n_repeats=n_repeats, scoring=metric, max_samples=0.5).importances_mean
    return dict(zip(current_features, [importances[i] for i in positions]))


def rank_random_forest(
        model: BaseModel, data: MyDataset, current_features, estimators=100, depth=7, features="sqrt", criterion="gini"):
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]
    if isinstance(model, RandomForestClassifier):
        rf = model
    else:
        rf = RandomForestClassifier(n_estimators=estimators, max_depth=depth,
                                    max_features=features, criterion=criterion)
        rf.fit(data.X.values, data.y.values)

    return dict(zip(current_features, [rf.feature_importances_[i] for i in positions]))


def rank_principal_component_analysis(model: BaseModel, data: MyDataset, current_features):
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
        model: BaseModel, data: MyDataset, fixed_rank=None, rank_algorithm=None, metric=accuracy_score, **kwargs):

    if fixed_rank is None and rank_algorithm is None:
        raise RuntimeError("One among fixed_rank and rank_algorithm must be different than None")

    data = deepcopy(data)
    current_features = list(fixed_rank.keys()) if fixed_rank else data.X.columns.values.tolist()
    results = []
    results_metric = []
    removed_names = []
    while current_features:
        results_metric.append(metric(data.y.values, model.predict(data.X.values)))
        if fixed_rank is None:
            scores = get_ranking_algorithm(rank_algorithm)(model, data, current_features, **kwargs)
        else:
            scores = {k: v for k, v in fixed_rank.items() if k in current_features}

        results.append(scores)
        worst_feature, _ = min(scores.items(), key=lambda x: x[1])

        removed_names.append(worst_feature)

        current_features.remove(worst_feature)
        data.X[worst_feature] = 0.

    return results, results_metric, removed_names
