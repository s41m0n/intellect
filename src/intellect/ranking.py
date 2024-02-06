"""
Module with utilities function and generic approaches to perform feature ranking
with a provided classifier.
"""
from typing import Callable, Generator

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from .dataset import Dataset
from .model.base import BaseModel


def rank_metric_zero(
        model: BaseModel, data: Dataset, current_features: list[str] = None,
        metric: Callable = accuracy_score) -> dict[str, float]:
    """Function to compute feature importance by measuring the model accuracy deltas
    when using the feature or not.

    Args:
        model (BaseModel): model to be used for inference
        data (Dataset): dataset of interest
        current_features (list[str], optional): list of features to preserve, while zeroing the others.
            Defaults to None.
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
        model: BaseModel, data: Dataset, current_features: list[str] = None,
        metric: str = 'accuracy', **kwargs) -> dict[str, float]:
    """Feature to compute feature importance using sklearn permutation method with the provided metric

    Args:
        model (BaseModel): model to be used for inference
        data (Dataset): dataset of interest
        current_features (list[str], optional): list of features to preserve, while zeroing the others.
            Defaults to None.
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
        features: str = 'sqrt', criterion='gini') -> dict[str, float]:
    """Function to compute feature importance by using an external random forest classifier. In case the model
    provided is a RandomForestClassifier, use the model itself.

    Args:
        model (BaseModel): model to be used
        data (Dataset): dataset of interest
        current_features (list[str], optional): list of features to preserve, while zeroing the others.
            Defaults to None.
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
        current_features (list[str], optional): list of features to preserve, while zeroing the others.
            Defaults to None.

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

    return dict(zip(values[positions], current_features))


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
        raise ValueError('One among fixed_rank and rank_algorithm must be different than None')

    data = data.clone()
    current_features = list(fixed_rank.keys()) if fixed_rank else data.X.columns.values.tolist()
    while current_features:
        score = metric(data.y.values, model.predict(data.X.values))

        if fixed_rank is None:
            scores = rank_algorithm(
                model, data, current_features=current_features, **kwargs)
        else:
            scores = {k: v for k, v in fixed_rank.items() if k in current_features}

        worst_feature, _ = min(scores.items(), key=lambda x: x[1])
        current_features.remove(worst_feature)
        data.X[worst_feature] = 0.
        yield score, worst_feature, scores

def subset_search(
        model: BaseModel, ds: Dataset, ratio: float, attempts: int,
        rank: dict[str, float] = None, metric: Callable = accuracy_score, performance_drop_ratio: float = None,
        baseline: float = None, stuck_guard: int = 1000) -> Generator[list[str], float, bool]:
    """Function to perform only subset search given a classifier.

    Args:
        model (BaseModel): the model to be used for the subset evaluation.
        ds (Dataset): the dataset containing the data to be used.
        ratio (float): percentage of the features to preserve.
        attempts (int): number of possible combinations to explore
        rank (dict[str, float], optional): previously computed rank of
            all the available features in the dataset. If not None, then
            a weighted random search of the subset is performed. Defaults to None.
        metric (Callable, optional): evaluation metric to be used. Defaults to accuracy_score.
        performance_drop_ratio (float, optional): maximum ratio of
            performance drop using the provided metric. Defaults to None.
        baseline (float, optional): baseline value to which refer during the computation
            of the performance drop. Defaults to None.
        stuck_guard (int, optional): maximum attempts to randomly pick a new
            combination of features to avoid infinite loops. Defaults to 1000.

    Raises:
        ValueError: when unable to pick a new unexplored combination of features.

    Yields:
       list[str], float, bool: for each explored attempt, the list of active features
                    in the subset, the metric score achieved and whether is accepted or
                    not with respect to the performance drop ratio value provided.
    """
    if baseline is None:
        baseline = metric(ds.y, model.predict(ds.X))

    weights = None
    if rank is not None:
        weights = np.array([rank[k] for k in ds.features])
        weights += np.abs(np.min(weights))
        weights = weights/weights.sum()

    explored_set_names = {}
    for _ in range(attempts):
        for i in range(stuck_guard):
            choices = tuple(np.random.choice(ds.features, round(ratio*ds.n_features), replace=False, p=weights))
            if choices not in explored_set_names:
                break
        if i == stuck_guard:
            raise ValueError(f'Unable to find a subset within the number {stuck_guard=}')
        explored_set_names[choices] = True
        tmp = ds.filter_features(choices, default=0)
        score = metric(tmp.y, model.predict(tmp.X))
        is_accepted = performance_drop_ratio is None or (score >= (1-performance_drop_ratio) * baseline).item()
        yield choices, score, is_accepted


def prune_search(
        model: BaseModel, ds: Dataset, prune_algorithm: Callable, prune_ratios: list[float], *args,
        metric: Callable = accuracy_score, performance_drop_ratio: float = None,
        baseline: float = None, **kwargs) -> Generator[float, float, bool]:
    """Function to perform only the pruning of a given classifier.

    Args:
        model (BaseModel): the model to be used for the subset evaluation.
        ds (Dataset): the dataset containing the data to be used.
        prune_algorithm (Callable): pruning algorithm to be used.
        prune_ratios (list[float]): list of pruning ratios to try.
        metric (Callable, optional): evaluation metric to be used. Defaults to accuracy_score.
        baseline (float, optional): baseline value to which refer during the computation
            of the performance drop. Defaults to None.
        baseline (float, optional): _description_. Defaults to None.

    Yields:
        Generator[float, float, bool]: for each explored attempt, the pruning ratio,
                    the score associated to the pruned model and whether it is accepted
                    or not considering the provided performance drop ratio.
    """
    if isinstance(prune_ratios, (float)):
        prune_ratios = [prune_ratios]

    if baseline is None:
        baseline = metric(ds.y, model.predict(ds.X))
    for ratio in prune_ratios:
        pruned = prune_algorithm(model, ratio, *args, **kwargs)
        score = metric(ds.y, pruned.predict(ds.X))

        is_accepted = performance_drop_ratio is None or (score >= (1 - performance_drop_ratio) * baseline).item()
        yield ratio, score, is_accepted

def prune_and_subset_search(
        model: BaseModel, prune_algorithm: Callable, ds: Dataset, prune_ratios: list[float],
        subset_ratio: float, subset_attempts: int, *args, rank: dict[str, float] = None,
        metric: Callable = accuracy_score, baseline: float = None,
        performance_drop_ratio: float = None, **kwargs) -> Generator[float, dict[tuple[str], float]]:
    """Function to perform jointly (a) model pruning and then (b) the feature subset search.

    Args:
        model (BaseModel): model to be pruned using during the process.
        prune_algorithm (Callable): the pruning algorithm.
        ds (Dataset): the data to be used.
        prune_ratios (list[float]): list of pruning ratios to explore.
        subset_ratio (float): percentage of active feature in the subset to look for.
        subset_attempts (int): number of explored subsets.
        rank (dict[str, float], optional): previously computed rank of
            all the available features in the dataset. If not None, then
            a weighted random search of the subset is performed. Defaults to None.
        metric (Callable, optional): the evaluation metric. Defaults to accuracy_score.
        baseline (float, optional): baseline value to which consider the performance drop. Defaults to None.
        performance_drop_ratio (float, optional): max accepted drop in the performance. Defaults to None.

    Yields:
        Generator[float, dict[tuple[str], float]]: for each value, returns the prune ratio,
            and a dictionary with the list of active features as a key and the obtained
            final accuracy as value.
    """

    if isinstance(prune_ratios, (float)):
        prune_ratios = [prune_ratios]

    if baseline is None:
        baseline = metric(ds.y, model.predict(ds.X))
    for ratio in prune_ratios:
        pruned = prune_algorithm(model, ratio, *args, **kwargs)
        accepted_res = {
            x: v for x, v,
            cond
            in
            subset_search(
                pruned, ds, subset_ratio, subset_attempts, performance_drop_ratio=performance_drop_ratio,
                rank=rank, baseline=baseline) if cond is True}
        yield ratio, accepted_res

def subset_search_and_prune(
        model: BaseModel, prune_algorithm: Callable, ds: Dataset, prune_ratios: list[float], subset_ratio: float,
        subset_attempts: int, *args, rank: dict[str, float] = None, metric: Callable = accuracy_score,
        baseline: float = None, performance_drop_ratio: float = None, **kwargs) -> Generator[list[str], float, float]:
    """Function to perform jointly (a) the feature subset search and then (b) the pruning of the classifier.

    Args:
        model (BaseModel): model to be pruned using during the process.
        prune_algorithm (Callable): the pruning algorithm.
        ds (Dataset): the data to be used.
        prune_ratios (list[float]): list of pruning ratios to explore.
        subset_ratio (float): percentage of active feature in the subset to look for.
        subset_attempts (int): number of explored subsets.
        rank (dict[str, float], optional): previously computed rank of
            all the available features in the dataset. If not None, then
            a weighted random search of the subset is performed. Defaults to None.
        metric (Callable, optional): the evaluation metric. Defaults to accuracy_score.
        baseline (float, optional): baseline value to which consider the performance drop. Defaults to None.
        performance_drop_ratio (float, optional): max accepted drop in the performance. Defaults to None.

    Yields:
        Generator[list[str], float, float]: list with active features for the subset, the metric score,
            and the pruning ratio accepted. Multiple ratios are provided in different return values.
    """

    if isinstance(prune_ratios, (float)):
        prune_ratios = [prune_ratios]

    if baseline is None:
        baseline = metric(ds.y, model.predict(ds.X))

    accepted_res = [x for x, _, cond in subset_search(
        model, ds, subset_ratio, subset_attempts, performance_drop_ratio=performance_drop_ratio,
        rank=rank, baseline=baseline) if cond is True]

    for k in accepted_res:
        for ratio in prune_ratios:
            pruned = prune_algorithm(model, ratio, *args, **kwargs)
            score = metric(ds.y, pruned.predict(ds.X))
            is_accepted = performance_drop_ratio is None or (score >= (1 - performance_drop_ratio) * baseline).item()
            if not is_accepted:
                continue
            yield k, score, ratio
