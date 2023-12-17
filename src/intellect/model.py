from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch


class MyDataset(torch.utils.data.Dataset):

    def __init__(
            self, csv_file=None, n_samples=None, shuffle=True, index_col=0,
            label="Label", label_type="Type", data=None):
        if csv_file is not None:
            self.X = pd.read_csv(csv_file, index_col=index_col)
        else:
            self.X = data
        if shuffle:
            self.X = self.X.sample(frac=1)
        if n_samples:
            self.X = self.X.groupby(label, group_keys=False).apply(lambda x: x.head(n_samples))
        self.X.reset_index(drop=True, inplace=True)
        self.y = self.X.pop(label)
        if label_type == label or label_type is None:
            self._y = self.y
        else:
            self._y = self.X.pop(label_type)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]

    def filter_features(self, features: list[str]):
        ds_tmp = self.clone()
        ds_tmp.X.loc[:, ~ds_tmp.X.columns.isin(features)] = 0.
        return ds_tmp

    def filter(self, categories: List[str]):
        ds_tmp = self.clone()
        indexes = ds_tmp._y[ds_tmp._y.isin(categories)].index.values
        ds_tmp.X = ds_tmp.X.iloc[indexes].reset_index(drop=True)
        ds_tmp.y = ds_tmp.y.iloc[indexes].reset_index(drop=True)
        ds_tmp._y = ds_tmp._y.iloc[indexes].reset_index(drop=True)
        return ds_tmp

    def take(self, n: int):
        ds_tmp = self.clone()
        indexes = ds_tmp.X.sample(n).index.values
        ds_tmp.X = ds_tmp.X.iloc[indexes].reset_index(drop=True)
        ds_tmp.y = ds_tmp.y.iloc[indexes].reset_index(drop=True)
        ds_tmp._y = ds_tmp._y.iloc[indexes].reset_index(drop=True)
        return ds_tmp

    def clone(self):
        return deepcopy(self)

    @property
    def categories(self):
        return self._y.unique().tolist()

    @property
    def shape(self):
        return self.X.shape

    @property
    def features(self):
        return self.X.columns.values.tolist()

    @property
    def n_samples(self):
        return self.X.shape[0]

    @property
    def n_features(self):
        return self.X.shape[1]


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, in_features, *args, seed=True, **kwargs) -> None:
        pass

    @abstractmethod
    def forward(self, x: Union[torch.Tensor, np.ndarray, Dict], *args, **kwargs):
        pass

    @abstractmethod
    def init_learn(self, *args, **kwargs):
        pass

    @abstractmethod
    def learn(
            self, x: Union[torch.Tensor, np.ndarray, pd.DataFrame, Dict],
            y: Union[torch.Tensor, np.ndarray, pd.Series, Dict],
            *args, **kwargs):
        pass

    @abstractmethod
    def learn_one(
            self, x: Union[torch.Tensor, np.ndarray, pd.DataFrame, Dict],
            y: Union[torch.Tensor, np.ndarray, pd.Series, Dict],
            *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x: Union[torch.Tensor, np.ndarray, pd.DataFrame, Dict], *args, **kwargs):
        pass

    @abstractmethod
    def predict_one(self, x: Union[torch.Tensor, np.ndarray, pd.DataFrame, Dict], *args, **kwargs):
        pass

    @abstractmethod
    def predict_proba(self, x: Union[torch.Tensor, np.ndarray, pd.DataFrame, Dict], *args, **kwargs):
        pass

    @abstractmethod
    def predict_proba_one(self, x: Union[torch.Tensor, np.ndarray, pd.DataFrame, Dict], *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(
            self, x: Union[torch.Tensor, np.ndarray, pd.DataFrame, Dict],
            y: Union[torch.Tensor, np.ndarray, pd.Series, Dict],
            *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, train_dataset: MyDataset, *args, validation_dataset: MyDataset = None, shuffle=True, **kwargs):
        pass

    @property
    @abstractmethod
    def prunable(self) -> List[Any]:
        pass

    @abstractmethod
    def clone(self, init=True):
        pass

    def copy(self):
        return self.clone(init=False)

    @property
    @abstractmethod
    def device(self):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @abstractmethod
    def distance(self, other, only_prunable=True, method="L2_NORM"):
        pass

    @abstractmethod
    def permanent_prune(self):
        pass

    @abstractmethod
    def copy_prune(self, other: "BaseModel", *args, **kwargs):
        pass

    @classmethod
    def get_pruning_algorithms(cls):
        return tuple()

    @classmethod
    def get_pruning_algorithm(cls, name):
        return next(x for x in cls.get_pruning_algorithms() if x.__name__ == name)

    @classmethod
    def get_ranking_algorithms(cls):
        from .ranking import (rank_metric_permutation_sklearn,
                              rank_metric_zero,
                              rank_principal_component_analysis,
                              rank_random_forest)
        return (rank_metric_zero, rank_metric_permutation_sklearn, rank_random_forest, rank_principal_component_analysis)

    @classmethod
    def get_ranking_algorithm(cls, name):
        return next(x for x in cls.get_ranking_algorithms() if x.__name__ == name)

    @classmethod
    def get_feedback_algorithms(cls):
        from .feedback import (knowledge_distillation, normal_learning,
                               weight_update, weight_update_oneshot,
                               weight_update_proxy)
        return (knowledge_distillation, normal_learning, weight_update_proxy, weight_update, weight_update_oneshot)

    @classmethod
    def get_feedback_algorithm(cls, name):
        return next(x for x in cls.get_feedback_algorithms() if x.__name__ == name)

    @abstractmethod
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        pass

    def cuda(self):
        return self

    def cpu(self):
        return self
