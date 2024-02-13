"""
Module providing compatibility for river Ensembles jointly with local defined models.
"""
import math

import numpy as np
from river.base import Ensemble
from river.ensemble import LeveragingBaggingClassifier, SRPClassifier
from tqdm import tqdm

from ..dataset import Dataset
from .base import BaseModel


class WrapRiverEnsemble(BaseModel):
    """Wrapper for the ensemble class when using a local model (e.g., torch.model.Mlp)
    with a river Ensemble. This class adds few methods to make it compatible and usable.
    Note that this is experimental, few functionalities might not work as they are not implemented.
    """

    def __init__(self, cls: Ensemble, *args, **kwargs):
        super().__init__()
        self.item: Ensemble = cls(*args, **kwargs)

    @property
    def prev_drifts(self) -> int:
        """Property to return the number of previous concept drift recorded.

        Returns:
            int: number of previous drifts
        """
        if self.item.__class__ == LeveragingBaggingClassifier:
            return self.item.n_detected_changes
        return 0

    def learn(self, X: list, y: list, *args, **kwargs) -> tuple[list[float], float]:
        drifts = []
        prev_drifts = self.prev_drifts
        for i in range(len(X)):
            self.item.learn_one(X.iloc[i:i+1], y.iloc[i:i+1].to_numpy(), *args, **kwargs)
            if self.is_concept_drift(prev_drifts):
                drifts.append(i)
        return np.array(drifts)

    def predict(self, X: list, *args, **kwargs) -> list[int]:
        return [self.item.predict_one(X.iloc[i:i+1], as_dict=True) for i in range(len(X))]

    def predict_proba(self, X: list, *args, **kwargs) -> list[float]:
        return [self.item.predict_proba_one(X.iloc[i:i+1], as_dict=True) for i in range(len(X))]

    def fit(self, *args, **kwargs) -> dict[str, list[float]] | None:
        raise NotImplementedError()

    def clone(self, init: bool = True) -> 'BaseModel':
        raise NotImplementedError()

    def continuous_learning(self, ds: Dataset, *args, batch_size=1, epochs=1, **kwargs) -> tuple[list[int],
                                                                                                 list[int],
                                                                                                 list[int]]:
        y_preds, y_true, y_labels, drifts = np.empty(0), np.empty(0), np.empty(0), np.empty(0)

        niter = math.ceil(len(ds)/batch_size)

        for i in tqdm(range(niter)):
            base = i*batch_size
            inputs = ds.X.iloc[base:base+batch_size]
            labels = ds.y.iloc[base:base+batch_size]
            type_labels = ds._y.iloc[base:base+batch_size]

            pred = self.predict(inputs)

            y_preds = np.concatenate((y_preds, pred), axis=0)
            y_true = np.concatenate((y_true, labels.to_numpy()), axis=0)
            y_labels = np.concatenate((y_labels, type_labels.to_numpy()), axis=0)

            for j in range(epochs):
                curr_drifts = self.learn(inputs, labels)
                if j+1 == epochs and curr_drifts.size != 0:
                    curr_drifts = curr_drifts * i + curr_drifts
                    drifts = np.concatenate((drifts, curr_drifts))
        return y_preds, y_true, y_labels, drifts

    def is_concept_drift(self, prev_concept, *args, **kwargs) -> bool:
        if self.item.__class__ == LeveragingBaggingClassifier:
            return self.item.n_detected_changes != prev_concept
        if self.item.__class__ == SRPClassifier:
            return self.item.drift_detector is not None and self.item.drift_detector.drift_detected
        raise ValueError()

    @property
    def prunable(self) -> list[str] | list[object]:
        return []
