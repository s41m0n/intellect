from river.base import Ensemble

from ..dataset import Dataset
from .base import BaseModel


class WrapRiverEnsemble(BaseModel):
    def __init__(self, cls: Ensemble, *args, **kwargs):
        super().__init__()
        self.item: Ensemble = cls(*args, **kwargs)

    def continuous_learning(self,):
        self.item.learn_one()

    def learn(self, X: list, y: list, *args, **kwargs) -> tuple[list[float], float]:
        for x, l in zip(X, y):
            self.item.learn_one(x, l)

    def predict(self, X: list, *args, **kwargs) -> list[int]:
        return [self.item.predict_one(v, as_dict=True) for v in X]

    def predict_proba(self, X: list, *args, **kwargs) -> list[float]:
        return [self.item.predict_proba_one(v, as_dict=True) for v in X]

    def fit(self, *args, **kwargs) -> dict[str, list[float]] | None:
        raise NotImplementedError()

    def clone(self, init: bool = True) -> 'BaseModel':
        raise NotImplementedError()

    def continuous_learning(self, data: Dataset, *args, **kwargs) -> tuple[list[int], list[int], list[int]]:
        ypred = []
        for i, (inputs, labels) in enumerate(data):
            ypred += self.predict([inputs])
            self.item.learn_one(inputs, labels)
            if i % 1000 == 0:
                print(i)
        return ypred, data.y, data._y, []

    @property
    def is_concept_drift(self) -> bool:
        pass

    def concept_react(self, *args, **kwarg) -> None:
        pass

    @property
    def prunable(self) -> list[str] | list[object]:
        return []
