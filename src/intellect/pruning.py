import math
from typing import List

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .model import BaseModel, MyDataset


def search_prune(model: BaseModel,
                 step: float,
                 data: MyDataset,
                 start_amount: float = 0.,
                 prune_method: str = "globally_unstructured_connections_l1",
                 performance_drop: float = None,
                 metric=accuracy_score,
                 **kwargs):
    res: List[BaseModel] = [model.clone(init=False)]
    amounts = [0.]
    scores = [metric(data.y.values, model.predict(data.X))]
    n_iter = math.floor((1 - start_amount) / step) + 1
    pruning_alg = model.get_pruning_algorithm(prune_method)
    with tqdm(range(n_iter)) as pbar:
        for i in range(n_iter):
            current_amount = round(start_amount + step * i, 2)
            pruned = model.clone(init=False)
            pruning_alg(pruned, current_amount, **kwargs)
            current_score = metric(data.y.values, pruned.predict(data.X))
            if performance_drop is not None and scores[0] - current_score > performance_drop:
                break
            res.append(pruned)
            scores.append(current_score)
            amounts.append(current_amount)
            pbar.update()
    return res, amounts, scores
