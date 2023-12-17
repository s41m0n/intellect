from typing import Dict, List

import numpy as np
from river import drift, metrics, stream
from tqdm import tqdm

from .model import BaseModel, MyDataset


def online_learning(model: BaseModel, data: MyDataset, metric=metrics.Accuracy(),
                    drift_detector: drift.DummyDriftDetector = None,
                    drift_checker: callable = None,
                    drift_reactor: callable = None,
                    verbose=1000):
    metric: metrics.Accuracy = metric.clone()
    yp, m, d = [], [], []

    if hasattr(model, "cuda"):
        model.cuda()

    with tqdm(range(data.n_samples)) as pbar:
        for i, (xi, yi) in enumerate(stream.iter_pandas(data.X, data.y)):
            ypred = model.predict_one(xi)
            model.learn_one(xi, yi)
            metric.update(yi, ypred)
            yp.append(ypred)
            m.append(metric.get())
            if drift_detector is not None:
                drift_detector.update(int(not yi == ypred))
                if drift_detector.drift_detected:
                    d.append(i)
            elif drift_checker is not None and drift_checker(model, drift_detector, len(d)):
                d.append(i)
            if drift_reactor is not None and d[-1] == i:
                drift_reactor(model)
            if verbose is not None and i % verbose == 0:
                pbar.set_description(
                    f"{metric.__class__.__name__} {round(m[-1], 2):.2f} - N Drifts {len(d)}", refresh=False)
                pbar.update(verbose if i != 0 else 0)
        pbar.update(data.n_samples)
    yp = np.nan_to_num(np.array(yp, dtype=float), nan=0)

    if hasattr(model, "cpu"):
        model.cpu()

    return m, yp, d


def get_data_drifts(ds_origin: MyDataset, ds_target: MyDataset, detector=drift.ADWIN()):
    feature_drifts: Dict[str, List[int]] = {}
    for col in ds_origin.X.columns:
        detector._reset()
        for row in ds_origin.X[col]:
            detector.update(row)
        for i, row in enumerate(ds_target.X[col]):
            detector.update(row)
            if detector.drift_detected:
                if col not in feature_drifts:
                    feature_drifts[col] = [i]
                else:
                    feature_drifts[col].append(i)
    return feature_drifts
