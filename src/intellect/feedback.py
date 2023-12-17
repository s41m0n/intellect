from copy import deepcopy

import numpy as np
import torch
from river import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import BaseModel, MyDataset


def knowledge_distillation(brm: BaseModel, lvm: BaseModel, data: torch.Tensor, labels: torch.Tensor, epochs=1):
    labels = brm.predict_proba(data)
    for _ in range(epochs):
        lvm.learn(data, labels)


def normal_learning(_, lvm: BaseModel, data: torch.Tensor, labels: torch.Tensor, epochs=1):
    for _ in range(epochs):
        lvm.learn(data, labels)


def weight_update_proxy(brm: BaseModel, lvm: BaseModel, data, labels, mu=0, epochs=1, restore=False):
    if restore:
        brm: BaseModel = deepcopy(brm)
        prev_state_dict = deepcopy(brm.state_dict())

    for _ in range(epochs):
        brm.learn(data, labels, adjust_factor=mu / 2 * brm.distance(lvm, only_prunable=True, method="L2_NORM"))

    lvm.load_state_dict(brm.state_dict())


def weight_update(brm: BaseModel, lvm: BaseModel, data, labels, epochs=1):
    return weight_update_proxy(brm, lvm, data, labels, mu=0, epochs=epochs, restore=False)


def weight_update_oneshot(brm: BaseModel, lvm: BaseModel, data, labels, epochs=1):
    return weight_update_proxy(brm, lvm, data, labels, mu=0, epochs=epochs, restore=True)


def feedback_learning(
        teacher: BaseModel, student: BaseModel, dataset: MyDataset, algorithm="algorithm_",
        batch_size=256, metric=metrics.Accuracy(), drop_ratio=None, categories=[], **kwargs):
    alg_func = student.get_feedback_algorithm(algorithm)
    y_pred = torch.empty(0)
    m = []
    ds_tmp = dataset
    if metric:
        metric: metrics.Accuracy = metric.clone()

    if categories:
        if not all(v in dataset.categories for v in categories):
            raise Exception("Invalid features, not present in dataset")
        ds_tmp = dataset.filter(categories)

    dl = DataLoader(ds_tmp, batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=True, persistent_workers=True)

    if torch.cuda.is_available():
        teacher.cuda()
        student.cuda()

    device = student.device
    with tqdm(range(len(dl))) as pbar:
        for i, (inputs, labels) in enumerate(dl):
            inputs = inputs.to(device)
            labels = labels.to(device)
            yp = student.predict(inputs)
            y_pred = torch.cat((y_pred, yp.cpu()), dim=0)
            if metric:
                for i in range(len(labels)):
                    metric.update(labels[i].item(), yp[i].item())
                    m.append(metric.get())
                pbar.set_description(f"{metric.__class__.__name__} {round(m[-1], 2)}", refresh=False)
            if drop_ratio is not None:
                indexes = np.random.choice(len(inputs), size=round(len(inputs) * (1 - drop_ratio)), replace=False)
                inputs, labels = inputs[indexes, ...], labels[indexes, ...]
            alg_func(teacher, student, inputs, labels, **kwargs)
            pbar.update()

    teacher.cpu()
    student.cpu()
    return m, y_pred, ds_tmp.y, ds_tmp._y
