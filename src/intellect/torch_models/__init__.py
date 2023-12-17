import io
import time
from abc import abstractmethod
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model import BaseModel, MyDataset
from ..utility import set_seed

torch.backends.cudnn.benchmark = True
torch.set_anomaly_enabled(False)


class TorchModel(torch.nn.Module, BaseModel):
    def __init__(self, *args, seed=True, **kwargs) -> None:
        if seed is True:
            set_seed()
        elif isinstance(seed, int):
            set_seed(default=seed)
        super().__init__(*args, **kwargs)
        self.init_params: dict[str, object] = None
        self.layers: torch.nn.Sequential = None
        self.optimizer: torch.optim.Optimizer = None
        self.loss_fn: torch.nn.modules.loss._Loss = None

    @abstractmethod
    def parse_output(self, y: torch.Tensor):
        pass

    @classmethod
    def get_pruning_algorithms(cls):
        from . import pruning
        return BaseModel.get_pruning_algorithms() + (
            pruning.globally_structured_connections_l1,
            pruning.globally_structured_connections_l2,
            pruning.globally_structured_connections_random,
            pruning.globally_structured_neurons_activation_l1,
            pruning.globally_structured_neurons_activation_l1_for_subset,
            pruning.globally_structured_neurons_activation_l2,
            pruning.globally_structured_neurons_activation_l2_for_subset,
            pruning.globally_structured_neurons_l1,
            pruning.globally_structured_neurons_l2,
            pruning.globally_structured_neurons_random,
            pruning.globally_unstructured_connections_l1,
            pruning.globally_unstructured_connections_random,
            pruning.locally_structured_connections_l1,
            pruning.locally_structured_connections_l2,
            pruning.locally_structured_connections_random,
            pruning.locally_structured_neurons_activation_l1,
            pruning.locally_structured_neurons_activation_l1_for_subset,
            pruning.locally_structured_neurons_activation_l2,
            pruning.locally_structured_neurons_activation_l2_for_subset,
            pruning.locally_structured_neurons_l1,
            pruning.locally_structured_neurons_l2,
            pruning.locally_structured_neurons_random,
            pruning.locally_unstructured_connections_l1,
            pruning.locally_unstructured_connections_random)

    @classmethod
    def get_ranking_algorithms(cls):
        from . import interpretability
        return BaseModel.get_ranking_algorithms() + (
            interpretability.rank_gradient_captum,
            interpretability.rank_perturbation_captum)

    def safe_cast_input(self, x: Union[torch.Tensor, dict, pd.DataFrame, np.ndarray], is_y=False):
        ret: torch.Tensor = None

        if isinstance(x, torch.Tensor):
            ret = x
        elif isinstance(x, dict):
            ret = torch.tensor([0 if k not in x else x[k]
                               for k in self.init_params["in_features"]])
        elif isinstance(x, (pd.DataFrame, pd.Series)):
            ret = torch.from_numpy(x.values)
        elif isinstance(x, np.ndarray):
            ret = torch.from_numpy(x)
        else:
            return torch.tensor([x], device=self.device)

        if not is_y and ret.ndim == 1:
            ret = ret[None, :]
        return ret.to(self.device)

    def clone(self, init=True):
        if not init:
            buffer = io.BytesIO()
            torch.save(self, buffer)
            buffer.seek(0)
            return torch.load(buffer)
        new_one = self.__class__(**self.init_params)
        new_optim = deepcopy(self.optimizer)
        if new_optim:
            new_optim.param_groups[0].update({"params": list(new_one.parameters())})
        new_loss = deepcopy(self.loss_fn)
        if new_loss or new_optim:
            new_one.init_learn(new_optim, new_loss)
        return new_one

    def init_learn(self, optimizer: torch.optim.Optimizer = None, loss_fn: torch.nn.modules.loss._Loss = None):
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters())
        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    @property
    def device(self):
        return next(self.parameters()).device

    def learn(self, x: torch.Tensor, y: torch.Tensor, adjust_factor=0.):
        if self.optimizer is None or self.loss_fn is None:
            self.init_learn()
        self.train()
        x, y = self.safe_cast_input(x), self.safe_cast_input(y, is_y=True)
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16 if self.device.type == "cpu" else torch.float16):
            outputs = self(x)
            loss: torch.Tensor = self.loss_fn(outputs, y.long() if y.ndim == 1 else y)
            loss += adjust_factor

        if self.device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.eval()
        return self.parse_output(outputs), loss

    def save(self, path: str):
        if not path.endswith(".pt"):
            path += ".pt"
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> "TorchModel":
        if not path.endswith(".pt"):
            path += ".pt"
        try:
            return torch.load(path)
        except Exception as e:
            return torch.load(path, map_location="cpu")

    def cuda(self):
        if self.device.type == "cuda":
            return self
        if not torch.cuda.is_available():
            return self
        self = super().cuda()
        if self.optimizer:
            self.optimizer_to_device("cuda")
        return self

    def cpu(self):
        if self.device.type == "cpu":
            return self
        self = super().cpu()
        if self.optimizer:
            self.optimizer_to_device("cpu")
        return self

    def learn_one(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        return self.learn(x, y)

    def predict(self, x: torch.Tensor):
        return self.parse_output(self.predict_proba(x))

    def predict_one(self, x: torch.Tensor):
        return self.predict(x)[0].item()

    def predict_proba(self, x: torch.Tensor):
        with torch.no_grad():
            y = self(self.safe_cast_input(x))
        if self.layers[-1].__class__.__name__ == "LogSoftmax":
            y = torch.exp(y)
        return y

    def predict_proba_one(self, x: torch.Tensor):
        return {i: v.item() for i, v in enumerate(self.predict_proba(x)[0])}

    def evaluate(self, x: torch.Tensor, y: torch.Tensor, metric=accuracy_score):
        return metric(self.safe_cast_input(y, is_y=True), self.predict(x))

    def load_state_dict_safe_pruned(self, state_dict: torch.ParameterDict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            if name in own_state:
                own_state[name].copy_(param)
            else:
                own_state[f"{name}_orig"].copy_(param)

    def copy_prune(self, other: "TorchModel"):
        for l1, l2 in zip(self.layers, other.layers):
            for param in ("weight", "bias"):
                if not hasattr(l2, f"{param}_mask"):
                    continue
                prune.custom_from_mask(l1, param, getattr(l2, f"{param}_mask").detach())

    def distance(self, other: "TorchModel", only_prunable=True, method="L2_NORM"):
        """Return the norm 2 difference between the two model parameters
        """
        if method == "L2_NORM":
            t_own = self.prunable if only_prunable else self.parameters()
            t_other = other.prunable if only_prunable else other.parameters()
            ret = 0
            for lay_lvm, lay_brm in zip(t_own, t_other):
                for k, v in lay_brm.named_parameters():
                    vv = getattr(lay_lvm, k)
                    ret += torch.sum((v - vv)**2).item()
            return ret
        raise ValueError()

    def benchmark_forward(self, X, times, warmup):
        scores = []
        with torch.no_grad(), tqdm(range(times + warmup)) as pbar:
            X = self.safe_cast_input(X)
            for _ in range(warmup):
                self(X)
                pbar.update()
            for _ in range(times):
                start = time.time_ns()
                self(X)
                end = time.time_ns()
                scores.append(len(X) / (end - start))
                pbar.update()
        return scores

    def optimizer_to_device(self, device):
        if not self.optimizer:
            return
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def fit(
            self, train_dataset: MyDataset, validation_dataset: MyDataset = None, batch_size: int = 256,
            max_epochs: int = 100, epochs_wo_improve: int = None, metric=accuracy_score, shuffle=True):
        if torch.cuda.is_available():
            self.cuda()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, persistent_workers=True)
        best_metric = 0
        current_epochs_wo_improve = 0
        history = {f"{metric.__name__}_train": [], "loss_train": []}
        best_state_dict: deepcopy(self.state_dict())
        if validation_dataset is not None:
            X_val = torch.from_numpy(validation_dataset.X.values).to(self.device)
            y_val = torch.from_numpy(validation_dataset.y.values).long()
            history[f"{metric.__name__}_validation"] = []
            history["loss_validation"] = []

        with tqdm(range(max_epochs)) as pbar:
            for i in range(max_epochs):
                training_loss = 0.0
                y_preds = torch.empty(0)
                y_true = torch.empty(0)

                for ii, (inputs, labels) in enumerate(train_loader):
                    y_true = torch.cat((y_true, labels), dim=0)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    predicted, loss = self.learn(inputs, labels)
                    training_loss += loss.item()
                    y_preds = torch.cat((y_preds, predicted.cpu()), dim=0)

                training_loss /= (ii + 1)

                metric_train = metric(y_true.cpu(), y_preds.cpu())
                history[f"{metric.__name__}_train"].append(metric_train)
                history["loss_train"].append(training_loss)

                if validation_dataset is not None:
                    y_raw_val = self.predict_proba(X_val).cpu()
                    validation_loss = self.loss_fn(y_raw_val, y_val).item()
                    metric_validation = metric(y_val, self.parse_output(y_raw_val))
                    history[f"{metric.__name__}_validation"].append(metric_validation)
                    history["loss_validation"].append(validation_loss)
                else:
                    metric_validation = metric_train

                pbar.set_description("Epoch {} {}".format(i + 1, {k: v[-1] for k, v in history.items()}), refresh=False)
                pbar.update()

                if metric_validation > best_metric:
                    current_epochs_wo_improve = 0
                    best_metric = metric_validation
                    best_state_dict = deepcopy(self.state_dict())
                else:
                    current_epochs_wo_improve += 1

                if current_epochs_wo_improve == epochs_wo_improve:
                    break
            self.load_state_dict(best_state_dict)

        self.cpu()
        return history

    def permanent_prune(self):
        if not prune.is_pruned(self):
            return
        for layer in self.prunable:
            if hasattr(layer, "weight_mask"):
                prune.remove(layer, "weight")
            if hasattr(layer, "bias_mask"):
                prune.remove(layer, "bias")

    def network_layers_sparsity(self: BaseModel, only_prunable=True):
        if prune.is_pruned(self):
            self = self.clone(init=False)
            self.permanent_prune()
        single = {}
        total = 0
        target = self.prunable if only_prunable else self.layers.children()
        for x in target:
            local = 0
            for p in x.parameters():
                local += float(torch.sum(p.data == 0) / p.data.nelement())
            single[x] = local
            total += local
        total /= len(single)
        return total, single


class Mlp(TorchModel):
    def __init__(
            self, in_features: list[str],
            hidden_units, hidden_layers, batch_norm=False, dropout_prob=0.0,
            activation="ReLU", final_activation="LogSoftmax", seed=True):
        super().__init__(seed=seed)

        act: torch.nn.ReLU = getattr(torch.nn, activation)
        final_act: torch.nn.LogSoftmax = getattr(torch.nn, final_activation)
        self.init_params = {
            "in_features": in_features,
            "hidden_units": hidden_units,
            "hidden_layers": hidden_layers,
            "batch_norm": batch_norm,
            "dropout_prob": dropout_prob,
            "activation": activation,
            "final_activation": final_activation,
            "seed": seed}

        self.layers = torch.nn.Sequential()

        if batch_norm:
            self.layers.add_module("input_batch", torch.nn.BatchNorm1d(len(in_features), dtype=torch.double))

        self.layers.add_module("input_layer", torch.nn.Linear(
            len(in_features), hidden_units, dtype=torch.double))
        self.layers.add_module("input_act", act())

        for i in range(hidden_layers):
            if batch_norm:
                self.layers.add_module(f"hidden_{i}_batch", torch.nn.BatchNorm1d(hidden_units, dtype=torch.double))
            if dropout_prob > 0.:
                self.layers.add_module(f"hidden_{i}_drop", torch.nn.Dropout(p=dropout_prob))
            self.layers.add_module(f"hidden_{i}_layer", torch.nn.Linear(
                hidden_units, hidden_units, dtype=torch.double))
            self.layers.add_module(f"hidden_{i}_act", act())

        self.layers.add_module("final_layer", torch.nn.Linear(hidden_units, 2, dtype=torch.double))
        self.layers.add_module("final_act", final_act(dim=1))

    def parse_output(self, y: torch.Tensor):
        return torch.argmax(y, 1)

    @property
    def prunable(self):
        return tuple(m for m in self.layers.children() if isinstance(m, torch.nn.Linear))
