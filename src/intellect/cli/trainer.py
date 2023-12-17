import os
from dataclasses import field
from importlib import import_module
from typing import Any

import pandas as pd
from pydantic.dataclasses import dataclass
from sklearn.metrics import get_scorer

from intellect.model import BaseModel, MyDataset
from intellect.utility import (argparse_config, compute_metric_percategory,
                               load, set_seed)


@dataclass
class Config:
    """Configuration class for using this script.
    """
    model: str
    package: str
    dataset_dir: str
    metric: str = field(default="accuracy")
    init_params: dict[str, Any] = field(default=None)
    train_params: dict[str, Any] = field(default=None)


def main():
    set_seed()

    parser, _ = argparse_config(Config)
    args = parser.parse_args()
    config: Config = load(args.config, convert_cls=Config)

    train_ds = MyDataset(os.path.join(config.dataset_dir, "train.csv"))
    validation_ds = MyDataset(os.path.join(config.dataset_dir, "validation.csv"))
    finetune_ds = MyDataset(os.path.join(config.dataset_dir, "finetune.csv"))
    test_ds = MyDataset(os.path.join(config.dataset_dir, "test.csv"))

    model: BaseModel = getattr(import_module(config.package), config.model)(train_ds.features, **config.init_params)
    scorer = get_scorer(config.metric)._score_func

    history = model.fit(train_ds, validation_ds, metric=scorer, shuffle=True, **config.train_params)
    model.save(os.path.join(config.dataset_dir, "model.pt"))
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(config.dataset_dir, "model_history.csv"))

    df_percategory = compute_percategory_on_datasets(model, scorer, train_ds, validation_ds, finetune_ds, test_ds)
    df_percategory.to_csv(os.path.join(config.dataset_dir, f"model_{config.metric}.csv"))


def compute_percategory_on_datasets(
        model: BaseModel, scorer: callable, train_ds: MyDataset, validation_ds: MyDataset, finetune_ds: MyDataset,
        test_ds: MyDataset):
    df_percategory = pd.DataFrame()
    for ds, name in zip((train_ds, validation_ds, finetune_ds, test_ds), ("train", "validation", "finetune", "test")):
        y_pred_tmp = model.predict(ds.X)
        tmp_percategory = compute_metric_percategory(ds.y.values, y_pred_tmp, ds._y, scorer=scorer, also_global=True)
        df_percategory = pd.concat((df_percategory, pd.DataFrame(tmp_percategory, index=[name])))
    return df_percategory


if __name__ == "__main__":
    main()
