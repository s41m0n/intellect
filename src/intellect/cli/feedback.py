import os
from dataclasses import field
from importlib import import_module
from typing import Any

import pandas as pd
from pydantic.dataclasses import dataclass
from sklearn.metrics import get_scorer

from intellect.feedback import feedback_learning
from intellect.model import BaseModel, MyDataset
from intellect.utility import (argparse_config, compute_metric_percategory,
                               load, set_seed)


@dataclass
class ModelConfig:
    model: str
    package: str
    store_path: str


@dataclass
class Config:
    """Configuration class for using this script.
    """
    teacher_config: ModelConfig
    student_config: ModelConfig
    dataset_dir: str
    algorithm: str
    batch_size: int
    drop_ratio: float
    categories: list[str] = field(default_factory=list)


def main():
    set_seed()

    parser, _ = argparse_config(Config)
    args = parser.parse_args()
    config: Config = load(args.config, convert_cls=Config)

    train_ds = MyDataset(os.path.join(config.dataset_dir, "train.csv"))
    validation_ds = MyDataset(os.path.join(config.dataset_dir, "validation.csv"))
    finetune_ds = MyDataset(os.path.join(config.dataset_dir, "finetune.csv"))
    test_ds = MyDataset(os.path.join(config.dataset_dir, "test.csv"))

    teacher_model: BaseModel = getattr(
        import_module(config.teacher_config.package),
        config.teacher_config.model).load(
        config.teacher_config.store_path)
    student_model: BaseModel = getattr(
        import_module(config.student_config.package),
        config.student_config.model).load(
        config.student_config.store_path)

    m, yp, yt, labels = feedback_learning(
        teacher_model, student_model, test_ds, config.algorithm, config.batch_size, metric=None,
        drop_ratio=config.drop_ratio, categories=config.categories)

    from sklearn.metrics import accuracy_score, f1_score
    print(f1_score(yt, yp), accuracy_score(yt, yp))


if __name__ == "__main__":
    main()
