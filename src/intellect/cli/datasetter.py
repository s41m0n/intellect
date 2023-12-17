import os
from dataclasses import field

import numpy as np
import pandas as pd
import psutil
from pydantic.dataclasses import dataclass
from sklearn.model_selection import train_test_split

from intellect.utility import (argparse_config, create_dir, load,
                               remove_non_ascii, set_seed)


class Composition(tuple):
    @property
    def train(self):
        return self[0]

    @property
    def validation(self):
        return self[1]

    @property
    def finetune(self):
        return self[2]

    @property
    def test(self):
        return self[3]


@dataclass
class Config:
    """Configuration class for using this script.
    """
    src_files: list[str]
    output_dir: str
    benign_labels: list[str]
    label_column: str

    balanced: bool = field(default=True)
    binary: bool = field(default=True)
    prune_constant: bool = field(default=True)
    std_threshold: float = field(default=0.)
    max_samples_ratio: float = field(default=1.)
    drop_columns: list[str] = field(default_factory=list)

    train_val_fine_test_default_ratios: Composition = field(default_factory=lambda: Composition([0.6, 0.1, 0.1, 0.2]))
    composition: dict[str, Composition] = field(default_factory=dict)

    def __post_init__(self):
        if sum(v for v in self.train_val_fine_test_default_ratios) != 1.:
            raise Exception("Default Ratios do not sum up to 1.")
        if self.composition is None:
            return
        for k in self.composition:
            self.composition[k] = Composition(self.composition[k])
        if any(len(v) != 4 for v in self.composition.values()):
            raise Exception("When providing a composition, 4 ratios must be given")
        if any(sum(y for y in v) > 1. for v in self.composition.values()):
            raise Exception("Components do not sum up to 1.")


def format_columns(df: pd.DataFrame, label_column: str):
    df = df.rename(columns={label_column: "Label", **{k: remove_non_ascii(k)
                                                      for k in df.columns.values}})
    df["Label"] = df["Label"].apply(lambda x: remove_non_ascii(x))
    df["Type"] = df["Label"]
    return df


def main():
    set_seed()
    parser, subparsers = argparse_config(Config)
    show_ds_parser = subparsers.add_parser(
        'show-dataset', help="Show the Labels distribution in the provided dataset files")
    show_ds_parser.add_argument("-p", "--path", help="Path to Dataset directories and/or files", nargs="+", type=str)
    show_ds_parser.add_argument("-l", "--label", help="Target Label to display", type=str)
    args = parser.parse_args()

    if args.command == "show-dataset":
        df: pd.DataFrame = load_files(expand_files(args.path), only_labels_str=args.label)
        print(df[args.label].value_counts())
        return

    config: Config = load(args.config, convert_cls=Config)
    config.src_files = expand_files(config.src_files)

    if not config.balanced or sum(
            os.path.getsize(x) for x in config.src_files) < 0.8 * (
            psutil.swap_memory().free + psutil.virtual_memory().available):
        df: pd.DataFrame = load_files(config.src_files)
    else:
        df: pd.DataFrame = load_files_low_memory_balanced(config.src_files, config.label_column, config.benign_labels)

    df = format_columns(df, config.label_column)

    df_wo_excluded = df.drop(config.drop_columns, axis=1, errors="ignore")
    print("Dropped", config.drop_columns)

    print(df_wo_excluded["Label"].value_counts())
    if config.binary:
        print("Changin labels to 0-1")
        df_wo_excluded["Label"] = (~df_wo_excluded["Label"].isin(config.benign_labels)).mul(1)
    print(df_wo_excluded["Label"].value_counts())

    df_numeric: pd.DataFrame = df_wo_excluded.apply(pd.to_numeric, errors='ignore')

    df_after_cat: pd.DataFrame = cols_to_categories(df_numeric)

    print("Shape before dropping NaN", df_after_cat.shape)
    df_wo_nan: pd.DataFrame = df_after_cat.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    print("Resulting shape", df_wo_nan.shape)

    if config.balanced:
        print("Balancing classes")
        df_wo_nan = balance_classes(df_wo_nan, config.benign_labels)

    if config.max_samples_ratio is not None:
        print("Taking only a fraction of data")
        g = df_wo_nan.groupby("Type")
        df_wo_nan = g.apply(lambda x: x.sample(frac=config.max_samples_ratio)).reset_index(drop=True)

    other_cols = df_wo_nan.columns.difference(["Label", "Type"])
    df_normalized = df_wo_nan[other_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    df_normalized[["Label", "Type"]] = df_wo_nan[["Label", "Type"]]

    if config.prune_constant:
        prevs = set(df_normalized.columns.values)
        vc = df_normalized.apply(lambda x: len(x.value_counts()))
        df_normalized = df_normalized.loc[:, vc[vc > 1].index.values]
        print("Remove constant removed the features", prevs - set(df_normalized.columns.values))

    if config.std_threshold is not None:
        prevs = set(df_normalized.columns.values)
        df_normalized.loc[:, df_normalized.apply(lambda x: x.std(
            numeric_only=True) if x.dtype.kind in 'iufc' else 1) > config.std_threshold]
        print("STD deviation removed the features", prevs - set(df_normalized.columns.values))

    train_ds, validation_ds, test_ds, finetune_ds = split_per_ratios(
        df_normalized, config.composition, config.train_val_fine_test_default_ratios)

    create_dir(config.output_dir, overwrite=False)
    for x, y in zip((train_ds, validation_ds, finetune_ds, test_ds),
                    ("train", "validation", "finetune", "test")):
        print(f"Storing {y=} with label value counts\n{x['Type'].value_counts()}", flush=True)
        x.index.name = "ID"
        x.to_csv(os.path.join(config.output_dir, f"{y}.csv"))


if __name__ == "__main__":
    main()


def stratified_split(df: pd.DataFrame, test_ratio: float, validation_ratio: float, finetune_ratio: float):
    big_train, big_test, _, _ = train_test_split(df, df["Label"],
                                                 test_size=test_ratio,
                                                 shuffle=True,
                                                 stratify=df["Type"])

    train_data, validation_data, _, _ = train_test_split(big_train, big_train["Label"],
                                                         test_size=validation_ratio,
                                                         shuffle=True,
                                                         stratify=big_train["Type"])

    test_data, finetune_data, _, _ = train_test_split(big_test, big_test["Label"],
                                                      test_size=finetune_ratio,
                                                      shuffle=True,
                                                      stratify=big_test["Type"])
    return train_data, validation_data, finetune_data, test_data


def balance_classes(df: pd.DataFrame, benign_labels: list[str]):
    d_malicious = df["Type"].value_counts()
    d_benign = {x: d_malicious.pop(x) for x in benign_labels}
    to_take = min(sum(x for _, x in d_benign.items()), sum(x for _, x in d_malicious.items()))
    g = df.groupby("Type")
    cats = {}
    for d in (d_benign, d_malicious):
        d_sorted = dict(sorted(d.items(), key=lambda x: x[1]))
        to_take_cat = to_take
        for i, (c, v) in enumerate(d_sorted.items()):
            per_cat = round(to_take_cat / (len(d) - i))
            cats[c] = min(per_cat, v)
            to_take_cat -= cats[c]
    ret = g.apply(lambda x: x.sample(cats[x.name]))
    return ret.reset_index(drop=True)


def cols_to_categories(df: pd.DataFrame):
    cat_columns = [col_name for col_name,
                   dtype in df.dtypes.items() if dtype == object and col_name not in ("Label", "Type")]

    if cat_columns:
        print("Converting following categorical to numerical", cat_columns)
        df[cat_columns] = df[cat_columns].astype('category')
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
        df[cat_columns] = df[cat_columns].astype('int')
    return df


def expand_files(files: list[str]):
    all_files = []
    for x in files:
        if os.path.isfile(x):
            all_files.append(x)
        else:
            [all_files.append(os.path.join(x, y)) for y in os.listdir(x) if y.endswith(".csv")]
    return all_files


def load_files(files: list[str], only_labels_str=None) -> pd.DataFrame:
    print("Loading", len(files), "files")
    df = pd.DataFrame()
    for x in files:
        df = pd.concat(
            (df, pd.read_csv(
                x, index_col=0 if not only_labels_str else None, skipinitialspace=True, usecols=[only_labels_str]
                if only_labels_str else None)),
            ignore_index=True)
    return df


def load_files_low_memory_balanced(files: list[str], label_col: str, benign_labels: list[str]) -> pd.DataFrame:
    df = pd.DataFrame()
    for x in files:
        tmp = pd.read_csv(x, usecols=[label_col], skipinitialspace=True)
        tmp["File"] = x
        tmp["Indexes"] = tmp.index.values
        tmp = format_columns(tmp, label_col)
        df = pd.concat((df, tmp))

    df = balance_classes(df, benign_labels)
    ret = pd.DataFrame()
    for x in files:
        tmp = dict.fromkeys((df[df["File"] == x]["Indexes"] + 1).tolist() + [0])
        tmp_csv = pd.read_csv(x, index_col=0, skipinitialspace=True, skiprows=lambda x: x not in tmp)
        ret = pd.concat((ret, tmp_csv), ignore_index=True)
    return ret


def split_per_ratios(df: pd.DataFrame, compositions: dict[str, Composition], default: Composition):
    dst_train, dst_validation, dst_finetune, dst_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for category in df["Type"].unique():
        composition: Composition = compositions.get(category, default)
        tmp = df[df["Type"] == category].copy(deep=True)
        n_samples = len(tmp)

        sampled_train = tmp.sample(n=int(composition.train * n_samples))
        dst_train = pd.concat((dst_train, sampled_train))
        tmp = tmp.drop(sampled_train.index)

        sampled_validation = tmp.sample(n=int(composition.validation * n_samples))
        dst_validation = pd.concat((dst_validation, sampled_validation))
        tmp = tmp.drop(sampled_validation.index)

        sampled_finetune = tmp.sample(n=int(composition.finetune * n_samples))
        dst_finetune = pd.concat((dst_finetune, sampled_finetune))
        tmp = tmp.drop(sampled_finetune.index)

        sampled_test = tmp.sample(n=int(composition.test * n_samples))
        dst_test = pd.concat((dst_test, sampled_test))
        tmp = tmp.drop(sampled_test.index)

    return dst_train, dst_validation, dst_finetune, dst_test
