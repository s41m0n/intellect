import re
from copy import deepcopy
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from .io import load


class ProblemType(Enum):
    """Enumeration to select the specific type of the problem"""
    BINARY = 0
    MULTILABEL = 1
    MULTIOUTPUT = 2


class FeatureAvailability(Enum):
    """Enumeration to select Availability of features in continuous learning"""
    bilateral = 0
    none = 1
    client = 2
    oracle = 3


class InputForLearn(Enum):
    """Enumeration to select the input type a model should use while doing continuous learning"""
    client = 0
    oracle = 1


class ContinuouLearningAlgorithm(Enum):
    """Enumeration to select the continuous learning algorithm type"""
    none = 0
    knowledge_distillation = 1
    ground_inferred = 3
    ground_truth = 5


class Dataset:
    """_summary_

    Args:
        file (str, optional): name of the file to be loaded. Defaults to None.
        data (pd.DataFrame, optional): data already loaded. Defaults to None.
        shuffle (bool, optional): shuffle data if true. Defaults to True.
        label (str | list, optional): label column in the dataset or as a list. Defaults to "Label".
        label_type (str, optional): column with categories in the dataset. Defaults to "Type".

    Attributes:
        X (pd.Dataframe): data of the dataset
        y (pd.Series | pd.DataFrame): label target
        _y (pd.Series): label categories
    """

    def __init__(self, file: str = None, data: pd.DataFrame = None, shuffle: bool = True,
                 label: str | list = "Label", label_type: str = "Type", **kwargs):
        self.X: pd.DataFrame
        self.y: pd.Series | pd.DataFrame
        self._y: pd.Series
        if file is not None:
            self.X: pd.DataFrame = load(file, **kwargs)
        else:
            self.X: pd.DataFrame = data
        if isinstance(label, str):
            self.y = self.X.pop(label)
        else:
            self.y = label
        self.X.reset_index(drop=True, inplace=True)
        if shuffle:
            self.X = self.X.sample(frac=1)
            self.y = self.y[self.X.index.values]
        if isinstance(label_type, str):
            self._y = self.X.pop(label_type)
        else:
            self._y = label_type

    def __len__(self) -> int:
        """Function to return the length of the dataset.

        Returns:
            int: length of the dataset
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[pd.Series, pd.Series | np.ndarray]:
        """Function to return the nth row in the dataset by looking at the position.

        Args:
            idx (int): row position (different than index)

        Returns:
            tuple[pd.Series, pd.Series | np.ndarray]: tuple containing X and y value
        """
        y = self.y.iloc[idx]
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        return (self.X.iloc[idx].values, y)

    def join(self, other: "Dataset") -> "Dataset":
        """Function to join two datasets.

        Args:
            other (Dataset): dataset to append to the current one

        Returns:
            Dataset: the resulting dataset
        """
        ds_tmp = self.clone()
        ds_tmp.X = pd.concat((ds_tmp.X, other.X), axis=0, ignore_index=True)
        ds_tmp.y = pd.concat((ds_tmp.y, other.y), axis=0, ignore_index=True)
        ds_tmp._y = pd.concat((ds_tmp._y, other._y), axis=0, ignore_index=True)
        return ds_tmp

    def filter_features(
            self, keep_features: list[str],
            default: object = None, get_idx: bool = False) -> Union["Dataset", list[int]]:
        """Function to return a new copy of the dataset with only the provided features.
        The remaining ones can be either completely removed from the dataset (default=None)
        or set to a default value. When get_idx is true, return only the indexes of the provided features/columns.

        Args:
            keep_features (list[str]): list of features to keep
            default (object, optional): default value to be set. Defaults to None.
            get_idx (bool, optional): true when returnin only the feature positions. Defaults to False.

        Returns:
            Dataset | list[int]: the new dataset or just the list of feature positions
        """
        if get_idx:
            return [i for i, v in enumerate(self.X.columns.values) if v in keep_features]

        ds_tmp: "Dataset" = self.clone()
        if default is None:
            ds_tmp.X = ds_tmp.X[ds_tmp.X.columns.intersection(keep_features)]
        else:
            ds_tmp.X.loc[:, ~ds_tmp.X.columns.isin(keep_features)] = default

        return ds_tmp

    def filter_categories(self, keep_categories: list[str]) -> "Dataset":
        """Function to return a new copy of the dataset with only the provided categories.

        Args:
            keep_categories (list[str]): categories to be kept

        Returns:
            Dataset: new dataset with only the provided categories
        """
        ds_tmp = self.clone()
        indexes = ds_tmp._y[ds_tmp._y.isin(keep_categories)].index.values
        ds_tmp.X = ds_tmp.X.loc[indexes]
        ds_tmp.y = ds_tmp.y.loc[indexes]
        ds_tmp._y = ds_tmp._y.loc[indexes]
        return ds_tmp

    def sample(self, frac: float, by_category: bool = True) -> "Dataset":
        """Function to return a new dataset with only a portion of samples.

        Args:
            frac (float): fraction of samples to keep
            by_category (bool, optional): true if preserve ratio between categories.
                Defaults to True.

        Raises:
            ValueError: if the fraction provided is not between 0 and 1

        Returns:
            Dataset: the new dataset with only the sampled data
        """
        ds_tmp = self.clone()

        if frac > 1. or frac <= 0.:
            raise ValueError("Wrong")

        if by_category:
            indexes = self._y.groupby(self._y).sample(frac=frac).index.values
        else:
            indexes = ds_tmp.X.sample(frac=frac).index.values

        ds_tmp.X = ds_tmp.X.loc[indexes]
        ds_tmp.y = ds_tmp.y.loc[indexes]
        ds_tmp._y = ds_tmp._y.loc[indexes]
        return ds_tmp

    def filter_indexes(self, remove_indexes: list[object]) -> "Dataset":
        """Function to return a new version of the dataset without the provided indexes.

        Args:
            indexes (list[object]): list of indexes to remove

        Returns:
            Dataset: the resulting dataset without the indexes
        """
        ds_tmp = self.clone()
        ds_tmp.X.drop(index=remove_indexes, inplace=True)
        ds_tmp.y.drop(index=remove_indexes, inplace=True)
        ds_tmp._y.drop(index=remove_indexes, inplace=True)
        return ds_tmp

    def clone(self) -> "Dataset":
        """Function to clone the current dataset.

        Returns:
            Dataset: a new object with the same data
        """
        return deepcopy(self)

    @property
    def categories(self) -> list[str]:
        """Function to return the list of categories in the dataset.

        Returns:
            list[str]: list with category names
        """
        return self._y.unique().tolist()

    @property
    def features(self) -> list[str]:
        """Function to return the list of features in the dataset.

        Returns:
            list[str]: list with feature names
        """
        return self.X.columns.values.tolist()

    @property
    def n_classes(self) -> int:
        """Function to return the number of predict classes in the dataset.

        Returns:
            int: number of classes
        """
        if isinstance(self.y, pd.Series):
            return self.y.nunique()
        tmp = np.argmax(self.y.to_numpy(), axis=-1)
        return len(np.unique(tmp))

    @property
    def n_categories(self) -> int:
        """Function to return the number of categories in the dataset.

        Returns:
            int: number of categories
        """
        return self._y.nunique()

    @property
    def n_samples(self) -> int:
        """Function to return the number of samples in the dataset.

        Returns:
            int: number of samples
        """
        return len(self)

    @property
    def n_features(self) -> int:
        """Property to return the number of features (columns) in the dataset.

        Returns:
            int: the number of features
        """
        return self.X.shape[1]


def portions_from_data(data: pd.DataFrame | str, ptype: ProblemType = ProblemType.BINARY,
                       benign_labels: list[str] = None, ratios: list[float] = (0.7, 0.1, 0.2)) -> list[Dataset]:
    """Function to split the given dataframe into the provided portions of specified sizes
    and return them as a Dataset class.

    Args:
        data (pd.DataFrame | str): the dataframe to be split
        ptype (ProblemType, optional): type of the problem to convert labels.
            Defaults to ProblemType.BINARY.
        benign_labels (list[str], optional): list of benign labels required for binary.
            Defaults to None.
        ratios (list[float], optional): list of ratios for each portion. Defaults to (0.7, 0.1, 0.2).

    Raises:
        ValueError: when benign labels not provided in a binary classification problems

    Returns:
        list[Dataset]: list of portions converted into a Dataset class
    """
    if isinstance(data, pd.DataFrame):
        data = data.copy()
    else:
        data = load(data)

    label = data.pop("Label")
    data = data.astype("float").apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))

    data["Type"] = label

    if ptype.value == ProblemType.BINARY.value:
        if not benign_labels:
            raise ValueError("Specifiy which labels are benign")
        label = pd.Series(label.apply(lambda x: x not in benign_labels).astype("int"))
    elif ptype.value == ProblemType.MULTILABEL.value:
        label = pd.Series(pd.factorize(label)[0])
    elif ptype.value == ProblemType.MULTIOUTPUT.value:
        label = pd.DataFrame(LabelBinarizer().fit_transform(label), index=data.index)

    ret = []
    data["Indexes"] = data.index.values

    for x in split_per_ratios(data, ratios, label_col="Type"):
        y = label.loc[x.pop("Indexes").values].reset_index(drop=True)
        ret.append(Dataset(data=x, shuffle=True, label=y, label_type="Type"))
    return tuple(ret)


def remove_below_std_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Function to remove columns from a dataframe with a std below a certain threshold.

    Args:
        df (pd.DataFrame): the dataset to be parsed
        threshold (float): the acceptance threshold

    Returns:
        pd.DataFrame: the dataframe without columns with std below threshold
    """
    return df.loc[:, df.apply(lambda x: x.std(
        numeric_only=True) if x.dtype.kind in 'iufc' else 1) > threshold]


def remove_non_ascii(text: str) -> str:
    """Function to remove non-ascii characters from text

    Args:
        text (str): text to be formatted

    Returns:
        str: the text provided without non-ascii characters
    """
    return ' '.join(re.sub(r'[^\x00-\x7F]', '', text).split())


def balance_classes(df: pd.DataFrame, benign_labels: list[str]) -> pd.DataFrame:
    """Function to balance samples among benign and malicious ones. 

    Args:
        df (pd.DataFrame): the dataframe of origin
        benign_labels (list[str]): list of labels to be considered as benign

    Returns:
        pd.DataFrame: the result of the balancing
    """
    d_malicious = df["Label"].value_counts()
    d_benign = {x: d_malicious.pop(x) for x in benign_labels}
    to_take = min(sum(x for _, x in d_benign.items()), sum(x for _, x in d_malicious.items()))
    g = df.groupby("Label")
    cats = {}
    for d in (d_benign, d_malicious):
        d_sorted = dict(sorted(d.items(), key=lambda x: x[1]))
        to_take_cat = to_take
        for i, (c, v) in enumerate(d_sorted.items()):
            per_cat = round(to_take_cat / (len(d) - i))
            cats[c] = min(per_cat, v)
            to_take_cat -= cats[c]
    return g.apply(lambda x: x.sample(cats[x.name])).reset_index(drop=True)


def cols_to_categories(df: pd.DataFrame, label_col="Label") -> pd.DataFrame:
    """Function to convert non-numeric data into categorical.

    Args:
        df (pd.DataFrame): dataframe to convert

    Returns:
        pd.DataFrame: the converted dataframe
    """
    cat_columns = [col_name for col_name,
                   dtype in df.dtypes.items() if dtype.kind in "biufc" and col_name != label_col]

    if cat_columns:
        print("Converting following categorical to numerical", cat_columns)
        df[cat_columns] = df[cat_columns].astype('category')
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
        df[cat_columns] = df[cat_columns].astype('int')
    return df


def load_dataframes(files: list[str], only_labels_str=None) -> dict[str, pd.DataFrame]:
    """Function to load data dataframes from list of files

    Args:
        files (list[str]): list of files
        only_labels_str (_type_, optional): string with the name of the label columns
            when loading only labels. Defaults to None.

    Returns:
        dict[str, pd.DataFrame]: dictionary with name of files and its DataFrame associated
    """
    return {x: load(x, skipinitialspace=True,
                    index_col=0 if not only_labels_str else None,
                    usecols=[only_labels_str] if only_labels_str else None)
            for x in files}


def load_dataframe_low_memory_balanced(files: list[str], label_col: str, benign_labels: list[str]) -> pd.DataFrame:
    """Function to load balanced samples between categories from list of files loading only labels into memory.

    Args:
        files (list[str]): list of files to be loaded
        label_col (str): label column 
        benign_labels (list[str]): list of labels to be treated as benign

    Returns:
        pd.DataFrame: _description_
    """
    ret = []
    for x in files:
        tmp = load(x, usecols=[label_col], skipinitialspace=True)
        tmp["File"] = x
        tmp["Indexes"] = tmp.index.values
        ret.append(tmp)
    df = pd.concat(ret)

    df = balance_classes(df, benign_labels)
    ret = []
    for x in files:
        tmp = dict.fromkeys((df[df["File"] == x]["Indexes"] + 1).tolist() + [0])
        ret.append(load(x, index_col=0, skipinitialspace=True, skiprows=lambda x: x not in tmp))
    return pd.concat(ret, ignore_index=True).drop(columns=["File", "Indexes"])


def split_per_ratios(df: pd.DataFrame, default: list[float],
                     label_col: str = "Label", compositions: dict[str, list[str]] = {}) -> list[pd.DataFrame]:
    """Function to split a dataset into the given portions, keeping proportions between categories.

    Args:
        df (pd.DataFrame): the dataset to be split
        default (list[float]): default composition if not available in compositions dictionary.
        label_col (str, optional): column to be used as label for category. Defaults to "Label".
        compositions (dict[str, list[str]], optional): dict with portions for each category. Defaults to {}.

    Returns:
        list[pd.DataFrame]: list with the various portions
    """
    ret = list([] for _ in range(len(default)))

    for category in df[label_col].unique():
        composition: tuple = compositions.get(category, default)
        tmp = df[df[label_col] == category].sample(frac=1.)
        composition = (np.cumsum(composition) * len(tmp)).astype(int)
        portions = np.split(tmp, composition, axis=0)
        [ret[i].append(portions[i]) for i in range(len(default))]
    return [pd.concat(ret[i], axis=0, ignore_index=True) for i in range(len(ret))]


def format_dataframe_columns_names(df: pd.DataFrame) -> pd.DataFrame:
    """Function to format column names removing non-ascii characters

    Args:
        df (pd.DataFrame): dataframe of interest

    Returns:
        pd.DataFrame: the resulting dataframe
    """
    return df.rename(columns={k: remove_non_ascii(k) for k in df.columns.values})


def format_dataframe_columns_values(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Function to remove non-ascii characters from dataframe column value

    Args:
        df (pd.DataFrame): dataframe of interest
        target_col (str): the target column to format

    Returns:
        pd.DataFrame: the resulting dataframe
    """
    return df.copy()[target_col].apply(lambda x: remove_non_ascii(x))


def drop_nan_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Function to drop entire rows with nan or infinite values from a dataframe.

    Args:
        df (pd.DataFrame): dataframe of interest

    Returns:
        pd.DataFrame: the dataframe containing only valid rows
    """
    return df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Function to remove columns with constant value

    Args:
        df (pd.DataFrame): the dataframe of interest

    Returns:
        pd.DataFrame: the resulting dataframe without constant columns
    """
    vc = df.apply(lambda x: len(x.value_counts()))
    return df.loc[:, vc[vc > 1].index.values]


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Function to remove columns with duplicate name and/or values.

    Args:
        df (pd.DataFrame): dataframe to be parsed

    Returns:
        pd.DataFrame: the resulting dataframe with unique columns
    """
    df = df.iloc[:, ~df.columns.duplicated()]
    return df[df.describe(include="all").T.drop_duplicates().T.columns]


def dataset_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Function to convert dataframe into numeric ignoring exceptions.

    Args:
        df (pd.DataFrame): the dataframe to be converted

    Returns:
        pd.DataFrame: the converted dataframe.
    """
    return df.apply(pd.to_numeric, errors='ignore')


def coerc_dataframe_to_type(df: pd.DataFrame, dtype: str) -> pd.DataFrame:
    """Function to cast dataframe to the provided dtype

    Args:
        df (pd.DataFrame): dataframe to be converted
        dtype (str): dtype wanted

    Returns:
        pd.DataFrame: the converted dataframe
    """
    return df.astype(dtype)
