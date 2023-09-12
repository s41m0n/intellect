import argparse
import functools
import os
from multiprocessing.pool import ThreadPool
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.common import create_dir, get_logger, set_seed

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def load_optional_partial_csv(path, dest=None, cols=None, index_col="ID"):
    if isinstance(path, tuple):
        path, dest = path
    return pd.read_csv(path, index_col=index_col, low_memory=False, usecols=cols, skipinitialspace=True,
                       encoding_errors='ignore', skiprows=None if dest is None else lambda x: x not in dest)


def df_to_sets(
        df: pd.DataFrame, test_size: float, finetune_size: float, validation_size: float, label_column, benign_label,
        columns_to_exclude) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.drop(columns_to_exclude, axis=1, errors='ignore')
    df.rename(columns={label_column: 'Label'}, inplace=True)
    label_column = "Label"

    df[label_column] = df[label_column].ne(benign_label).mul(1)
    df = df.apply(pd.to_numeric, errors='ignore')
    cat_columns = [col_name for col_name,
                   dtype in df.dtypes.items() if dtype == object and col_name != label_column]

    if cat_columns:
        logger.info(f"Convert to Categorical columns {cat_columns=}")
        df[cat_columns] = df[cat_columns].astype('category')
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    cols_to_norm = [x for x in df.columns if x != label_column]

    logger.info("Remove inf and nan samples")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how="any", inplace=True)
    logger.info(f"Normalize columns {cols_to_norm=} and replace NaN with zero (Sample legit, but min==max==value)")

    df[cols_to_norm] = df[cols_to_norm].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    # df.replace([np.nan], 0., inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    logger.info("Split in Train and Test")
    glob_x, glob_xt, _, _ = train_test_split(df, df[label_column],
                                             test_size=test_size,
                                             shuffle=True,
                                             stratify=df[label_column],
                                             random_state=int(os.environ.get("PYTHONHASHSEED", 0)))

    logger.info("Split Train in Train and Validation")
    glob_x_train, glob_x_val, _, _ = train_test_split(glob_x, glob_x[label_column],
                                                      test_size=validation_size,
                                                      shuffle=True,
                                                      stratify=glob_x[label_column],
                                                      random_state=int(os.environ.get("PYTHONHASHSEED", 0)))

    glob_xt_test, glob_xt_finetune, _, _ = train_test_split(glob_xt, glob_xt[label_column],
                                                            test_size=finetune_size,
                                                            shuffle=True,
                                                            stratify=glob_xt[label_column],
                                                            random_state=int(os.environ.get("PYTHONHASHSEED", 0)))

    return glob_x_train, glob_x_val, glob_xt_test, glob_xt_finetune


def pick_samples_from_files(files: List[str], label_column: str = 'Label',
                            benign_label: str = "Normal", balance: bool = True, cpus: int = None):
    if balance:
        dff = []
        prev_rounded = 0
        logger.info("Chosing a balanced number of samples")
        with ThreadPool(processes=cpus) as pool:
            for x in pool.imap(functools.partial(load_optional_partial_csv, dest=None, cols=[label_column]), files):
                dff.append(x)
                perc = round(len(dff) * 100 / len(files))
                if perc - 1 >= prev_rounded:
                    prev_rounded = perc
                    logger.info(f"Loading Labels at {perc=}%")

        lens = [len(x) for x in dff]
        df = pd.concat(dff)
        logger.info("Computing Indexes and quantity to take")

        df.reset_index(drop=True, inplace=True)
        df[label_column] = df[label_column].ne(benign_label).mul(1)
        ben_indexes = df[df[label_column] == 0].index.values.astype(int).tolist()
        mal_indexes = df[df[label_column] == 1].index.values.astype(int).tolist()
        minv = min(len(ben_indexes), len(mal_indexes))
        logger.info(
            f"Randomly choosing {minv=} out of {len(ben_indexes)=} Benign and {minv=} out of "
            f"{len(mal_indexes)=} Malicious samples from {len(files)=} files")
        res = sorted(np.random.choice(ben_indexes, minv, replace=False).tolist(
        ) + np.random.choice(mal_indexes, minv, replace=False).tolist())
        prev_len = 0
        for i in range(len(lens)):
            next_len = prev_len + lens[i]
            next_i = next((ii for ii, x in enumerate(res)
                          if x >= next_len), len(res))
            lens[i] = dict.fromkeys(
                [0] + [x - prev_len + 1 for x in res[:next_i]], None)
            res = res[next_i:]
            prev_len = next_len
    else:
        lens = [None for _ in files]

    logger.info("Loading chosen samples from files")
    ret = []
    prev_rounded = 0
    names = []
    with ThreadPool(processes=cpus) as pool:
        for i, x in enumerate(pool.imap(functools.partial(load_optional_partial_csv), zip(files, lens))):
            ret.append(x)
            names += [os.path.basename(files[i])] * len(x)
            perc = round(i * 100 / len(files))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info("Loading Samples at %f%%", perc)
    ret = pd.concat(ret)
    ret["original_ID"] = ret.index.values
    ret['original_file'] = names
    return ret.reset_index(drop=True)


def async_store(args):
    (df, store_path, label_col, benign_label) = args
    df: pd.DataFrame

    benign = len(df[df[label_col] == benign_label])
    malicious = len(df) - benign
    logger.info(
        f"Storing {os.path.basename(store_path)} with {benign} Benign and "
        f"{malicious} Malicious samples to file")
    df.to_csv(store_path)


def main(args):
    """Main function of the module.
    Given a certain directory of a dataset and a series of parameters,
    create the Train, Validation, and Test set. Features are normalised
    within the range [0,1], and the sets can be either balance or
    unbalance. Currently, despite keeping a column representing the
    name of different anomalied (i.e. Attack Types), the entire
    methodology is based on Binary Classification (0/1).
    For each dataset, few columns are dropped, depending on the type
    of the feature. For instance, IP addresses and L4 ports are
    not considered for the classification problem.
    """
    global logger
    set_seed()

    store_path = os.path.join(args.dataset, os.pardir, args.type)
    create_dir(store_path, overwrite=False)

    logger.info(f"Setting variable for dataset {args.dataset}")
    if 'CICIDS2017' in args.dataset:
        type_column = 'Label'
        benign_label = 'BENIGN'
        excluded_columns = ["Destination Port"]
        files = [os.path.join(args.dataset, x) for x in os.listdir(
            args.dataset) if x.endswith(".csv")]
    elif "CICIDS2019" in args.dataset:
        type_column = 'Label'
        benign_label = 'BENIGN'
        excluded_columns = ["Unnamed: 0", "Flow ID", "Source IP", "Source Port",
                            "Destination IP", "Destination Port", "Timestamp", "SimillarHTTP"]
        files = [os.path.join(args.dataset, x) for x in os.listdir(
            args.dataset) if x.endswith(".csv")]
    elif "CICIOT2023" in args.dataset:
        type_column = 'label'
        benign_label = 'BenignTraffic'
        excluded_columns = ['Protocol Type', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
                            'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']
        files = [os.path.join(args.dataset, x) for x in os.listdir(
            args.dataset) if x.endswith(".csv")]
    elif "EDGE2022" in args.dataset:
        type_column = 'Attack_type'
        benign_label = "Normal"
        excluded_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                            "tcp.srcport", "tcp.dstport", "udp.port", "http.tls_port",
                            "tcp.options", "http.file_data", "tcp.payload", "mqtt.msg",
                            "http.request.uri.query", "http.request.full_uri",
                            "icmp.transmit_timestamp", "Attack_label"]
        files = [os.path.join(args.dataset, x) for x in os.listdir(
            args.dataset) if x.endswith(".csv")]
    elif "ICS-D1" in args.dataset:
        type_column = 'marker'
        benign_label = "Natural"
        excluded_columns = []
        files = [os.path.join(args.dataset, x)
                 for x in os.listdir(args.dataset) if x.endswith(".csv")]
    elif "ICS-D2" in args.dataset:
        type_column = 'Label'
        benign_label = "Good"
        excluded_columns = []
        files = [os.path.join(args.dataset, y, x)
                 for y in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, y))
                 for x in os.listdir(os.path.join(args.dataset, y)) if x.endswith(".csv")]
    elif "ICS-D3" in args.dataset:
        type_column = 'result'
        benign_label = 0
        excluded_columns = ['time']
        files = [os.path.join(args.dataset, x)
                 for x in os.listdir(args.dataset) if x.endswith(".csv")]
    else:
        raise ValueError(args.dataset)

    logger.info(f"{len(files)=} {len(excluded_columns)=} {benign_label=} {type_column=}")
    df: pd.DataFrame = pick_samples_from_files(files,
                                               label_column=type_column,
                                               benign_label=benign_label,
                                               balance=args.type == "balanced",
                                               cpus=args.cpu)
    excluded_columns += ["original_ID", "original_file"]

    logger.info("Splitting dataset into train validation and test")
    train, val, test, finetune = df_to_sets(df.copy(deep=True), args.test, args.finetune, args.validation,
                                            type_column, benign_label, excluded_columns)

    prev_rounded = 0
    tasks = [(x, os.path.join(store_path, f"{y}.csv"), z, k) for x, y, z, k in zip([train, val, test, finetune, df], [
        "train", "validation", "test", "finetune", "original"], ['Label'] * 4 + [type_column], [0] * 4 + [benign_label])]
    for x in [train, val, test, finetune, df]:
        x.index.name = "ID"
    logger.info(f"Launching storing to file {len(tasks)=} tasks")
    with ThreadPool(processes=args.cpu) as pool:
        for i, _ in enumerate(pool.imap(async_store, tasks)):
            perc = round(i * 100 / len(tasks))
            if perc - 1 >= prev_rounded:
                prev_rounded = perc
                logger.info(f"Storing sets at {perc=}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset', help='path to the dataset directory', type=str)
    parser.add_argument(
        'type', help='balanced or unbalanced dataset', type=str, choices=["balanced", "unbalanced"])
    parser.add_argument(
        '-f', '--finetune', help='test data used to finetune/explorations/ranking', type=float, default=0.5)
    parser.add_argument(
        '-t', '--test', help='test size', type=float, default=0.3)
    parser.add_argument(
        '-v', '--validation', help='train data used for validation', type=float, default=0.2)
    parser.add_argument(
        '-c', '--cpu', help='number of cpu cores to assign', type=int, default=round(0.7 * os.cpu_count()))
    args = parser.parse_args()
    args.dataset = os.path.realpath(os.path.normpath(args.dataset))
    main(args)
