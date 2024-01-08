import argparse
import inspect
import json
import logging
import pickle
import typing
from ctypes import Array, Structure
from ctypes import Union as CUnion
from ctypes import _Pointer, _SimpleCData
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from json import JSONEncoder
from pathlib import Path
from typing import TypeVar

import joblib
import numpy as np
import pandas as pd
import yaml

T = TypeVar('T')


def argparse_config(cls: T) -> T:
    """Function to parse arguments from argparse into configuration class T.

    Args:
        cls (T): the configuration class to be used.

    Returns:
        T: The configuration class populated.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help="Action to be performed")

    config_parser = subparsers.add_parser(
        'run-config', help="Run the script main function with the provided configuration",
        formatter_class=argparse.RawTextHelpFormatter, epilog="Additional Notes:\n" +
        "This script accepts either a <.json>, <.yml> or <.yaml> configuration file.\n" +
        f"Please refer to {cls} as reported below.\n\n" + json.dumps(
            get_annotations(cls),
            cls=CDataJSONEncoder, max_width=20))

    config_parser.add_argument("-c", "--config", type=str, help="Path to Config provided", required=True)
    return parser, subparsers


def recursive_find_file(path: str, endswith_condition: str) -> str:
    """Function to iteratively find all files with the given extension.

    Args:
        path (str): the start path
        endswith_condition (str): the extension condition

    Returns:
        list[str]:list of files
    """
    if path.endswith(endswith_condition):
        return [path]

    if not os.path.isdir(path):
        return []
    return [x for name in os.listdir(path) for x in recursive_find_file(os.path.join(path, name), endswith_condition)]


def dataclass_from_dict(klass: T, d: dict[object, object], raise_err=False) -> T:
    """_summary_

    Args:
        klass (T): the class to be populated
        d (dict[object, object]): the object to be used for populating the class
        raise_err (bool, optional): raise errors, if any, when true. Defaults to False.

    Returns:
        T: The class populated
    """
    fieldtypes = {f.name: f.type for f in fields(klass)}
    v = {}
    for f, fv in d.items():
        if is_dataclass(fieldtypes[f]):
            v[f] = dataclass_from_dict(fieldtypes[f], fv)
        else:
            v[f] = fieldtypes[f](fv)
    return klass(**v)


def get_annotations(klass: object) -> dict[str, object]:
    """Function to return class annotations. It works also for dataclasses.

    Args:
        klass (object): the class for which retrieve the annotations.

    Returns:
        dict[str, object]: the annotations
    """
    return {k: (v if not is_dataclass(v) else get_annotations(v)) for k, v in klass.__annotations__.items()}


class CDataJSONEncoder(JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

    def __init__(self, *args, max_width=120, max_items=15, container_types=(list, tuple, dict), **kwargs):
        # using this class without indentation is pointless
        if kwargs.get("indent") is None:
            kwargs.update({"indent": 2})
        super().__init__(*args, **kwargs)
        """Container datatypes include primitives or other containers."""
        self.container_types = container_types
        """Maximum width of a container that might be put on a single line."""
        self.max_width = max_width
        """Maximum number of items in container that might be put on single line."""
        self.max_items = max_items
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        o = self.default(o)

        if isinstance(o, (list, tuple)):
            if self._put_on_single_line(o):
                return "[" + ", ".join(self.encode(el) for el in o) + "]"
            else:
                self.indentation_level += 1
                output = [self.indent_str + self.encode(el) for el in o]
                self.indentation_level -= 1
                return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"
        elif isinstance(o, dict):
            if o:
                if self._put_on_single_line(o):
                    return "{ " + ", ".join(f"{self.encode(k)}: {self.encode(el)}" for k, el in o.items()) + " }"
                else:
                    self.indentation_level += 1
                    output = [
                        self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]
                    self.indentation_level -= 1
                    return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
            else:
                return "{}"
        elif isinstance(o, str):  # escape newlines
            o = o.replace("\n", "\\n")
            return f'"{o}"'
        else:
            return json.dumps(o)

    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)

    def _put_on_single_line(self, o):
        return self._primitives_only(o) and len(o) <= self.max_items and len(str(o)) - 2 <= self.max_width

    def _primitives_only(self, o: typing.Union[list, tuple, dict]):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.container_types) for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.container_types) for el in o.values())

    @property
    def indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " " * (self.indentation_level * self.indent)
        elif isinstance(self.indent, str):
            return self.indentation_level * self.indent
        else:
            raise ValueError(
                f"indent must either be of type int or str (is: {type(self.indent)})")

    def default(self, obj):
        if inspect.isclass(obj):
            return str(obj)

        if isinstance(obj, (Array, list)):
            return [self.default(e) for e in obj]

        if isinstance(obj, _Pointer):
            return self.default(obj.contents) if obj else None

        if isinstance(obj, _SimpleCData):
            return self.default(obj.value)

        if isinstance(obj, (bool, int, float, str)):
            return obj

        if obj is None:
            return obj

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, typing._GenericAlias):
            return str(obj)

        if isinstance(obj, (Structure, CUnion)):
            result = {}
            anonymous = getattr(obj, '_anonymous_', [])

            for key, _ in getattr(obj, '_fields_', []):
                value = getattr(obj, key)

                # private fields don't encode
                if key.startswith('_'):
                    continue

                if key in anonymous:
                    result.update(self.default(value))
                else:
                    result[key] = self.default(value)

            return result

        if is_dataclass(obj):
            if hasattr(obj, "to_json"):
                return obj.to_json()
            else:
                return {k.name: self.default(getattr(obj, k.name)) for k in fields(obj)}

        if isinstance(obj, dict):
            if obj and not isinstance(next(iter(obj), None), (int, float, str, bool)):
                return [{'key': self.default(k), 'value': self.default(v)} for k, v in obj.items()]
            else:
                return {k: self.default(v) for k, v in obj.items()}

        if isinstance(obj, tuple):
            if hasattr(obj, "_asdict"):
                return self.default(obj._asdict())
            else:
                return [self.default(e) for e in obj]

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        return JSONEncoder.default(self, obj)


def dump(x: object, filename: str, **kwargs):
    """Function to dump an object to file.

    Args:
        x (object): the object to be dump.
        filename (str): the file where to dump the object

    Returns:
        object: The output of the dump
    """
    if hasattr(x, "save"):
        return x.save(filename)

    if filename.endswith(".json"):
        with open(filename, "w") as fp:
            return json.dump(x, fp, cls=CDataJSONEncoder, **kwargs)

    if filename.endswith(".yaml") or filename.endswith(".yml"):
        with open(filename, "w") as fp:
            return yaml.safe_dump(x, fp, **kwargs)

    if filename.endswith(".csv"):
        return pd.DataFrame(x).to_csv(filename, **kwargs)

    if filename.endswith(".joblib"):
        return joblib.dump(x, filename, **kwargs)

    if filename.endswith(".pkl"):
        with open(filename, "wb") as fp:
            return pickle.dump(x, fp)

    if filename.endswith(".h5"):
        return x.to_hdf(filename, 'data', mode='w')

    filename += ".pkl"
    print("Dumping using fallback pickle method")
    with open(filename, "wb") as fp:
        return pickle.dump(x, fp)


def load(filename: str, convert_cls=None, **kwargs) -> object:
    """Function to load an object from a file.

    Args:
        filename (str): the file where the object is contained
        convert_cls (_type_, optional): optional class to which convert the
            loaded object. Defaults to None.

    Raises:
        ValueError: when no format is specified and/or implemented.

    Returns:
        object: the loaded object.
    """
    x = None
    if convert_cls and hasattr(convert_cls, "load"):
        return convert_cls.load(filename)

    if filename.endswith(".json"):
        with open(filename, "r") as fp:
            x = json.load(fp, **kwargs)
    elif filename.endswith(".yaml") or filename.endswith(".yml"):
        with open(filename, "r") as fp:
            x = yaml.safe_load(fp, **kwargs)
    elif filename.endswith(".csv"):
        x = pd.read_csv(filename, **kwargs)
    elif filename.endswith(".pkl"):
        with open(filename, "rb") as fp:
            x = pickle.load(filename)
    elif filename.endswith(".joblib"):
        x = joblib.load(filename, **kwargs)
    elif filename.endswith(".h5"):
        x = pd.read_hdf(filename, **kwargs)
    else:
        raise ValueError("Don't know load method")

    if convert_cls:
        x = dataclass_from_dict(convert_cls, x)
    return x


def create_dir(name: str):
    """Function to create a directory and, in case, backup the old one
    if already present.

    Args:
        name (str): name of the directory to be created
    """
    p = Path(name)
    if p.absolute() == Path.cwd():
        return
    if p.exists():
        p.rename(p.name + "_backup" + str(datetime.now()))
    p.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, filepath: str = None, log_level: int = logging.INFO) -> logging.Logger:
    """Function to create a logger, or return the existing one.

    Args:
        name (str): name of the logger
        filepath (str, optional): additional file where to log. Defaults to None.
        log_level (int, optional): level detail of logging. Defaults to logging.INFO.

    Returns:
        logging.Logger: the new logger, or the old one patched.
    """
    if name not in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        handlers = [logging.StreamHandler()]
    else:
        logger = logging.getLogger(name)
        handlers = logger.handlers
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if filepath and not any(isinstance(x, logging.FileHandler) for x in logger.handlers):
        handlers.append(logging.FileHandler(filepath, mode="w"))
    for handle in handlers:
        handle.setLevel(log_level)
        handle.setFormatter(formatter)
        logger.addHandler(handle)
    return logger
