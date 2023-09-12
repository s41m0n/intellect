"""
Utility module containing common functions used by other scripts.
"""
import inspect
import io
import json
import logging
import os
import random
import shutil
import sys
import typing
from ctypes import Array, Structure
from ctypes import Union as CUnion
from ctypes import _Pointer, _SimpleCData
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from json import JSONEncoder

import numpy as np


def set_seed():
    """Function for setting the seed"""
    seed = int(os.environ.get("PYTHONHASHSEED", 0))

    np.random.seed(seed)
    random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass


def create_dir(name: str, overwrite=False):
    """Function to create a directory and, in case, overwrite or
    backup the old one if already present. If already exists,
    the old one is backed up.

    Args:
        name (str): name of the directory to be created
    """
    try:
        os.makedirs(name)
    except FileExistsError:
        if overwrite is None:
            pass
        elif overwrite is False:
            os.rename(name, name + '_backup_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
            os.makedirs(name)
        elif overwrite is True:
            shutil.rmtree(name)
            os.makedirs(name)


def load_json_data(path: str) -> typing.Dict:
    """Function to load json file into a dictionary

    Args:
        path (str): path to the file

    Returns:
        typing.Dict: the loaded dictionary
    """
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)


def dump_json_data(data: typing.Any, path: str = None) -> str:
    """Function to dump dictionary or dataclass into json file
    with the custom encoder.

    Args:
        data (typing.Any): the data to be dumped.
        path (str, optional): path to file if also wants to print. Defaults to None.

    Returns:
        str: dumped data structure as a string
    """
    ret = json.dumps(data, indent=2, cls=CDataJSONEncoder)
    if path is not None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(ret)
    return ret


def safe_division(numerator: typing.Any,
                  denominator: typing.Any, default: typing.Any = None) -> float:
    """Function to perform a safe division, meaning no exceptions are thrown
    in case of a division by 0 or infinite number

    Args:
        numerator (typing.Any): the numerator of the division
        denominator (typing.Any): the denominator of the division.
        default (typing.Any, optional): default value to be returned
            in case of errors. Defaults to None.

    Returns:
        float: result of the division
    """

    ret = np.divide(numerator, denominator)
    if default is not None:
        return np.nan_to_num(ret, copy=False, nan=default, posinf=default, neginf=default)
    return ret


def get_logger(name: str, filepath: str = None, log_level: int = logging.INFO) -> logging.Logger:
    """Function to create a logger, or return the existing one"""
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


class CDataJSONEncoder(JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    MAX_WIDTH = 120
    """Maximum width of a container that might be put on a single line."""

    MAX_ITEMS = 15
    """Maximum number of items in container that might be put on single line."""

    def __init__(self, *args, **kwargs):
        # using this class without indentation is pointless
        if kwargs.get("indent") is None:
            kwargs.update({"indent": 1})
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o, *_):
        """Encode JSON object *o* with respect to single line lists."""
        o = self.default(o)

        if isinstance(o, (list, tuple)):
            if self._put_on_single_line(o):
                return "[" + ", ".join(self.encode(el) for el in o) + "]"
            self.indentation_level += 1
            output = [self.indent_str + self.encode(el) for el in o]
            self.indentation_level -= 1
            return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"

        if isinstance(o, dict):
            if o:
                if self._put_on_single_line(o):
                    return "{ " + ", ".join(f"{self.encode(k)}: {self.encode(el)}" for k, el in o.items()) + " }"
                self.indentation_level += 1
                output = [
                    self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]
                self.indentation_level -= 1
                return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
            return "{}"

        if isinstance(o, str):  # escape newlines
            o = o.replace("\n", "\\n")
            return f'"{o}"'
        return json.dumps(o)

    def iterencode(self, o, _one_shot: bool = False):
        """Required to also work with `json.dump`."""
        return self.encode(o, _one_shot)

    def _put_on_single_line(self, obj):
        return self._primitives_only(obj) and len(obj) <= self.MAX_ITEMS and len(str(obj)) - 2 <= self.MAX_WIDTH

    def _primitives_only(self, obj: typing.Union[list, tuple, dict]):
        if isinstance(obj, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in obj)

        if isinstance(obj, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in obj.values())

        raise ValueError(f"No primitive for {obj}")

    @ property
    def indent_str(self) -> str:
        """Indent the string according to the indentation level set"""

        if isinstance(self.indent, int):
            return " " * (self.indentation_level * self.indent)

        if isinstance(self.indent, str):
            return self.indentation_level * self.indent

        raise ValueError(
            f"indent must either be of type int or str (is: {type(self.indent)})")

    def default(self, o):
        if inspect.isclass(o):
            return o.__name__

        if isinstance(o, (Array, list, set)):
            return [self.default(e) for e in o]

        if isinstance(o, _Pointer):
            return self.default(o.contents) if o else None

        if isinstance(o, _SimpleCData):
            return self.default(o.value)

        if isinstance(o, (bool, int, float, str)):
            return o

        if o is None:
            return o

        if isinstance(o, Enum):
            return o.value

        if isinstance(o, (Structure, CUnion)):
            result = {}
            anonymous = getattr(o, '_anonymous_', [])

            for key, *_ in getattr(o, '_fields_', []):
                value = getattr(o, key)

                # private fields don't encode
                if key.startswith('_'):
                    continue

                if key in anonymous:
                    result.update(self.default(value))
                else:
                    result[key] = self.default(value)

            return result

        if is_dataclass(o):
            if hasattr(o, "to_json"):
                return o.to_json()
            return {k.name: self.default(getattr(o, k.name)) for k in fields(o)}

        if isinstance(o, dict):
            if o and not isinstance(next(iter(o), None), (int, float, str, bool)):
                return [{'key': self.default(k), 'value': self.default(v)} for k, v in o.items()]
            return {k: self.default(v) for k, v in o.items()}

        if isinstance(o, tuple):
            if hasattr(o, "_asdict"):
                return self.default(o._asdict())
            return [self.default(e) for e in o]

        if isinstance(o, np.ndarray):
            return o.tolist()

        if isinstance(o, np.integer):
            return int(o)

        if isinstance(o, np.floating):
            return float(o)

        try:
            return JSONEncoder.default(self, o)
        except TypeError:
            try:
                return o.__dict__
            except AttributeError:
                return str(o)


def deep_get_size(obj, seen=None):
    """Recursively finds size of objects"""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([deep_get_size(v, seen) for v in obj.values()])
        size += sum([deep_get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += deep_get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (io.TextIOBase, io.BufferedIOBase, io.RawIOBase,
                                                           io.IOBase, str, bytes, bytearray)):
        size += sum([deep_get_size(i, seen) for i in obj])

    return size
