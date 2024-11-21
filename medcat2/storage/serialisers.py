from enum import Enum, auto
from typing import Union, Type, Any
import os
from abc import ABC, abstractmethod
from importlib import import_module

import dill as _dill

from medcat2.storage.serialisables import Serialisable
from medcat2.storage.serialisables import get_all_serialisable_members
from medcat2.storage.schema import load_schema, save_schema
from medcat2.storage.schema import DEFAULT_SCHEMA_FILE, IllegalSchemaException


class Serialiser(ABC):
    RAW_FILE = 'raw_dict.dat'

    @abstractmethod
    def serialise(self, raw_parts: dict[str, Any], target_file: str) -> None:
        pass

    @abstractmethod
    def deserialise(self, target_file: str) -> dict[str, Any]:
        pass

    def serialise_all(self, obj: Serialisable, target_folder: str) -> None:
        # attr2file_and_cls: dict[str, tuple[str, str]] = {}
        ser_parts, raw_parts = get_all_serialisable_members(obj)
        for part, name in ser_parts:
            basename = part.get_save_name(name)
            part_folder = os.path.join(target_folder, basename)
            if os.path.exists(part_folder):
                raise IllegalSchemaException(
                    f"File already exists: {part_folder}. Unable to overwrite")
            os.mkdir(part_folder)
            # if name in attr2file_and_cls:
            #     raise IllegalSchemaException(f"Multiple values for {name}")
            # recursive
            self.serialise_all(part, part_folder)
            # attr2file_and_cls[name] = (basename, _cls2path(type(part)))
        if raw_parts:
            raw_file = os.path.join(target_folder, self.RAW_FILE)
            self.serialise(raw_parts, raw_file)
        schema_path = os.path.join(target_folder, DEFAULT_SCHEMA_FILE)
        save_schema(schema_path, obj.__class__)

    def deserialise_all(self, folder_path: str) -> Serialisable:
        schema_path = os.path.join(folder_path, DEFAULT_SCHEMA_FILE)
        cls_path = load_schema(schema_path)
        module_path, cls_name = cls_path.rsplit('.', 1)
        module = import_module(module_path)
        cls: Type = getattr(module, cls_name)
        init_kwargs: dict[str, Serialisable] = {}
        for part_name in os.listdir(folder_path):
            if part_name == DEFAULT_SCHEMA_FILE or part_name == self.RAW_FILE:
                continue
            part_path = os.path.join(folder_path, part_name)
            if not os.path.isdir(part_path):
                continue
            part = self.deserialise_all(part_path)
            init_kwargs[part_name] = part
        obj = cls(**init_kwargs)
        raw_file = os.path.join(folder_path, self.RAW_FILE)
        raw_parts = self.deserialise(raw_file)
        for attr_name, attr in raw_parts.items():
            setattr(obj, attr_name, attr)
        return obj


class DillSerialiser(Serialiser):

    def serialise(self, raw_parts: dict[str, Any], target_file: str) -> None:
        with open(target_file, 'wb') as f:
            _dill.dump(raw_parts, f)

    def deserialise(self, target_file: str) -> dict[str, Any]:
        with open(target_file, 'rb') as f:
            return _dill.load(f)


class AvailableSerialisers(Enum):
    dill = auto()


def get_serialiser(serialiser_type: Union[str, AvailableSerialisers]
                   ) -> Serialiser:
    if isinstance(serialiser_type, str):
        serialiser_type = AvailableSerialisers[serialiser_type.lower()]
    if serialiser_type == AvailableSerialisers.dill:
        return DillSerialiser()
    raise ValueError("Unknown or unimplemented serialsier type: "
                     f"{serialiser_type}")


def serialise(serialiser_type: Union[str, AvailableSerialisers],
              obj: Serialisable, target_folder: str) -> None:
    ser = get_serialiser(serialiser_type)
    ser.serialise_all(obj, target_folder)


def deserialise(serialiser_type: Union[str, AvailableSerialisers],
                folder_path: str) -> Serialisable:
    ser = get_serialiser(serialiser_type)
    return ser.deserialise_all(folder_path)
