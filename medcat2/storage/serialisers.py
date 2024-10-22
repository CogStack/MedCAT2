from enum import Enum, auto
from typing import Any, Union, cast, Type
import os
from abc import ABC, abstractmethod
from importlib import import_module

import dill as _dill

from medcat2.storage.serialisables import Serialisable
from medcat2.storage.serialisables import get_all_serialisable_members
from medcat2.storage.schema import load_schema, save_schema
from medcat2.storage.schema import DEFAULT_SCHEMA_FILE, IllegalSchemaException


class Serialiser(ABC):

    @abstractmethod
    def serialise(self, serialisable: Serialisable, target_file: str) -> None:
        pass

    @abstractmethod
    def deserialise(self, target_file: str) -> Serialisable:
        pass

    def serialise_all(self, obj: Any, target_folder: str) -> None:
        attr2file: dict[str, str] = {}
        for part, name in get_all_serialisable_members(obj):
            basename = part.get_save_name(name)
            target_file = os.path.join(target_folder, basename)
            if os.path.exists(target_file):
                raise IllegalSchemaException(
                    f"File already exists: {target_file}. Unable to overwrite")
            self.serialise(part, target_file)
            if name in attr2file:
                raise IllegalSchemaException(f"Multiple values for {name}")
            attr2file[name] = basename
        schema_path = os.path.join(target_folder, DEFAULT_SCHEMA_FILE)
        save_schema(schema_path, obj.__class__, attr2file)

    def deserialise_all(self, folder_path: str) -> Any:
        schema_path = os.path.join(folder_path, DEFAULT_SCHEMA_FILE)
        cls_path, schema = load_schema(schema_path)
        module_path, cls_name = cls_path.rsplit('.', 1)
        module = import_module(module_path)
        cls: Type = getattr(module, cls_name)
        init_kwargs: dict[str, Serialisable] = {}
        for part_name in os.listdir(folder_path):
            if part_name == DEFAULT_SCHEMA_FILE:
                continue
            part_path = os.path.join(folder_path, part_name)
            desered = self.deserialise(part_path)
            attr_name_candidates = [attr_name for attr_name, file_name
                                    in schema.items()
                                    if file_name == part_name]
            if len(attr_name_candidates) != 1:
                raise IllegalSchemaException(
                    f"Got {len(attr_name_candidates)} attribute name "
                    f"candidates for {part_name} (at {part_path}): "
                    f"{attr_name_candidates}")
            init_kwargs[attr_name_candidates[0]] = desered
        obj = cls(**init_kwargs)
        return obj


class DillSerialiser(Serialiser):

    def serialise(self, serialisable: Serialisable, target_file: str) -> None:
        with open(target_file, 'wb') as f:
            _dill.dump(serialisable, f)

    def deserialise(self, target_file: str) -> Serialisable:
        with open(target_file, 'rb') as f:
            obj = _dill.load(f)
        return cast(Serialisable, obj)


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
              obj: Any, target_folder: str) -> None:
    ser = get_serialiser(serialiser_type)
    ser.serialise_all(obj, target_folder)


def deserialise(serialiser_type: Union[str, AvailableSerialisers],
                folder_path: str) -> Any:
    ser = get_serialiser(serialiser_type)
    return ser.deserialise_all(folder_path)
