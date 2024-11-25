from typing import Type
import json


_CLASS_PATH = "serialised-class"
_INIT_PARTS_PATH = "init-parts"

# hidden file so that it doesn't get overwritten by other things
DEFAULT_SCHEMA_FILE = ".schema.json"


def _cls2path(cls: Type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def save_schema(file_name: str, cls: Type, init_parts: list[str]) -> None:
    """Saves the schema of a class to the specified file.

    Args:
        file_name (str): The file to save to.
        cls (Type): The class in question
        init_parts list[str]: The parts of the .
    """
    out_data = {
        _CLASS_PATH: _cls2path(cls),
        _INIT_PARTS_PATH: init_parts,
    }
    with open(file_name, 'w') as f:
        json.dump(out_data, f)


def load_schema(file_name: str) -> tuple[str, list[str]]:
    """Loads the schema for a folder of deserialisable files from the file.

    Args:
        file_name (str): The schema file

    Returns:
        tuple[str, list[str]]: The class package/name along
            with the parts needed for initialising.
    """
    with open(file_name) as f:
        data = json.load(f)
    return data[_CLASS_PATH], data[_INIT_PARTS_PATH]


class IllegalSchemaException(ValueError):

    def __init__(self, *args):
        super().__init__("Illegal schema:", *args)
