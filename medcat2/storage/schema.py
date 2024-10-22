from typing import Type
import json


_CLASS_PATH = "serialised-class"
_PARTS_PATH = "serialised-parts"

# hidden file so that it doesn't get overwritten by other things
DEFAULT_SCHEMA_FILE = ".schema.json"


def _cls2path(cls: Type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def save_schema(file_name: str, cls: Type, schema: dict[str, str]) -> None:
    """Saves the schema of a class to the specified file.

    Args:
        file_name (str): The file to save to.
        cls (Type): The class in question
        schema (dict[str, str]): The schema mapping attributes
            to the appropriate file names.
    """
    out_data = {
        _CLASS_PATH: _cls2path(cls),
        _PARTS_PATH: schema,
    }
    with open(file_name, 'w') as f:
        json.dump(out_data, f)


def load_schema(file_name: str) -> tuple[str, dict[str, str]]:
    """Loads the schema for a folder of deserialisable files from the file.

    Args:
        file_name (str): The schema file

    Returns:
        tuple[str, dict[str, str]]: The class package/name along with
            the mapping from attribute to file name.
    """
    with open(file_name) as f:
        data = json.load(f)
    return data[_CLASS_PATH], data[_PARTS_PATH]


class IllegalSchemaException(ValueError):

    def __init__(self, *args):
        super().__init__("Illegal schema:", *args)
