from typing import Any
from contextlib import contextmanager
from pydantic import BaseModel


@contextmanager
def temp_changed_config(config: BaseModel, target: str, value: Any):
    try:
        prev_value = getattr(config, target)
    except AttributeError as e:
        raise IllegalConfigPathException(target) from e
    setattr(config, target, value)
    try:
        yield
    finally:
        setattr(config, target, prev_value)


class IllegalConfigPathException(ValueError):

    def __init__(self, target_path: str):
        super().__init__(
            f"Config has no target path: {target_path}")
