from typing import Any, List, Tuple, Union, Protocol, runtime_checkable


@runtime_checkable
class Serialisable(Protocol):

    def get_save_name(self, name: str) -> str:
        pass


class AbstractSerialisable:

    def __init__(self, name_format: str = "{0}.dat") -> None:
        self._name_format = name_format

    def get_save_name(self, name: str) -> str:
        return self._name_format.format(name)


def name_all_serialisable_elements(target_list: Union[List, Tuple],
                                   name_start: str = '',
                                   all_or_nothing: bool = True
                                   ) -> List[Tuple[Serialisable, str]]:
    """Gets all serialisable elements from a list or tuple.

    There's two strategies for finding the parts:
    1) If `all_or_nothing == True` either all the elements
        in the list must be Serialisable or None of them.
    2) If `all_or_nothing == False` some elements may be
        serialisable while others may not be.

    Args:
        target_list (Union[List, Tuple]): The list/tuple of objects to look in.
        name_start (str, optional): The start of the name. Defaults to ''.
        all_or_nothing (bool, optional):
            Whether to disallow lists/tuple where only some elements are
            serialisable. Defaults to True.

    Raises:
        ValueError: If `all_or_nothing` is specified and not all elements
            are serialisable.

    Returns:
        List[Tuple[Serialisable, str]]: The serialisable parts along with name.
    """
    out_parts: List[Tuple[Serialisable, str]] = []
    if not target_list:
        return out_parts
    for el_nr, el in enumerate(target_list):
        if isinstance(el, Serialisable):
            out_parts.append((el, name_start + f"_el_{el_nr}"))
        elif all_or_nothing and out_parts:
            raise ValueError(f"The first {len(out_parts)} were serialisable "
                             "whereas the next one was not. Specify "
                             "`all_or_nothing=False` to allow for only "
                             "some of the list elements to be serialisable.")
    return out_parts


def get_all_serialisable_members(object: Any
                                 ) -> List[Tuple[Serialisable, str]]:
    """Gets all serialisable members of an object.

    This looks for public and protected members, but not private ones.
    It should also be able to return parts of lists and tuples.
    It also provides the name of each serialisable object.

    Args:
        object (Any): The target object.

    Returns:
        List[Tuple[Serialisable, str]]:
            List of serialisable objects along with their names
    """
    serialisable_parts: List[Tuple[Serialisable, str]] = []
    for name, obj in object.__dict__.items():
        if name.startswith("__"):
            # ignore private members
            continue
        if isinstance(obj, Serialisable):
            serialisable_parts.append((obj, name))
        elif isinstance(obj, list) or isinstance(obj, tuple):
            cur_parts = name_all_serialisable_elements(obj, name_start=name)
            serialisable_parts.extend(cur_parts)
    return serialisable_parts
