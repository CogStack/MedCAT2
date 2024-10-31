from typing import Callable, Iterator, TypeVar

T = TypeVar("T")


def callback_iterator(identifier: str, data_iterator: Iterator[T],
                      callback: Callable[[str, int], None]) -> Iterator[T]:
    # identifier += 1
    if hasattr(data_iterator, "__len__"):
        callback(identifier, len(data_iterator))
        yield from data_iterator
        return
    count = 0
    for item in data_iterator:
        count += 1
        yield item
    callback(identifier, count)
