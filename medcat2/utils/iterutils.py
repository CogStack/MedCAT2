from typing import Callable, Iterator, TypeVar

T = TypeVar("T")


def _callback_iterator_iterable(identifier: str, data_iterator: Iterator[T],
                                callback: Callable[[str, int], None]
                                ) -> Iterator[T]:
    count = 0
    try:
        for item in data_iterator:
            count += 1
            yield item
    finally:
        callback(identifier, count)


def callback_iterator(identifier: str, data_iterator: Iterator[T],
                      callback: Callable[[str, int], None]) -> Iterator[T]:
    if hasattr(data_iterator, "__len__"):
        callback(identifier, len(data_iterator))
        return data_iterator
    return _callback_iterator_iterable(identifier, data_iterator, callback)
