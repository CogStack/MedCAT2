from typing import Callable, Iterable, TypeVar, Sized

T = TypeVar("T")


def _callback_iterator_iterable(identifier: str, data_iterator: Iterable[T],
                                callback: Callable[[str, int], None]
                                ) -> Iterable[T]:
    count = 0
    try:
        for item in data_iterator:
            count += 1
            yield item
    finally:
        callback(identifier, count)


def callback_iterator(identifier: str, data_iterator: Iterable[T],
                      callback: Callable[[str, int], None]) -> Iterable[T]:
    if isinstance(data_iterator, Sized):
        callback(identifier, len(data_iterator))
        return data_iterator
    return _callback_iterator_iterable(identifier, data_iterator, callback)
