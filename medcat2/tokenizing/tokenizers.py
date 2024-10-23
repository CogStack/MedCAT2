from typing import Protocol, Type
import logging

from medcat2.tokenizing.tokens import BaseDocument
from medcat2.utils.registry import Registry


logger = logging.getLogger(__name__)


class BaseTokenizer(Protocol):

    def __call__(selt, text: str) -> BaseDocument:
        pass


_DEFAULT_TOKENIZING: dict[str, tuple[str, str]] = {
    "default": ("medcat2.components.tokenizing.spacy", "Tokenizer")
}

_TOKENIZERS_REGISTRY = Registry(BaseTokenizer,
                                lazy_defaults=_DEFAULT_TOKENIZING)


def get_tokenizer(tokenizer_name: str, *args, **kwargs) -> BaseTokenizer:
    return _TOKENIZERS_REGISTRY.get_component(tokenizer_name)(*args, **kwargs)


def register_platform(name: str, clazz: Type[BaseTokenizer]) -> None:
    _TOKENIZERS_REGISTRY.register(name, clazz)
    logger.debug("Registered tokenizer '%s': '%s.%s'",
                 name, clazz.__module__, clazz.__name__)
