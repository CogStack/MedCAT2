from typing import Protocol, Type, Any, Callable
import logging

from medcat2.config import Config
from medcat2.tokenizing.tokens import (MutableDocument, MutableEntity,
                                       MutableToken)
from medcat2.utils.registry import Registry


logger = logging.getLogger(__name__)


class BaseTokenizer(Protocol):

    def create_entity(self, doc: MutableDocument,
                      token_start_index: int, token_end_index: int,
                      label: str) -> MutableEntity:
        pass

    def entity_from_tokens(self, tokens: list[MutableToken]) -> MutableEntity:
        pass

    def __call__(selt, text: str) -> MutableDocument:
        pass

    @classmethod
    def get_init_args(cls, config: Config) -> list[Any]:
        pass

    @classmethod
    def get_init_kwargs(cls, config: Config) -> dict[str, Any]:
        pass

    def get_doc_class(self) -> Type[MutableDocument]:
        pass

    def get_entity_class(self) -> Type[MutableEntity]:
        pass


_DEFAULT_TOKENIZING: dict[str, tuple[str, str]] = {
    "regex": ("medcat2.tokenizing.regex_impl.tokenizer", "RegexTokenizer"),
    "spacy": ("medcat2.tokenizing.spacy_impl.tokenizers", "SpacyTokenizer")
}

_TOKENIZERS_REGISTRY = Registry(BaseTokenizer,  # type: ignore
                                lazy_defaults=_DEFAULT_TOKENIZING)


def get_tokenizer_creator(tokenizer_name: str) -> Callable[..., BaseTokenizer]:
    return _TOKENIZERS_REGISTRY.get_component(tokenizer_name)


def create_tokenizer(tokenizer_name: str, *args, **kwargs) -> BaseTokenizer:
    return _TOKENIZERS_REGISTRY.get_component(tokenizer_name)(*args, **kwargs)


def list_available_tokenizers() -> list[tuple[str, str]]:
    return _TOKENIZERS_REGISTRY.list_components()


def register_tokenizer(name: str, clazz: Type[BaseTokenizer]) -> None:
    _TOKENIZERS_REGISTRY.register(name, clazz)
    logger.debug("Registered tokenizer '%s': '%s.%s'",
                 name, clazz.__module__, clazz.__name__)
