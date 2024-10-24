from typing import Optional, Protocol, Callable, runtime_checkable
from enum import Enum, auto

from medcat2.utils.registry import Registry
from medcat2.tokenizing.tokens import MutableDocument


class CoreComponentType(Enum):
    tagging = auto()
    token_normalizing = auto()
    ner = auto()
    linking = auto()


@runtime_checkable
class BaseComponent(Protocol):

    @property
    def name(self) -> Optional[str]:
        pass

    def get_type(self) -> CoreComponentType:
        pass

    def __call__(self, doc: MutableDocument) -> MutableDocument:
        pass


# TODO: look into these
_DEFAULT_TAGGERS: dict[str, tuple[str, str]] = {
    "default": ("medcat2.components.tagging.tagger", "TagAndSkipTagger"),
}
_DEFAULT_NORMALIZERS: dict[str, tuple[str, str]] = {
    # "default": ("medcat2.components.normalizers.normalizers",
    #             "TokenNormalizer"),
}
_DEFAULT_NER: dict[str, tuple[str, str]] = {
    # "default": ("medcat2.components.ner.vocab_based_ner", "NER"),
}
_DEFAULT_LINKING: dict[str, tuple[str, str]] = {
    # "default": ("medcat2.components.linking.context_based_linker", "Linker"),
}


_CORE_REGISTRIES: dict[CoreComponentType, Registry[BaseComponent]] = {
    CoreComponentType.tagging: Registry(
        BaseComponent, lazy_defaults=_DEFAULT_TAGGERS),  # type: ignore
    CoreComponentType.token_normalizing: Registry(
        BaseComponent, lazy_defaults=_DEFAULT_NORMALIZERS),  # type: ignore
    CoreComponentType.ner: Registry(BaseComponent,  # type: ignore
                                    lazy_defaults=_DEFAULT_NER),
    CoreComponentType.linking: Registry(BaseComponent,  # type: ignore
                                        lazy_defaults=_DEFAULT_LINKING),
}


def register_core_component(comp_type: CoreComponentType,
                            comp_name: str,
                            comp_clazz: Callable[..., BaseComponent]) -> None:
    _CORE_REGISTRIES[comp_type].register(comp_name, comp_clazz)


def get_core_registry(comp_type: CoreComponentType) -> Registry[BaseComponent]:
    return _CORE_REGISTRIES[comp_type]


def create_core_component(comp_type: CoreComponentType,
                          comp_name: str,
                          *args, **kwargs) -> BaseComponent:
    comp_getter = get_core_registry(comp_type).get_component(comp_name)
    return comp_getter(*args, **kwargs)


def get_registered_components(comp_type: CoreComponentType
                              ) -> list[tuple[str, str]]:
    registry = get_core_registry(comp_type)
    return registry.list_components()
