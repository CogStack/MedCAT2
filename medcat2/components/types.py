from typing import Optional, Protocol, Callable, runtime_checkable, Union
from enum import Enum, auto

from medcat2.utils.registry import Registry
from medcat2.tokenizing.tokens import MutableDocument, MutableEntity


class CoreComponentType(Enum):
    tagging = auto()
    token_normalizing = auto()
    ner = auto()
    linking = auto()


@runtime_checkable
class BaseComponent(Protocol):

    @property
    def full_name(self) -> Optional[str]:
        """Name with the component type (e.g ner, linking, meta)."""
        pass

    @property
    def name(self) -> Optional[str]:
        pass

    def is_core(self) -> bool:
        pass

    def __call__(self, doc: MutableDocument) -> MutableDocument:
        pass


@runtime_checkable
class CoreComponent(BaseComponent, Protocol):

    def get_type(self) -> CoreComponentType:
        pass


class AbstractCoreComponent(CoreComponent):

    @property
    def full_name(self) -> str:
        return self.get_type().name + ":" + str(self.name)

    def is_core(self) -> bool:
        return True


@runtime_checkable
class TrainableComponent(Protocol):

    def train(self, cui: str,
              entity: MutableEntity,
              doc: MutableDocument,
              negative: bool = False,
              names: Union[list[str], dict] = []) -> None:
        pass


_DEFAULT_TAGGERS: dict[str, tuple[str, str]] = {
    "default": ("medcat2.components.tagging.tagger", "TagAndSkipTagger"),
}
_DEFAULT_NORMALIZERS: dict[str, tuple[str, str]] = {
    "default": ("medcat2.components.normalizing.normalizer",
                "TokenNormalizer"),
}
_DEFAULT_NER: dict[str, tuple[str, str]] = {
    "default": ("medcat2.components.ner.vocab_based_ner", "NER"),
}
_DEFAULT_LINKING: dict[str, tuple[str, str]] = {
    "default": ("medcat2.components.linking.context_based_linker", "Linker"),
}


_CORE_REGISTRIES: dict[CoreComponentType, Registry[CoreComponent]] = {
    CoreComponentType.tagging: Registry(
        CoreComponent, lazy_defaults=_DEFAULT_TAGGERS),  # type: ignore
    CoreComponentType.token_normalizing: Registry(
        CoreComponent, lazy_defaults=_DEFAULT_NORMALIZERS),  # type: ignore
    CoreComponentType.ner: Registry(CoreComponent,  # type: ignore
                                    lazy_defaults=_DEFAULT_NER),
    CoreComponentType.linking: Registry(CoreComponent,  # type: ignore
                                        lazy_defaults=_DEFAULT_LINKING),
}


def register_core_component(comp_type: CoreComponentType,
                            comp_name: str,
                            comp_clazz: Callable[..., CoreComponent]) -> None:
    _CORE_REGISTRIES[comp_type].register(comp_name, comp_clazz)


def get_core_registry(comp_type: CoreComponentType) -> Registry[CoreComponent]:
    return _CORE_REGISTRIES[comp_type]


def create_core_component(comp_type: CoreComponentType,
                          comp_name: str,
                          *args, **kwargs) -> CoreComponent:
    comp_getter = get_core_registry(comp_type).get_component(comp_name)
    return comp_getter(*args, **kwargs)


def get_registered_components(comp_type: CoreComponentType
                              ) -> list[tuple[str, str]]:
    registry = get_core_registry(comp_type)
    return registry.list_components()
