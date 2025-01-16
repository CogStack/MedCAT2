from typing import Callable, Protocol

from medcat2.components.types import BaseComponent
from medcat2.utils.registry import Registry


class AddonComponent(BaseComponent, Protocol):

    def is_core(self) -> bool:
        return False


_DEFAULT_ADDONS: dict[str, tuple[str, str]] = {
    # 'addon name' : ('module name', 'class name')
}

# NOTE: type error due to non-concrete type
_ADDON_REGISTRY = Registry(AddonComponent, _DEFAULT_ADDONS)  # type: ignore


def register_addon(addon_name: str,
                   addon_cls: Callable[..., AddonComponent]) -> None:
    _ADDON_REGISTRY.register(addon_name, addon_cls)


def get_addon_creator(addon_name: str) -> Callable[..., AddonComponent]:
    return _ADDON_REGISTRY.get_component(addon_name)


def create_addon(addon_name: str, *args, **kwargs) -> AddonComponent:
    return get_addon_creator(addon_name)(addon_name, *args, **kwargs)
