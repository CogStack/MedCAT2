from typing import Callable, Protocol

from medcat2.components.types import BaseComponent
from medcat2.utils.registry import Registry
from medcat2.config.config import ComponentConfig


class AddonComponent(BaseComponent, Protocol):
    NAME_PREFIX = "addon_"
    NAME_SPLITTER = ""
    config: ComponentConfig

    @property
    def addon_type(self) -> str:
        pass

    def is_core(self) -> bool:
        return False

    @property
    def should_save(self) -> bool:
        pass

    def save(self, folder: str) -> None:
        pass

    def get_folder_name(self) -> str:
        return self.NAME_PREFIX + self.full_name

    @property
    def full_name(self) -> str:
        return self.addon_type + self.NAME_SPLITTER + str(self.name)


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


def create_addon(addon_name: str, cnf: ComponentConfig,
                 *args, **kwargs) -> AddonComponent:
    return get_addon_creator(addon_name)(cnf, *args, **kwargs)
