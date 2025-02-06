from typing import Callable, Protocol

from medcat2.components.types import BaseComponent
from medcat2.utils.registry import Registry
from medcat2.config.config import ComponentConfig


class AddonComponent(BaseComponent, Protocol):
    """Base/abstract addon component class."""

    def is_core(self) -> bool:
        return False


_DEFAULT_ADDONS: dict[str, tuple[str, str]] = {
    # 'addon name' : ('module name', 'class name')
}

# NOTE: type error due to non-concrete type
_ADDON_REGISTRY = Registry(AddonComponent, _DEFAULT_ADDONS)  # type: ignore


def register_addon(addon_name: str,
                   addon_cls: Callable[..., AddonComponent]) -> None:
    """Register a new addon.

    Args:
        addon_name (str): The addon name.
        addon_cls (Callable[..., AddonComponent]): The addon creator.
    """
    _ADDON_REGISTRY.register(addon_name, addon_cls)


def get_addon_creator(addon_name: str) -> Callable[..., AddonComponent]:
    """Get the creator for an addon.

    Args:
        addon_name (str): The name of the addonl

    Returns:
        Callable[..., AddonComponent]: The creator of the addon.
    """
    return _ADDON_REGISTRY.get_component(addon_name)


def create_addon(addon_name: str, cnf: ComponentConfig,
                 *args, **kwargs) -> AddonComponent:
    """Create an addon of the specified name with the specified arguments.

    All the `*args`, and `**kwrags` are passed to the creator.

    Args:
        addon_name (str): The name of the addon.
        cnf (ComponentConfig): The addon config.

    Returns:
        AddonComponent: The resulting / created addon.
    """
    return get_addon_creator(addon_name)(cnf, *args, **kwargs)
