from typing import Type, Generic, TypeVar, cast, Optional, Callable
import importlib
import logging


logger = logging.getLogger(__name__)

P = TypeVar('P')


class Registry(Generic[P]):
    def __init__(self, type: Type[P],
                 lazy_defaults: Optional[dict[str, tuple[str, str]]] = None
                 ) -> None:
        self._components: dict[str, Callable[..., P]] = {}
        self._type = type
        self._lazy_defaults = lazy_defaults.copy() if lazy_defaults else {}

    def register(self, component_name: str,
                 creator: Callable[..., P]):
        if component_name in self._components:
            prev = self._components[component_name]
            raise MedCATRegistryException(
                f"Component '{component_name}' already registered: {prev}")
        self._components[component_name] = creator

    def get_component(self, component_name: str
                      ) -> Callable[..., P]:
        # NOTE: some default implementations may be big imports,
        #       so we only want to import them if/when required.
        if component_name in self._lazy_defaults:
            self._ensure_lazy_default(component_name)
        if component_name not in self._components:
            raise MedCATRegistryException(
                f"No component registered by nane '{component_name}'. "
                f"Available components: {self.list_components()}")
        return self._components[component_name]

    def _ensure_lazy_default(self, component_name: str) -> None:
        module_name, class_name = self._lazy_defaults.pop(component_name)
        logger.debug("Registering default %s '%s': '%s.%s'",
                     self._type.__name__, component_name, module_name,
                     class_name)
        module_in = importlib.import_module(module_name)
        cls = getattr(module_in, class_name)
        self.register(component_name, cast(Callable[..., P], cls))

    def register_all_defaults(self) -> None:
        for comp_name in list(self._lazy_defaults):
            self._ensure_lazy_default(comp_name)

    def list_components(self) -> list[tuple[str, str]]:
        comps = [(comp_name, comp.__name__)
                 for comp_name, comp in self._components.items()]
        for lazy_def_name, (_, lazy_def_class) in self._lazy_defaults.items():
            comps.append((lazy_def_name, lazy_def_class))
        return comps

    def unregister_component(self, component_name: str
                             ) -> Callable[..., P]:
        if component_name not in self._components:
            raise MedCATRegistryException(
                f"No such component: {component_name}")
        logger.debug("Unregistering %s '%s'", self._type.__name__,
                     component_name)
        return self._components.pop(component_name)

    def unregister_all_components(self) -> None:
        for comp_name in list(self._components):
            self.unregister_component(comp_name)

    def __contains__(self, component_name: str) -> bool:
        return (component_name in self._components or
                component_name in self._lazy_defaults)

    def __getitem__(self, component_name: str) -> Callable[..., P]:
        return self.get_component(component_name)


class MedCATRegistryException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
