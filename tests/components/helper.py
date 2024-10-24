from typing import runtime_checkable, Protocol, Type

from medcat2.components import types
from medcat2.config.config import Config, CoreComponentConfig


class FModule(Protocol):

    def set_def_args_kwargs(config: Config):
        pass


class ComponentInitTests:
    default = 'default'
    # these need to be specified when overriding
    comp_type: types.CoreComponentType
    default_cls: Type[types.BaseComponent]
    module: FModule

    @classmethod
    def set_def_args(cls):
        cls.module.set_def_args_kwargs(cls.cnf)

    @classmethod
    def setUpClass(cls):
        cls.cnf = Config()
        cls.set_def_args()
        cls.comp_cnf: CoreComponentConfig = getattr(
            cls.cnf.components, cls.comp_type.name)

    def test_has_default(self):
        avail_components = types.get_registered_components(self.comp_type)
        self.assertEqual(len(avail_components), 1)
        name, cls_name = avail_components[0]
        self.assertEqual(name, self.default)
        self.assertIs(cls_name, self.default_cls.__name__)

    def test_can_create_def_component(self):
        component = types.create_core_component(
            self.comp_type,
            self.default, *self.comp_cnf.init_args,
            **self.comp_cnf.init_kwargs)
        self.assertIsInstance(component,
                              runtime_checkable(types.BaseComponent))
        self.assertIsInstance(component, self.default_cls)
