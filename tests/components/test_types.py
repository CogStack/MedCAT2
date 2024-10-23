from medcat2.components import types

from medcat2.tokenizing.tokens import BaseDocument, MutableDocument
from medcat2.utils.registry import MedCATRegistryException

import unittest


class FakeBaseComponent:

    def __init__(self, comp_type: types.CoreComponentType,
                 name: str = "test-fake-component"):
        self.name = name
        self.comp_type = comp_type

    def get_type(self) -> types.CoreComponentType:
        return self.comp_type

    def __call__(self, raw: BaseDocument, mutable: MutableDocument
                 ) -> MutableDocument:
        return mutable


class TypesRegistrationTests(unittest.TestCase):
    # NOTE: if/when default commponents get added, this needs to change
    _DEF_COMPS = len(types._DEFAULT_LINKING)
    COMP_TYPE = types.CoreComponentType.linking
    WRONG_TYPE = types.CoreComponentType.ner
    COMP_NAME = "test-linker"
    BCC = FakeBaseComponent

    def setUp(self):
        types.register_core_component(self.COMP_TYPE, self.COMP_NAME, self.BCC)
        self.registered = types.create_core_component(
            self.COMP_TYPE, self.COMP_NAME, self.COMP_TYPE)

    def tearDown(self):
        for registry in types._CORE_REGISTRIES.values():
            registry.unregister_all_components()

    def test_registered_is_base_component(self):
        self.assertIsInstance(self.registered, types.BaseComponent)

    def test_registered_is_fake_component(self):
        self.assertIsInstance(self.registered, FakeBaseComponent)

    def test_does_not_get_incorrect_type(self):
        with self.assertRaises(MedCATRegistryException):
            types.create_core_component(self.WRONG_TYPE, self.COMP_NAME)

    def test_does_not_get_incorrect_name(self):
        with self.assertRaises(MedCATRegistryException):
            types.create_core_component(self.COMP_TYPE, "#" + self.COMP_NAME)

    def test_lists_registered_component(self):
        comps = types.get_registered_components(self.COMP_TYPE)
        self.assertEqual(len(comps), 1 + self._DEF_COMPS)
        comp_name, comp_cls = comps[0]
        self.assertEqual(comp_name, self.COMP_NAME)
        self.assertEqual(comp_cls, self.BCC.__name__)
