from typing import Any, Optional

from medcat.components.addons import addons

from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.config.config import Config, ComponentConfig
from medcat.tokenizing.tokenizers import BaseTokenizer, MutableEntity

import unittest
import unittest.mock
import tempfile


class FakeAddonNoInit:
    name = 'fake_addon'

    def __init__(self, cnf: ComponentConfig):
        assert cnf.comp_name == self.name
        self.config = cnf

    def __call__(self, doc):
        return doc

    @property
    def should_save(self) -> bool:
        return False

    def save(self, path: str) -> None:
        return

    @property
    def addon_type(self) -> str:
        return 'FAKE'

    def get_folder_name(self) -> str:
        return "addon_" + self.full_name

    @property
    def full_name(self) -> str:
        return self.addon_type + "_" + str(self.name)

    def get_output_key_val(self, ent: MutableEntity
                           ) -> tuple[str, dict[str, Any]]:
        return '', {}

    @classmethod
    def create_new_component(
            cls, cnf: ComponentConfig, tokenizer: BaseTokenizer,
            cdb: CDB, vocab: Vocab, model_load_path: Optional[str]) -> 'FakeAddonNoInit':
        return cls(cnf)


class FakeAddonWithInit:
    name = 'fake_addon_w_init'

    def __init__(self, cnf: ComponentConfig,
                 tokenizer: BaseTokenizer, cdb: CDB):
        assert cnf.comp_name == self.name
        self._token = tokenizer
        self._cdb = cdb
        self.config = cnf

    def __call__(self, doc):
        return doc

    @classmethod
    def create_new_component(
            cls, cnf: ComponentConfig, tokenizer: BaseTokenizer,
            cdb: CDB, vocab: Vocab, model_load_path: Optional[str]) -> 'FakeAddonWithInit':
        return cls(cnf, tokenizer, cdb)

    @property
    def should_save(self) -> bool:
        return False

    def save(self, path: str) -> None:
        return

    @property
    def addon_type(self) -> str:
        return 'FAKE'

    def get_folder_name(self) -> str:
        return "addon_" + self.full_name

    @property
    def full_name(self) -> str:
        return self.addon_type + "_" + str(self.name)


class AddonsRegistrationTests(unittest.TestCase):
    addon_cls = FakeAddonNoInit

    @classmethod
    def setUpClass(cls):
        cls.addon_creator = cls.addon_cls.create_new_component
        addons.register_addon(cls.addon_cls.name, cls.addon_creator)

    @classmethod
    def tearDownClass(cls):
        addons._ADDON_REGISTRY.unregister_all_components()
        addons._ADDON_REGISTRY._lazy_defaults.update(addons._DEFAULT_ADDONS)

    def creator_args(self):
        return (
            ComponentConfig(comp_name=self.addon_cls.name),
            None, None, None, None)

    def test_has_registration(self):
        addon_creator = addons.get_addon_creator(self.addon_cls.name)
        self.assertIs(addon_creator, self.addon_creator)

    def test_can_create_empty_addon(self):
        addon = addons.create_addon(
            self.addon_cls.name, *self.creator_args())
        self.assertIsInstance(addon, self.addon_cls)


class AddonUsageTests(unittest.TestCase):
    addon_cls = FakeAddonNoInit
    EXP_ADDONS = 1

    @classmethod
    def setUpClass(cls):
        cls.addon_creator = cls.addon_cls.create_new_component
        addons.register_addon(cls.addon_cls.name, cls.addon_creator)
        cls.cnf = Config()
        cls.cdb = CDB(cls.cnf)
        cls.vocab = Vocab()
        cls.cnf.components.addons.append(ComponentConfig(
            comp_name=cls.addon_cls.name))
        cls.cat = CAT(cls.cdb, cls.vocab)

    def test_has_addon(self):
        self.assertTrue(self.cat._pipeline._addons)
        addon = self.cat._pipeline._addons[0]
        self.assertIsInstance(addon, self.addon_cls)

    def test_addon_runs(self):
        with unittest.mock.patch.object(self.addon_cls, "__call__",
                                        unittest.mock.MagicMock()
                                        ) as mock_call:
            self.cat.get_entities("Some text")
            mock_call.assert_called_once()

    @classmethod
    def tearDownClass(cls):
        addons._ADDON_REGISTRY.unregister_all_components()
        addons._ADDON_REGISTRY._lazy_defaults.update(addons._DEFAULT_ADDONS)

    def test_can_create_cat_with_addon(self):
        self.assertIsInstance(self.cat, CAT)
        self.assertEqual(len(self.cat._pipeline._addons), self.EXP_ADDONS)

    def test_can_save_model(self):
        with tempfile.TemporaryDirectory() as ntd:
            full_path = self.cat.save_model_pack(ntd)
        self.assertIsInstance(full_path, str)

    def test_can_save_and_load(self):
        with tempfile.TemporaryDirectory() as ntd:
            full_path = self.cat.save_model_pack(ntd)
            cat = CAT.load_model_pack(full_path)
        self.assertIsInstance(cat, CAT)
        self.assertEqual(len(self.cat._pipeline._addons), self.EXP_ADDONS)


class AddonUsageWithInitTests(AddonUsageTests):
    addon_cls = FakeAddonWithInit
