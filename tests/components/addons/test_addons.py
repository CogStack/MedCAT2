from typing import Any

from medcat2.components.addons import addons

from medcat2.cat import CAT
from medcat2.cdb import CDB
from medcat2.vocab import Vocab
from medcat2.config.config import Config, ComponentConfig
from medcat2.tokenizing.tokenizers import BaseTokenizer

import unittest


class FakeAddonNoInit:
    name = 'fake_addon'


class FakeAddonWithInit:
    name = 'fake_addon_w_init'

    def __init__(self, tokenizer: BaseTokenizer, cdb: CDB):
        self._token = tokenizer
        self._cdb = cdb

    @classmethod
    def get_init_args(cls, tokenizer: BaseTokenizer, cdb: CDB, vocab: Vocab
                      ) -> list[Any]:
        return [tokenizer, cdb]

    @classmethod
    def get_init_kwargs(cls, tokenizer: BaseTokenizer, cdb: CDB, vocab: Vocab
                        ) -> dict[str, Any]:
        return {}


class AddonsRegistrationTests(unittest.TestCase):
    addon_cls = FakeAddonNoInit

    @classmethod
    def setUpClass(cls):
        addons.register_addon(cls.addon_cls.name, cls.addon_cls)

    @classmethod
    def tearDownClass(cls):
        addons._ADDON_REGISTRY.unregister_all_components()

    def test_has_registration(self):
        addon_cls = addons.get_addon_creator(self.addon_cls.name)
        self.assertIs(addon_cls, self.addon_cls)

    def test_can_create_empty_addon(self):
        addon = addons.create_addon(self.addon_cls.name)
        self.assertIsInstance(addon, self.addon_cls)


class AddonUsageTests(unittest.TestCase):
    addon_cls = FakeAddonNoInit
    EXP_ADDONS = 1

    @classmethod
    def setUpClass(cls):
        addons.register_addon(cls.addon_cls.name, cls.addon_cls)
        cls.cnf = Config()
        cls.cdb = CDB(cls.cnf)
        cls.vocab = Vocab()
        cls.cnf.components.addons.append(ComponentConfig(
            comp_name=cls.addon_cls.name))

    @classmethod
    def tearDownClass(cls):
        addons._ADDON_REGISTRY.unregister_all_components()

    def test_can_create_cat_with_addon(self):
        cat = CAT(self.cdb, self.vocab)
        self.assertIsInstance(cat, CAT)
        self.assertEqual(len(cat._platform._addons), self.EXP_ADDONS)


class AddonUsageWithInitTests(AddonUsageTests):
    addon_cls = FakeAddonWithInit
