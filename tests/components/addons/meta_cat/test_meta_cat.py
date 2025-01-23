from typing import runtime_checkable, Type, Any

from medcat2.components.addons.meta_cat import meta_cat
from medcat2.components.addons.addons import AddonComponent
from medcat2.storage.serialisables import Serialisable
from medcat2.storage.serialisers import serialise, AvailableSerialisers
from medcat2.config.config_meta_cat import ConfigMetaCAT

import unittest
import unittest.mock
import tempfile

from transformers import AutoTokenizer
from medcat2.components.addons.meta_cat.meta_cat_tokenizers import (
    TokenizerWrapperBERT)
from medcat2.cat import CAT

from .... import EXAMPLE_MODEL_PACK_ZIP


class FakeEntity:

    @classmethod
    def register_addon_path(self, path: str, def_val: Any, force: bool = True):
        pass


class FakeTokenizer:

    @classmethod
    def get_entity_class(cls) -> Type:
        return FakeEntity

    @classmethod
    def get_doc_class(cls) -> Type:
        return FakeEntity


class MetaCATBaseTests(unittest.TestCase):
    SER_TYPE = AvailableSerialisers.dill
    VOCAB_SIZE = 10
    PAD_IDX = 5

    @classmethod
    def setUpClass(cls):
        cls.cnf = ConfigMetaCAT()
        cls.cnf.comp_name = meta_cat.MetaCATAddon.addon_type
        cls.cnf.general.vocab_size = cls.VOCAB_SIZE
        cls.cnf.model.padding_idx = cls.PAD_IDX
        cls.tokenizer = FakeTokenizer()
        mc_tokenizer = TokenizerWrapperBERT(
            AutoTokenizer.from_pretrained('prajjwal1/bert-tiny'))
        cls.meta_cat = meta_cat.MetaCATAddon(
            cls.cnf, cls.tokenizer, None, tokenizer=mc_tokenizer)


class MetaCATTests(MetaCATBaseTests):

    def test_is_addon(self):
        self.assertIsInstance(self.meta_cat, runtime_checkable(AddonComponent))

    def test_is_serialisable_meta_cat(self):
        self.assertIsInstance(self.meta_cat.mc,
                              runtime_checkable(Serialisable))


class MetaCATWithCATTests(MetaCATBaseTests):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cat = CAT.load_model_pack(EXAMPLE_MODEL_PACK_ZIP)
        cls.cat.add_addon(cls.meta_cat)

    def assert_has_meta_cat(self, cat: CAT, same: bool = True):
        self.assertEqual(len(cat._pipeline._addons), 1)
        addon = self.cat._pipeline._addons[0]
        if same:
            self.assertIs(addon, self.meta_cat)
        else:
            self.assertEqual(addon, self.meta_cat)

    def test_has_added_meta_cat(self):
        self.assert_has_meta_cat(self.cat, True)

    def test_can_save(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            serialise(self.SER_TYPE, self.cat, temp_dir)

    def test_can_save_and_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = self.cat.save_model_pack(
                temp_dir, serialiser_type=self.SER_TYPE)
            cat2 = CAT.load_model_pack(file_name)
        self.assert_has_meta_cat(cat2, False)
