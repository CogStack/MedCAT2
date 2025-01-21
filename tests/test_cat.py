import os
import pandas as pd
import json
from typing import Optional

from medcat2 import cat
from medcat2.vocab import Vocab
from medcat2.config import Config
from medcat2.model_creation.cdb_maker import CDBMaker
from medcat2.cdb import CDB
from medcat2.tokenizing.tokens import UnregisteredDataPathException

import unittest
import unittest.mock

from . import EXAMPLE_MODEL_PACK_ZIP
from .utils.legacy.test_conversion_all import ConvertedFunctionalityTests


orig_init = cat.CAT.__init__


class ModelLoadTests(unittest.TestCase):

    def assert_has_model_name(self, func):
        expected_model_packl_name = EXAMPLE_MODEL_PACK_ZIP.replace(".zip", "")

        def wrapper(*args, **kwargs):
            if 'model_load_path' in kwargs:
                self.assertEqual(kwargs['model_load_path'],
                                 expected_model_packl_name)
            else:
                self.assertEqual(args[-1], expected_model_packl_name)
            return func(*args, **kwargs)
        return wrapper

    def setUp(self):
        cat.CAT.__init__ = self.assert_has_model_name(cat.CAT.__init__)

    def tearDown(self):
        cat.CAT.__init__ = orig_init

    def test_loaded_model_knows_model_path(self):
        # NOTE: the assertion is checked due to wrapper on CAT.__init__
        inst = cat.CAT.load_model_pack(EXAMPLE_MODEL_PACK_ZIP)
        self.assertIsInstance(inst, cat.CAT)


class TrainedModelTests(unittest.TestCase):
    TRAINED_MODEL_PATH = EXAMPLE_MODEL_PACK_ZIP

    @classmethod
    def setUpClass(cls):
        cls.model = cat.CAT.load_model_pack(cls.TRAINED_MODEL_PATH)
        if cls.model.config.components.linking.train:
            print("TRAINING WAS ENABLE! NEED TO DISABLE")
            cls.model.config.components.linking.train = False


class InferenceFromLoadedTests(TrainedModelTests):

    def test_can_load_model(self):
        self.assertIsInstance(self.model, cat.CAT)

    def test_has_training(self):
        self.assertTrue(self.model.cdb.cui2info)
        self.assertTrue(self.model.cdb.name2info)

    def test_inference_works(self):
        ents = self.model.get_entities(
            ConvertedFunctionalityTests.TEXT)['entities']
        for nr, ent in enumerate(ents.values()):
            with self.subTest(f"{nr}"):
                ConvertedFunctionalityTests.assert_has_ent(ent)


class CATIncludingTests(unittest.TestCase):
    TOKENIZING_PROVIDER = 'regex'
    EXPECT_TRAIN = {}

    # paths
    VOCAB_DATA_PATH = os.path.join(
        os.path.dirname(__file__), 'resources', 'vocab_data.txt'
    )
    CDB_PREPROCESSED_PATH = os.path.join(
        os.path.dirname(__file__), 'resources', 'preprocessed4cdb.txt'
    )

    @classmethod
    def setUpClass(cls):

        # vocab

        vocab = Vocab()
        vocab.add_words(cls.VOCAB_DATA_PATH)

        # CDB
        config = Config()

        # tokenizer
        config.general.nlp.provider = cls.TOKENIZING_PROVIDER

        maker = CDBMaker(config)

        cls.cdb: CDB = maker.prepare_csvs([cls.CDB_PREPROCESSED_PATH])

        # CAT
        cls.cat = cat.CAT(cls.cdb, vocab)
        cls.cat.config.general.nlp.provider


class CATCReationTests(CATIncludingTests):

    @classmethod
    def get_cui2ct(cls, cat: Optional[cat.CAT] = None):
        if cat is None:
            cat = cls.cat
        return {
            cui: info.count_train for cui, info in cat.cdb.cui2info.items()
            if info.count_train}

    def test_has_expected_training(self):
        self.assertEqual(self.get_cui2ct(), self.EXPECT_TRAIN)


class CATUnsupTrainingTests(CATCReationTests):
    SELF_SUPERVISED_DATA_PATH = os.path.join(
        os.path.dirname(__file__), 'resources', 'selfsupervised_data.txt'
    )
    EXPECT_TRAIN = {'C01': 2, 'C02': 2, 'C03': 2, 'C04': 1, 'C05': 1}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        data = pd.read_csv(cls.SELF_SUPERVISED_DATA_PATH)
        cls.cat.trainer.train_unsupervised(data.text.values)


class CATSupTrainingTests(CATUnsupTrainingTests):
    SUPERVISED_DATA_PATH = os.path.join(
        os.path.dirname(__file__), 'resources', 'supervised_mct_export.json'
    )
    EXPECT_TRAIN = {'C01': 6, 'C02': 7, 'C03': 6, 'C04': 7, 'C05': 7}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with open(cls.SUPERVISED_DATA_PATH) as f:
            data = json.load(f)
        cls.cat.trainer.train_supervised_raw(data)


class CATWithDocAddonTests(CATIncludingTests):
    EXAMPLE_TEXT = "Example text to tokenize"
    ADDON_PATH = 'SMTH'
    EXAMPLE_VALUE = 'something else'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        doc = cls.cat(cls.EXAMPLE_TEXT)
        cls.doc_cls = doc.__class__
        cls.doc_cls.register_addon_path(cls.ADDON_PATH)

    def setUp(self):
        self.doc = self.cat(self.EXAMPLE_TEXT)

    def test_can_set_value(self):
        self.doc.set_addon_data(self.ADDON_PATH, self.EXAMPLE_VALUE)

    def test_cannot_set_incorrect_value(self):
        with self.assertRaises(UnregisteredDataPathException):
            self.doc.set_addon_data(self.ADDON_PATH * 2 + "#",
                                    self.EXAMPLE_TEXT)

    def test_cannot_get_incorrect_value(self):
        with self.assertRaises(UnregisteredDataPathException):
            self.doc.get_addon_data(self.ADDON_PATH * 2 + "#")

    def test_can_load_value(self):
        self.doc.set_addon_data(self.ADDON_PATH, self.EXAMPLE_VALUE)
        got = self.doc.get_addon_data(self.ADDON_PATH)
        self.assertEqual(self.EXAMPLE_VALUE, got)


class CATWithDocAddonSpacyTests(CATWithDocAddonTests):
    TOKENIZING_PROVIDER = 'spacy'


class CATWithEntityAddonTests(CATIncludingTests):
    EXAMPLE_TEXT = "Example text to tokenize"
    EXAMPLE_ENT_START = 0
    EXAMPLE_ENT_END = 2
    ADDON_PATH = 'SMTH'
    EXAMPLE_VALUE = 'something else'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        doc = cls.cat(cls.EXAMPLE_TEXT)
        doc.__getitem__
        entity = doc[0:-1]
        cls.entity_cls = entity.__class__
        cls.entity_cls.register_addon_path(cls.ADDON_PATH)

    def setUp(self):
        self.doc = self.cat(self.EXAMPLE_TEXT)
        self.entity = self.doc[self.EXAMPLE_ENT_START: self.EXAMPLE_ENT_END]

    def test_can_add_data(self):
        self.entity.set_addon_data(self.ADDON_PATH, self.EXAMPLE_VALUE)

    def test_cannot_add_data_to_wrong_path(self):
        with self.assertRaises(UnregisteredDataPathException):
            self.entity.set_addon_data(self.ADDON_PATH * 2 + "£",
                                       self.EXAMPLE_VALUE)

    def test_cannot_get_data_to_wrong_path(self):
        with self.assertRaises(UnregisteredDataPathException):
            self.entity.get_addon_data(self.ADDON_PATH * 2 + "£")

    def test_can_get_data(self):
        self.entity.set_addon_data(self.ADDON_PATH, self.EXAMPLE_VALUE)
        got = self.entity.get_addon_data(self.ADDON_PATH)
        self.assertEqual(self.EXAMPLE_VALUE, got)

    def test_data_is_persistent(self):
        self.entity.set_addon_data(self.ADDON_PATH, self.EXAMPLE_VALUE)
        ent = self.doc[self.EXAMPLE_ENT_START: self.EXAMPLE_ENT_END]
        # new instance
        self.assertFalse(ent is self.entity)
        got = ent.get_addon_data(self.ADDON_PATH)
        self.assertEqual(self.EXAMPLE_VALUE, got)


class CATWithEntityAddonSpacyTests(CATWithEntityAddonTests):
    TOKENIZING_PROVIDER = 'spacy'
