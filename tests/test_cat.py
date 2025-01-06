import os
import pandas as pd
import json
from typing import Optional

from medcat2 import cat
from medcat2.vocab import Vocab
from medcat2.config import Config
from medcat2.model_creation.cdb_maker import CDBMaker
from medcat2.cdb import CDB
from medcat2.cat import CAT

import unittest

from . import EXAMPLE_MODEL_PACK_ZIP
from .utils.legacy.test_conversion_all import ConvertedFunctionalityTests


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


class CATCReationTests(unittest.TestCase):
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

        maker = CDBMaker(config)

        cls.cdb: CDB = maker.prepare_csvs([cls.CDB_PREPROCESSED_PATH])

        # CAT
        cls.cat = CAT(cls.cdb, vocab)

    @classmethod
    def get_cui2ct(cls, cat: Optional[CAT] = None):
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
