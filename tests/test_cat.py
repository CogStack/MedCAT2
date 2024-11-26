import os

from medcat2 import cat

import unittest

import shutil

from .utils.legacy.test_convert_config import TESTS_PATH
from .utils.legacy.test_conversion_all import ConvertedFunctionalityTests


class TrainedModelTests(unittest.TestCase):
    TRAINED_MODEL_PATH = os.path.join(TESTS_PATH, 'resources',
                                      'mct2_model_pack.zip')

    @classmethod
    def setUpClass(cls):
        cls._model_folder_no_zip = cls.TRAINED_MODEL_PATH.rsplit(".zip", 1)[0]
        cls._folder_existed = os.path.exists(cls._model_folder_no_zip)
        cls.model = cat.CAT.load_model_pack(cls.TRAINED_MODEL_PATH)
        if cls.model.config.components.linking.train:
            print("TRAINING WAS ENABLE! NEED TO DISABLE")
            cls.model.config.components.linking.train = False


class InferenceFromLoadedTests(TrainedModelTests):

    @classmethod
    def tearDownClass(cls):
        if cls.TRAINED_MODEL_PATH.endswith(".zip"):
            folder = cls._model_folder_no_zip
            if os.path.exists(folder) and not cls._folder_existed:
                shutil.rmtree(folder)

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
