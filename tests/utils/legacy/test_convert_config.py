import os

from medcat2.utils.legacy import convert_config

from medcat2.config import Config

import unittest


TESTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", ".."))


class ValAndModelGetterTests(unittest.TestCase):
    EXP_MODEL = 'SPACY MODEL'
    EXP_PATH = 'general.nlp.spacy_model'
    MODEL = Config()
    DICT = {
        'general': {
            'nlp': {
                'spacy_model': EXP_MODEL,
            }
        }
    }

    def test_can_get_correct_val_and_model(self):
        val, model = convert_config.get_val_and_parent_model(
            self.DICT, self.MODEL, self.EXP_PATH)
        self.assertEqual(val, self.EXP_MODEL)
        self.assertEqual(model, self.MODEL.general.nlp)

    def test_can_get_val_only(self):
        val, _ = convert_config.get_val_and_parent_model(
            self.DICT, None, self.EXP_PATH)
        self.assertEqual(val, self.EXP_MODEL)

    def test_no_model_when_val_only(self):
        _, model = convert_config.get_val_and_parent_model(
            self.DICT, None, self.EXP_PATH)
        self.assertIsNone(model)

    def test_can_get_model_only(self):
        _, model = convert_config.get_val_and_parent_model(
            None, self.MODEL, self.EXP_PATH)
        self.assertEqual(model, self.MODEL.general.nlp)

    def test_no_val_when_model_only(self):
        val, _ = convert_config.get_val_and_parent_model(
            None, self.MODEL, self.EXP_PATH)
        self.assertIsNone(val)


class ConfigConverstionTests(unittest.TestCase):
    FILE_PATH = os.path.join(TESTS_PATH, "resources", "mct_v1_cnf.json")

    def test_can_convert(self):
        cnf = convert_config.get_config_from_old(self.FILE_PATH)
        self.assertIsInstance(cnf, Config)
