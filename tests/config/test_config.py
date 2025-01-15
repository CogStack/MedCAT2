from medcat2.config import config
from medcat2.storage.serialisables import Serialisable

import unittest


class ConfigTests(unittest.TestCase):
    CHANGED_NLP = '#NON#-#EXISTENT#'
    TO_MERGE_SIMPLE = {
        "general": {
            "nlp": {
                "provider": CHANGED_NLP,
            },
        },
    }
    TO_MERGE_INCORRECT = {
        "general": 1,
    }
    TO_MERGE_INCORRECT_VAL = {
        "general": {
            "nlp": {
                "provider": -1,
            },
        },
    }
    NEW_KEY_PATH = 'general.nlp.this_key_is_new_woo'
    TO_MERGE_NEW_VAL = 123
    TO_MERGE_NEW_KEY = {
        NEW_KEY_PATH.split(".")[0]: {
            NEW_KEY_PATH.split(".")[1]: {
                NEW_KEY_PATH.split(".")[2]: TO_MERGE_NEW_VAL,
            }
        }
    }
    NEW_KEY_PATH_INCORRECT = 'general.this_key_is_new_woo'
    TO_MERGE_NEW_KEY_INCORRECT = {
        NEW_KEY_PATH_INCORRECT.split(".")[0]: {
            NEW_KEY_PATH_INCORRECT.split(".")[1]: TO_MERGE_NEW_VAL,
        }
    }

    def setUp(self):
        self.cnf = config.Config()

    def test_is_serialisable(self):
        self.assertIsInstance(self.cnf, Serialisable)

    def test_can_merge(self):
        self.assertNotEqual(self.cnf.general.nlp.provider, self.CHANGED_NLP)
        self.cnf.merge_config(self.TO_MERGE_SIMPLE)
        self.assertEqual(self.cnf.general.nlp.provider, self.CHANGED_NLP)

    def test_fails_to_merge_incorrect_model(self):
        with self.assertRaises(config.IncorrectConfigValues):
            self.cnf.merge_config(self.TO_MERGE_INCORRECT)

    def test_fails_to_merge_incorrect_value(self):
        with self.assertRaises(config.IncorrectConfigValues):
            self.cnf.merge_config(self.TO_MERGE_INCORRECT_VAL)

    def test_can_merge_new_value_where_allowed(self):
        self.cnf.merge_config(self.TO_MERGE_NEW_KEY)
        cur = self.cnf
        for path in self.NEW_KEY_PATH.split("."):
            cur = getattr(cur, path)
        self.assertEqual(cur, self.TO_MERGE_NEW_VAL)

    def test_cannot_mege_new_value_not_allowed(self):
        with self.assertRaises(config.IncorrectConfigValues):
            self.cnf.merge_config(self.TO_MERGE_NEW_KEY_INCORRECT)
