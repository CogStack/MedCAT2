import os

from medcat2.utils.legacy import convert_cdb
from medcat2.cdb import CDB

import unittest

from .test_convert_config import TESTS_PATH


class CDBConversionTest(unittest.TestCase):
    FILE_PATH = os.path.join(TESTS_PATH, "resources", "mct_v1_cdb.dat")
    EXP_CUIS = 5

    @classmethod
    def setUpClass(cls):
        cls.cdb = convert_cdb.get_cdb_from_old(cls.FILE_PATH)

    def test_conversion_works(self):
        self.assertIsInstance(self.cdb, CDB)

    def test_has_concepts(self):
        self.assertTrue(self.cdb.cui2info)
        self.assertEqual(len(self.cdb.cui2info), self.EXP_CUIS)

    def test_has_names(self):
        self.assertTrue(self.cdb.name2info)

    def test_all_cui_names_in_names(self):
        for cui, cui_info in self.cdb.cui2info.items():
            for name in cui_info.names:
                with self.subTest(f"{cui}: {name}"):
                    self.assertIn(name, self.cdb.name2info)

    def test_all_name_cuis_in_cuis(self):
        for name, nameinfo in self.cdb.name2info.items():
            for cui in nameinfo.cuis:
                with self.subTest(f"{name}: {cui}"):
                    self.assertIn(cui, self.cdb.cui2info)


# print("GO!")
# from medcat2.utils.legacy.convert_vocab import get_vocab_from_old
# from medcat2.utils.legacy.convert_config import get_config_from_old
# from medcat2.cat import CAT
# TEMP_FILE = "tests/resources/mct_v1_cdb.dat"
# cdb = get_cdb_from_old(TEMP_FILE)
# VOCAB_PATH = "tests/resources/mct_v1_vocab.dat"
# vocab = get_vocab_from_old(VOCAB_PATH)
# CONFIG_PATH = "tests/resources/mct_v1_cnf.json"
# config = get_config_from_old(CONFIG_PATH)
# # FIX CONFIG
# config.preprocessing.max_document_length = 1_000_000 # a million
# cat = CAT(cdb, vocab, config)

# TEXT = ("Man was diagnosed with severe kidney failure and acute diabetes "
#         "and presented with a light fever")
# print("FROM")
# print(TEXT)
# print("TO", cat.get_entities(TEXT))
