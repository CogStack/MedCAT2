import os

from medcat2.utils.legacy import legacy_converter
from medcat2.cat import CAT

import shutil

import unittest

from .test_convert_vocab import TESTS_PATH


class ConversionFromZIPTests(unittest.TestCase):
    MODEL_FOLDER = os.path.join(TESTS_PATH, "resources",
                                "mct_v1_model_pack.zip")

    @classmethod
    def setUpClass(cls):
        cls._model_folder_no_zip = cls.MODEL_FOLDER.rsplit(".zip", 1)[0]
        cls._folder_existed = os.path.exists(cls._model_folder_no_zip)
        cls.converter = legacy_converter.Converter(cls.MODEL_FOLDER, None)
        cls.cat = cls.converter.convert()

    @classmethod
    def tearDownClass(cls):
        if cls.MODEL_FOLDER.endswith(".zip"):
            folder = cls._model_folder_no_zip
            if os.path.exists(folder) and not cls._folder_existed:
                shutil.rmtree(folder)

    def test_creates_cat(self):
        self.assertIsInstance(self.cat, CAT)


class ConversionFromFolderTests(unittest.TestCase):
    MODEL_FOLDER = os.path.join(TESTS_PATH, "resources")
    VOCAB_NAME = 'mct_v1_vocab.dat'
    CDB_NAME = 'mct_v1_cdb.dat'
    CNF_NAME = 'mct_v1_cnf.json'

    @classmethod
    def setUpClass(cls):
        cls._def_vocab_name = legacy_converter.Converter.vocab_name
        cls._def_cdb_name = legacy_converter.Converter.cdb_name
        cls._def_cnf_name = legacy_converter.Converter.config_name
        legacy_converter.Converter.vocab_name = cls.VOCAB_NAME
        legacy_converter.Converter.cdb_name = cls.CDB_NAME
        legacy_converter.Converter.config_name = cls.CNF_NAME
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        legacy_converter.Converter.vocab_name = cls._def_vocab_name
        legacy_converter.Converter.cdb_name = cls._def_cdb_name
        legacy_converter.Converter.config_name = cls._def_cnf_name
