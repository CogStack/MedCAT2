import os
import json

from medcat2.trainer import Trainer
from medcat2.config import Config
from medcat2.vocab import Vocab
from medcat2.data.mctexport import MedCATTrainerExport
from medcat2.cat import CAT

import unittest

import shutil
import random
import pandas as pd

from .platform.test_platform import FakeCDB
from .utils.legacy.test_convert_config import TESTS_PATH


class TrainerTestsBase(unittest.TestCase):
    DATA_CNT = 14
    TRAIN_DATA = [
        "TEXT#{num}" for num in range(DATA_CNT)
    ]
    DATA_GEN = (dp for dp in TRAIN_DATA)

    @classmethod
    def setUpClass(cls):
        cls.cnf = Config()
        cls.cdb = FakeCDB(cls.cnf)
        cls.vocab = Vocab()
        cls.trainer = Trainer(cls.cdb,
                              cls.caller, cls.unlinker, cls.adder)

    def setUp(self):
        self.cnf = Config()
        self.cdb.config = self.cnf
        self.trainer.config = self.cnf

    @classmethod
    def caller(cls, text: str):
        return text

    @classmethod
    def unlinker(cls, *args, **kwargs):
        return

    @classmethod
    def adder(cls, *args, **kwargs):
        return

    def assert_remembers_training_data(self,
                                       num_docs: int,
                                       num_epochs: int,
                                       unsup: bool = True,
                                       exp_total: int = 1):
        if unsup:
            trained = self.cnf.meta.unsup_trained
        else:
            trained = self.cnf.meta.sup_trained
        self.assertEqual(len(trained), exp_total)
        last_trained = trained[0]
        self.assertEqual(last_trained.num_docs, num_docs)
        self.assertEqual(last_trained.num_epochs, num_epochs)


class TrainerUnsupervisedTests(TrainerTestsBase):
    NEPOCHS = 1
    UNSUP = True

    def train(self, data):
        if self.UNSUP:
            return self.trainer.train_unsupervised(data, nepochs=self.NEPOCHS)
        else:
            return self.trainer.train_supervised_raw(data)

    def test_training_gets_remembered_list(self):
        self.train(self.TRAIN_DATA)
        self.assert_remembers_training_data(self.DATA_CNT, self.NEPOCHS,
                                            unsup=self.UNSUP)

    def test_training_gets_remembered_gen(self):
        self.train(self.DATA_GEN)
        self.assert_remembers_training_data(self.DATA_CNT, self.NEPOCHS,
                                            unsup=self.UNSUP)

    def test_training_gets_remembered_multi(self, repeats: int = 3):
        for _ in range(repeats):
            self.train(self.TRAIN_DATA)
        self.assert_remembers_training_data(self.DATA_CNT, self.NEPOCHS,
                                            exp_total=repeats,
                                            unsup=self.UNSUP)


class TrainerSupervisedTests(TrainerUnsupervisedTests):
    DATA_CNT = 1
    UNSUP = False
    TRAIN_DATA: MedCATTrainerExport = {
        "projects": [
            {
                'cuis': '',
                'tuis': '',
                'documents': [
                    {
                        'id': "P1D1",
                        'name': "Project#1Doc#1",
                        'last_modified': 'N/A',
                        'text': 'Some long text',
                        'annotations': [
                            {
                                'cui': "C1",
                                'start': 0,
                                'end': 4,
                                'value': 'SOME',
                            }
                        ]
                    }
                ],
                'id': "PID#1",
                'name': "PROJECT#1",
            }
        ]
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_training_gets_remembered_gen(self):
        pass  # NOTE: no generation for supervised training


class FromSratchBase(unittest.TestCase):
    TRAINED_MODEL_PATH = os.path.join(TESTS_PATH, 'resources',
                                      'mct2_model_pack.zip')
    RNG_SEED = 42

    @classmethod
    def setUpClass(cls):
        cls._model_folder_no_zip = cls.TRAINED_MODEL_PATH.rsplit(".zip", 1)[0]
        cls._folder_existed = os.path.exists(cls._model_folder_no_zip)
        cls.model = CAT.load_model_pack(cls.TRAINED_MODEL_PATH)
        if cls.model.config.components.linking.train:
            print("TRAINING WAS ENABLE! NEED TO DISABLE")
            cls.model.config.components.linking.train = False
        cls.model.cdb.reset_training()
        random.seed(cls.RNG_SEED)

    @classmethod
    def tearDownClass(cls):
        if cls.TRAINED_MODEL_PATH.endswith(".zip"):
            folder = cls._model_folder_no_zip
            if os.path.exists(folder) and not cls._folder_existed:
                shutil.rmtree(folder)


class TrainFromScratchTests(FromSratchBase):
    UNSUP_DATA_PATH = os.path.join(
        TESTS_PATH, "resources", "selfsupervised_data.txt")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.all_words = list(cls.model.vocab.vocab.keys())
        cls.all_concepts = [(cui, cls.model.cdb.get_name(cui))
                            for cui in cls.model.cdb.cui2info]
        cls.model.trainer.train_unsupervised(cls.get_data())

    @classmethod
    def get_data(cls) -> list[str]:
        df = pd.read_csv(cls.UNSUP_DATA_PATH)
        return df['text'].tolist()

    def test_can_train_unsupervised(self):
        for cui, _ in self.all_concepts:
            with self.subTest(cui):
                self.assertGreater(self.model.cdb.cui2info[cui].count_train, 0)


class TrainFromScratchSupervisedTests(TrainFromScratchTests):
    SUP_DATA_PATH = os.path.join(
        TESTS_PATH, "resources", "supervised_mct_export.json"
    )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cnts_before = {
            cui: info.count_train
            for cui, info in cls.model.cdb.cui2info.items()
        }
        cls.model.trainer.train_supervised_raw(
            cls.get_sup_data()
        )

    @classmethod
    def get_sup_data(cls) -> MedCATTrainerExport:
        with open(cls.SUP_DATA_PATH) as f:
            return json.load(f)

    def test_has_trained_all(self):
        for cui, prev_count in self.cnts_before.items():
            with self.subTest(cui):
                info = self.model.cdb.cui2info[cui]
                self.assertGreater(info.count_train, prev_count)
