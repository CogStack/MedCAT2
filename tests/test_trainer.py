from medcat2.trainer import Trainer
from medcat2.config import Config
from medcat2.vocab import Vocab

import unittest

from .platform.test_platform import FakeCDB


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

    def test_training_gets_remembered_list(self):
        self.trainer.train_unsupervised(self.TRAIN_DATA, nepochs=self.NEPOCHS)
        self.assert_remembers_training_data(len(self.TRAIN_DATA), self.NEPOCHS,
                                            unsup=self.UNSUP)

    def test_training_gets_remembered_gen(self):
        self.trainer.train_unsupervised(self.DATA_GEN, nepochs=self.NEPOCHS)
        self.assert_remembers_training_data(self.DATA_CNT, self.NEPOCHS,
                                            unsup=self.UNSUP)

    def test_training_gets_remembered_multi(self, repeats: int = 3):
        for _ in range(repeats):
            self.trainer.train_unsupervised(self.TRAIN_DATA,
                                            nepochs=self.NEPOCHS)
        self.assert_remembers_training_data(len(self.TRAIN_DATA), self.NEPOCHS,
                                            exp_total=repeats,
                                            unsup=self.UNSUP)
