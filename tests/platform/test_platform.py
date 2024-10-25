from medcat2.platform import platform
from medcat2.vocab import Vocab
from medcat2.config import Config

from ..components.ner.test_vocab_based_ner import FakeCDB as BFakeCDB

import unittest


class FakeCDB(BFakeCDB):

    def __init__(self, config):
        super().__init__(config)
        self.token_counts: dict = {}
        self.cui2info: dict = {}
        self.name2info: dict = {}

    def weighted_average_function(self, v: int) -> float:
        return v / 2.0


class PlatformInitTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cnf = Config()
        cls.cdb = FakeCDB(cls.cnf)
        cls.vocab = Vocab()

    def test_can_create_platform(self):
        pf = platform.Platform(self.cdb, self.vocab)
        self.assertIsInstance(pf, platform.Platform)
