from medcat2.config import config
from medcat2.storage.serialisables import Serialisable

import unittest


class ConfigTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cnf = config.Config()

    def test_is_serialisable(self):
        self.assertIsInstance(self.cnf, Serialisable)
