from medcat2.components.linking import context_based_linker
from medcat2.components.types import TrainableComponent
from medcat2.config import Config

import unittest

from ..platform.test_platform import FakeCDB


class TrainableLinkerTests(unittest.TestCase):
    cnf = Config()
    linker = context_based_linker.Linker(FakeCDB(cnf), None, cnf)

    def test_linker_is_trainable(self):
        self.assertIsInstance(self.linker, TrainableComponent)
