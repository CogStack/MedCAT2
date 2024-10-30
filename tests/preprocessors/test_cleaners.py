from typing import runtime_checkable

from medcat2.preprocessors import cleaners
from medcat2.config import Config

import unittest


class AbstractionTests(unittest.TestCase):

    def test_config_fits_local_config(self):
        self.assertIsInstance(Config(), runtime_checkable(cleaners.LConfig))
