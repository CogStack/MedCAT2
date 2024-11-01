import os

from medcat2.utils.legacy import convert_vocab
from medcat2.vocab import Vocab

import unittest

TESTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", ".."))


class VocabConvertTests(unittest.TestCase):
    VOCAB_PATH = os.path.join(TESTS_PATH, "resources", "mct_v1_vocab.dat")
    EXP_WORDS = ("house", "dog")

    @classmethod
    def setUpClass(cls):
        cls.vocab = convert_vocab.get_vocab_from_old(cls.VOCAB_PATH)

    def test_conversion_works(self):
        self.assertIsInstance(self.vocab, Vocab)

    def test_conversion_recoers_words(self):
        for word in self.EXP_WORDS:
            with self.subTest(word):
                self.assertIn(word, self.vocab.vocab)
