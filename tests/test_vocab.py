import os

from medcat2.vocab import Vocab
from medcat2.storage.serialisers import get_serialiser

import numpy as np

import unittest
import tempfile


class VocabCreationTests(unittest.TestCase):
    word1 = {
        "word": "WORD1", "cnt": 1,
    }
    word2 = {
        "word": "WORD2", "cnt": 1,
        "vec": np.array([0,   100, 100]),
    }
    word3 = {
        "word": "WORD3", "cnt": 1,
        "vec": np.array([100,   0, 100]),
        "replace": True,
    }
    word4 = {
        "word": "WORD4", "cnt": 1,
        "vec": np.array([100, 100,   0]),
        "replace": False,
    }
    all_words = [
        word1, word2, word3, word4,
    ]

    def setUp(self):
        self._temp_file = tempfile.NamedTemporaryFile()
        self.temp_file = self._temp_file.name
        self.vocab = Vocab()

    def tearDown(self):
        self._temp_file.close()

    def test_does_not_have_words_by_default(self):
        for word in self.all_words:
            self.assertNotIn(word["word"], self.vocab)

    def test_remembers_words(self):
        for word in self.all_words:
            self.vocab.add_word(**word)
            self.assertIn(word["word"], self.vocab)

    def test_can_overwrite_word(self):
        word = self.word3.copy()  # allows to overwrite
        # print("WORD", word)
        self.vocab.add_word(**word)
        word_cp = word.copy()
        word_cp["vec"] = 2 * word["vec"]
        self.vocab.add_word(**word_cp)
        got_vec = self.vocab.vec(word["word"])
        self.assertTrue(np.all(got_vec == word_cp["vec"]))


class VocabSaveTests(unittest.TestCase):
    serialiser = get_serialiser('dill')
    all_words = VocabCreationTests.all_words

    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.target_file = os.path.join(cls._temp_dir.name, "vocab.dat")
        cls.vocab = Vocab()
        for word in cls.all_words:
            cls.vocab.add_word(**word)

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def _serialise_default(self):
        self.serialiser.serialise(self.vocab, self.target_file)

    def test_can_serialise(self):
        self._serialise_default()
        self.assertTrue(os.path.exists(self.target_file))

    def assert_equal_dicts(self, d1: dict, d2: dict, path: str = '') -> None:
        keys1, keys2 = d1.keys(), d2.keys()
        if keys1 != keys2:
            raise AssertionError(
                f"Incompatible keys for vocabs: {keys1} vs {keys2}")
        for k, v1 in d1.items():
            v2 = d2[k]
            if isinstance(v2, dict):
                self.assert_equal_dicts(v1, v2,  f"{path}.{k}" if path else k)
                continue
            if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                is_eq = np.all(v1 == v2)
            else:
                is_eq = v1 == v2
            if not is_eq:
                raise AssertionError(
                    f"Incompatible values for {k}: {v1} vs {v2}")

    def assert_vocab_equals(self, vocab1: Vocab, vocab2: Vocab):
        self.assert_equal_dicts(vocab1.__dict__, vocab2.__dict__)

    def test_can_deserialise(self):
        self._serialise_default()
        loaded = self.serialiser.deserialise(self.target_file)
        self.assertIsInstance(loaded, Vocab)
        self.assert_vocab_equals(self.vocab, loaded)

    def test_deserialised_has_all_words(self):
        self._serialise_default()
        vocab = self.serialiser.deserialise(self.target_file)
        for word in self.all_words:
            with self.subTest(word):
                self.assertIn(word["word"], vocab)
