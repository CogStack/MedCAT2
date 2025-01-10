from medcat2.components.normalizing import normalizer
from medcat2.components import types
from medcat2.vocab import Vocab

import unittest

from ..helper import ComponentInitTests


class FakeDocument:

    def __init__(self, text):
        self.text = text


class FakeTokenizer:

    def __call__(selt, text: str) -> FakeDocument:
        return FakeDocument(text)


class NormaliserInitTests(ComponentInitTests, unittest.TestCase):
    comp_type = types.CoreComponentType.token_normalizing
    default_cls = normalizer.TokenNormalizer
    module = normalizer

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FakeTokenizer()
        cls.cdb_vocab = dict()
        cls.vocab = Vocab()
        return super().setUpClass()
