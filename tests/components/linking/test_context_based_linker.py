from medcat2.components.linking import context_based_linker
from medcat2.components import types
from medcat2.config import Config
from medcat2.vocab import Vocab
from medcat2.cdb.concepts import CUIInfo, NameInfo

import unittest

from ..helper import ComponentInitTests


class FakeDocument:

    def __init__(self, text):
        self.text = text


class FakeTokenizer:

    def __call__(selt, text: str) -> FakeDocument:
        return FakeDocument(text)


class FakeCDB:

    def __init__(self, config: Config):
        self.config = config
        self.cui2info: dict[str, CUIInfo] = dict()
        self.name2info: dict[str, NameInfo] = dict()
        self.name_separator: str

    def weighted_average_function(self, nr: int) -> float:
        return nr // 2.0


class LinkingInitTests(ComponentInitTests, unittest.TestCase):
    comp_type = types.CoreComponentType.linking
    default_cls = context_based_linker.Linker
    module = context_based_linker

    @classmethod
    def setUpClass(cls):
        cls.vocab = Vocab()
        cls.cdb = FakeCDB(Config())
        return super().setUpClass()

    @classmethod
    def set_def_args(cls):
        context_based_linker.set_def_args_kwargs(
            cls.cnf, cls.cdb, cls.vocab)
