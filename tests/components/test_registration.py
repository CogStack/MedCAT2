from typing import Any

from medcat2.components import types
from medcat2.config.config import Config, CoreComponentConfig
from medcat2.cdb.cdb import CDB
from medcat2.vocab import Vocab
from medcat2.cat import CAT
from medcat2.tokenizing.tokenizers import BaseTokenizer
from .helper import FakeCDB, FVocab, FTokenizer

import unittest

# NOTE:
# The following 2 classes are (trivial!) examples of a component
# having been overwritten.
# After they are registered, they can be used just as the corresponding
# default implementations


class NoInitNER(types.AbstractCoreComponent):
    name = 'no-init-ner'

    def __call__(self, doc):
        return doc

    @classmethod
    def get_init_args(cls, tokenizer: BaseTokenizer, cdb: CDB, vocab: Vocab
                      ) -> list[Any]:
        return []

    @classmethod
    def get_init_kwargs(cls, tokenizer: BaseTokenizer, cdb: CDB, vocab: Vocab
                        ) -> dict[str, Any]:
        return {}


class WithInitNER(types.AbstractCoreComponent):
    name = 'with-init-ner'

    def __init__(self, tokenizer: BaseTokenizer,
                 cdb: CDB):
        super().__init__()
        self.tokenizer = tokenizer
        self.cdb = cdb

    def __call__(self, doc):
        return doc

    @classmethod
    def get_init_args(cls, tokenizer: BaseTokenizer, cdb: CDB, vocab: Vocab
                      ) -> list[Any]:
        return [tokenizer, cdb]

    @classmethod
    def get_init_kwargs(cls, tokenizer: BaseTokenizer, cdb: CDB, vocab: Vocab
                        ) -> dict[str, Any]:
        return {}


class RegisteredCompBaseTests(unittest.TestCase):
    TYPE = types.CoreComponentType.ner
    TO_REGISTR = NoInitNER

    @classmethod
    def setUpClass(cls):
        types.register_core_component(cls.TYPE, cls.TO_REGISTR.name,
                                      cls.TO_REGISTR)

    @classmethod
    def tearDownClass(cls):
        # unregister component
        types._CORE_REGISTRIES[cls.TYPE].unregister_component(
            cls.TO_REGISTR.name)


class CoreCompNoInitRegistrationTests(RegisteredCompBaseTests):
    CNF = Config()
    FCDB = FakeCDB(CNF)
    FVOCAB = FVocab()
    FTOK = FTokenizer()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.init_args = cls.TO_REGISTR.get_init_args(
            cls.FTOK, cls.FCDB, cls.FVOCAB)
        cls.init_kwargs = cls.TO_REGISTR.get_init_kwargs(
            cls.FTOK, cls.FCDB, cls.FVOCAB)

    def test_can_create_component(self):
        comp = types.create_core_component(self.TYPE, self.TO_REGISTR.name,
                                           *self.init_args, **self.init_kwargs)
        self.assertIsInstance(comp, self.TO_REGISTR)


class CoreCompWithInitRegistrationTests(CoreCompNoInitRegistrationTests):
    TO_REGISTR = WithInitNER


class CoreCompNoInitCATTests(RegisteredCompBaseTests):

    @classmethod
    def setUpClass(cls):
        # register
        super().setUpClass()
        # setup CDB/vocab
        cls.cdb = CDB(Config())
        cls.vocab = Vocab()
        # set name in component config
        comp_cnf: CoreComponentConfig = getattr(cls.cdb.config.components,
                                                cls.TYPE.name)
        comp_cnf.comp_name = cls.TO_REGISTR.name
        # NOTE: init arguments should be handled automatically

    def test_can_be_used_in_config(self):
        cat = CAT(cdb=self.cdb, vocab=self.vocab)
        self.assertIsInstance(cat, CAT)


class CoreCompWithInitCATTests(CoreCompNoInitCATTests):
    TO_REGISTR = WithInitNER
