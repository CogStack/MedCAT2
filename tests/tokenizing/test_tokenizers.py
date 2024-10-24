from typing import runtime_checkable

from medcat2.tokenizing import tokenizers
from medcat2.tokenizing.spacy_impl.tokenizers import (
    SpacyTokenizer, set_def_args_kwargs)
from medcat2.config import Config

import unittest


class DefaultTokenizerInitTests(unittest.TestCase):
    default_provider = 'spacy'
    default_cls = SpacyTokenizer

    @classmethod
    def setUpClass(cls):
        cls.cnf = Config()
        set_def_args_kwargs(cls.cnf)

    def test_has_default(self):
        avail_tokenizers = tokenizers.list_available_tokenizers()
        self.assertEqual(len(avail_tokenizers), 1)
        name, cls_name = avail_tokenizers[0]
        self.assertEqual(name, self.default_provider)
        self.assertIs(cls_name, self.default_cls.__name__)

    def test_can_create_def_tokenizer(self):
        tokenizer = tokenizers.create_tokenizer(
            self.default_provider, *self.cnf.general.nlp.init_args,
            **self.cnf.general.nlp.init_kwargs)
        self.assertIsInstance(tokenizer,
                              runtime_checkable(tokenizers.BaseTokenizer))
