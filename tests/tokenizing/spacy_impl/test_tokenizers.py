from typing import runtime_checkable

from medcat2.tokenizing import tokenizers
from medcat2.tokenizing.spacy_impl.tokenizers import (
    SpacyTokenizer, set_def_args_kwargs as sda_spacy)
from medcat2.tokenizing.regex_impl.tokenizer import (
    RegexTokenizer, set_def_args_kwargs as sda_regex)
from medcat2.config import Config

import unittest


class DefaultTokenizerInitTests(unittest.TestCase):
    default_provider = 'spacy'
    default_cls = SpacyTokenizer
    exp_num_def_tokenizers = 2
    set_def_args_kwargs = sda_spacy

    @classmethod
    def setUpClass(cls):
        cls.cnf = Config()
        cls.set_def_args_kwargs(cls.cnf)

    def test_has_default(self):
        avail_tokenizers = tokenizers.list_available_tokenizers()
        self.assertEqual(len(avail_tokenizers), self.exp_num_def_tokenizers)
        name, cls_name = [(t_name, t_cls) for t_name, t_cls in avail_tokenizers
                          if t_name == self.default_provider][0]
        self.assertEqual(name, self.default_provider)
        self.assertIs(cls_name, self.default_cls.__name__)

    def test_can_create_def_tokenizer(self):
        tokenizer = tokenizers.create_tokenizer(
            self.default_provider, *self.cnf.general.nlp.init_args,
            **self.cnf.general.nlp.init_kwargs)
        self.assertIsInstance(tokenizer,
                              runtime_checkable(tokenizers.BaseTokenizer))
        self.assertIsInstance(tokenizer, self.default_cls)


class DefaultTokenizerInitTests2(DefaultTokenizerInitTests):
    default_provider = 'regex'
    default_cls = RegexTokenizer
    set_def_args_kwargs = sda_regex
