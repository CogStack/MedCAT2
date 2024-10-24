from typing import Optional, Callable
import re
import os

import spacy
from spacy.tokenizer import Tokenizer  # type: ignore
from spacy.language import Language

from medcat2.tokenizing.tokens import MutableDocument
from medcat2.tokenizing.tokenizers import BaseTokenizer
from medcat2.tokenizing.spacy_impl.tokens import Document
from medcat2.tokenizing.spacy_impl.utils import ensure_spacy_model
from medcat2.config import Config


def spacy_split_all(nlp: Language, use_diacritics: bool) -> Tokenizer:

    token_characters = r'[^A-Za-z0-9\@]'

    if use_diacritics:
        token_characters = r'[^A-Za-zÀ-ÖØ-öø-ÿ0-9\@]'

    infix_re = re.compile(token_characters)
    suffix_re = re.compile(token_characters + r'$')
    prefix_re = re.compile(r'^' + token_characters)
    return Tokenizer(nlp.vocab,
                     rules={},
                     token_match=None,
                     prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer
                     )


class SpacyTokenizer(BaseTokenizer):

    def __init__(self, spacy_model_name: str,
                 spacy_disabled_components: list[str],
                 use_diacritics: bool,
                 max_document_lenth: int,
                 tokenizer_getter: Callable[[Language, bool], Tokenizer
                                            ] = spacy_split_all,
                 stopwords: Optional[set[str]] = None,):
        ensure_spacy_model(spacy_model_name)
        if stopwords is not None:
            lang_str = os.path.basename(spacy_model_name).split('_', 1)[0]
            cls = spacy.util.get_lang_class(lang_str)
            cls.Defaults.stop_words = set(stopwords)
        self._nlp = spacy.load(spacy_model_name,
                               disable=spacy_disabled_components)
        self._nlp.tokenizer = tokenizer_getter(self._nlp, use_diacritics)
        self._nlp.max_length = max_document_lenth

    def __call__(self, text: str) -> MutableDocument:
        return Document(self._nlp(text))


def set_def_args_kwargs(config: Config) -> None:
    nlp_cnf = config.general.nlp
    nlp_cnf.init_args = [
        nlp_cnf.modelname,
        nlp_cnf.disabled_components,
        config.general.diacritics,
        config.preprocessing.max_document_length,
    ]
    nlp_cnf.init_kwargs = {
        "stopwords": config.preprocessing.stopwords
    }
