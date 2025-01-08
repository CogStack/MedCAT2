"""This module exists purely to set the default arguments
in the config for the default tokenizer and the defualt
components creation.
"""
from typing import Optional

from medcat2.tokenizing.tokenizers import BaseTokenizer
from medcat2.config.config import CoreComponentConfig
from medcat2.config import Config
from medcat2.cdb import CDB
from medcat2.vocab import Vocab

import logging


logger = logging.getLogger(__name__)


def set_tokenizer_defaults(config: Config) -> None:
    nlp_cnf = config.general.nlp
    if nlp_cnf.provider == 'spacy':
        logging.debug("Setting default arguments for spacy constructor")
        try:
            from medcat2.tokenizing.spacy_impl.tokenizers import (
                set_def_args_kwargs)
        except ModuleNotFoundError as err:
            raise OptionalPartNotInstalledException(
                "spacy-specific parts of the library are optional. "
                "You need to specify to install these optional extras "
                "explicitly (i.e `pip install medcat2[spacy]`)."
            ) from err
        set_def_args_kwargs(config)
    elif nlp_cnf.provider == 'regex':
        logging.debug("Setting default arguments for regex constructor")
        from medcat2.tokenizing.regex_impl.tokenizer import (
            set_def_args_kwargs)
        set_def_args_kwargs(config)
    else:
        logger.warning("Could not set default tokenizer arguments for "
                       "toknizer '%s'. It must be a custom tokenizer. "
                       "You may need to specify the default arguments "
                       "at `config.general.nlp.init_args` and "
                       "`config.general.nlp.init_kwargs` manually.",)


# NOTE: this method does dynamic imports so that
#       we don't need to import these parts if we don't
#       use them (i.e non-default components are used).
def set_components_defaults(cdb: CDB, vocab: Optional[Vocab],
                            tokenizer: BaseTokenizer):
    for comp_name, comp_cnf in cdb.config.components:
        if not isinstance(comp_cnf, CoreComponentConfig):
            # e.g ignore order
            continue
        if comp_cnf.comp_name != 'default':
            continue
        logging.debug("Setting default arguments for component '%s'",
                      comp_name)
        if comp_name == "tagging":
            from medcat2.components.tagging import tagger
            tagger.set_def_args_kwargs(cdb.config)
        if comp_name == 'token_normalizing':
            from medcat2.components.normalizing import normalizer
            normalizer.set_default_args(cdb.config, tokenizer,
                                        cdb.token_counts, vocab)
        if comp_name == 'ner':
            from medcat2.components.ner import vocab_based_ner
            vocab_based_ner.set_def_args_kwargs(cdb.config, tokenizer, cdb)
        if comp_name == 'linking':
            from medcat2.components.linking import context_based_linker
            context_based_linker.set_def_args_kwargs(cdb.config, cdb, vocab)


class OptionalPartNotInstalledException(ValueError):

    def __init__(self, *args):
        super().__init__(*args)
