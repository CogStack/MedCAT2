"""This module exists purely to set the default arguments
in the config for the default tokenizer and the defualt
components creation.
"""
from medcat2.config import Config

import logging


logger = logging.getLogger(__name__)


def set_defaults(config: Config):
    nlp_cnf = config.general.nlp
    if nlp_cnf.provider == 'spacy':
        logging.debug("Setting default arguments for spacy constructor")
        from medcat2.tokenizing.spacy_impl.tokenizers import (
            set_def_args_kwargs)
        set_def_args_kwargs(config)
    # TODO: set defaults of components
    # for comp_name, comp_cnf in config.components:
    #     pass
