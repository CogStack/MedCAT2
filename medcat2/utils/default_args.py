"""This module exists purely to set the default arguments
in the config for the default tokenizer and the defualt
components creation.
"""
from typing import cast

from medcat2.config.config import CoreComponentConfig
from medcat2.cdb import CDB

import logging


logger = logging.getLogger(__name__)


def set_defaults(cdb: CDB):
    nlp_cnf = cdb.config.general.nlp
    if nlp_cnf.provider == 'spacy':
        logging.debug("Setting default arguments for spacy constructor")
        from medcat2.tokenizing.spacy_impl.tokenizers import (
            set_def_args_kwargs)
        set_def_args_kwargs(cdb.config)
    for comp_name, comp_config in cdb.config.components:
        comp_cnf = cast(CoreComponentConfig, comp_config)
        if comp_cnf.comp_name != 'default':
            continue
        logging.debug("Setting default arguments for component '%s'",
                      comp_name)
        if comp_name == "tagging":
            from medcat2.components.tagging import tagger
            tagger.set_def_args_kwargs(cdb.config)
        # TODO: other components
