"""This module exists purely to set the default arguments
in the config for the default tokenizer and the defualt
components creation.
"""
from typing import Optional

from medcat2.components.types import get_component_creator, CoreComponentType
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


def set_components_defaults(cdb: CDB, vocab: Optional[Vocab],
                            tokenizer: BaseTokenizer):
    for comp_name, comp_cnf in cdb.config.components:
        if not isinstance(comp_cnf, CoreComponentConfig):
            # e.g ignore order
            continue
        comp_cls = get_component_creator(CoreComponentType[comp_name],
                                         comp_cnf.comp_name)
        if hasattr(comp_cls, 'get_init_args'):
            comp_cnf.init_args = comp_cls.get_init_args(tokenizer, cdb, vocab)
        else:
            logger.warning(
                "The component %s (%s) does not define init arguments. "
                "You generally need to specify these with the class method "
                "get_init_args(BaseTokenizer, CDB, Vocab) -> list[Any]")
        if hasattr(comp_cls, 'get_init_kwargs'):
            comp_cnf.init_kwargs = comp_cls.get_init_kwargs(
                tokenizer, cdb, vocab)
        else:
            logger.warning(
                "The component %s (%s) does not define init keyword arguments."
                " You generally need to specify these with the class method "
                "get_init_kwargs(BaseTokenizer, CDB, Vocab) -> dict[str, Any]")


class OptionalPartNotInstalledException(ValueError):

    def __init__(self, *args):
        super().__init__(*args)
