"""This module exists purely to set the default arguments
in the config for the default tokenizer and the defualt
components creation.
"""
from typing import Optional

from medcat2.components.types import get_component_creator, CoreComponentType
from medcat2.tokenizing.tokenizers import BaseTokenizer, get_tokenizer_creator
from medcat2.config.config import CoreComponentConfig
from medcat2.config import Config
from medcat2.cdb import CDB
from medcat2.vocab import Vocab

import logging


logger = logging.getLogger(__name__)


def set_tokenizer_defaults(config: Config) -> None:
    nlp_cnf = config.general.nlp
    tok_cls = get_tokenizer_creator(nlp_cnf.provider)
    if hasattr(tok_cls, 'get_init_args'):
        nlp_cnf.init_args = tok_cls.get_init_args(config)
    else:
        logger.warning(
            "Could not set init arguments for tokenizer (%s). "
            "You generally need to specify these with the class method "
            "get_init_args(Config) -> list[Any].")
    if hasattr(tok_cls, 'get_init_kwargs'):
        nlp_cnf.init_kwargs = tok_cls.get_init_kwargs(config)
    else:
        logger.warning(
            "Could not set init keyword arguments for tokenizer (%s). "
            "You generally need to specify these with the class method "
            "get_init_kwargs(Config) -> dict[str, Any].")


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
