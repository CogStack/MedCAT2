from typing import Optional

from medcat2.tokenizing.tokenizers import BaseTokenizer, create_tokenizer
from medcat2.components.types import (CoreComponentType, create_core_component,
                                      BaseComponent)
from medcat2.tokenizing.tokens import (MutableDocument, MutableEntity,
                                       MutableToken)
from medcat2.vocab import Vocab
from medcat2.cdb import CDB
from medcat2.config.config import CoreComponentConfig
from medcat2.utils.default_args import (set_tokenizer_defaults,
                                        set_components_defaults)


class Platform:

    def __init__(self, cdb: CDB, vocab: Optional[Vocab]):
        # NOTE: this only sets the default arguments if the
        #       default tokenizer is used
        set_tokenizer_defaults(cdb.config)
        self.cdb = cdb
        self.config = self.cdb.config
        self._tokenizer = self._init_tokenizer()
        self._components: list[BaseComponent] = []
        set_components_defaults(cdb, vocab, self._tokenizer)
        # NOTE: this only sets the default arguments if the
        #       a specific default component is used
        self._init_components()

    @property
    def tokenizer(self) -> BaseTokenizer:
        return self._tokenizer

    def _init_tokenizer(self) -> BaseTokenizer:
        nlp_cnf = self.config.general.nlp
        try:
            return create_tokenizer(nlp_cnf.provider, *nlp_cnf.init_args,
                                    **nlp_cnf.init_kwargs)
        except TypeError as type_error:
            if nlp_cnf.provider == 'spacy':
                raise type_error
            raise IncorrectArgumentsForTokenizer(
                nlp_cnf.provider) from type_error

    def _init_component(self, comp_type: CoreComponentType) -> BaseComponent:
        comp_config: CoreComponentConfig = getattr(self.config.components,
                                                   comp_type.name)
        comp_name = comp_config.comp_name
        try:
            comp = create_core_component(comp_type, comp_name,
                                         *comp_config.init_args,
                                         **comp_config.init_kwargs)
        except TypeError as type_error:
            if comp_name == 'default':
                raise type_error
            raise IncorrectArgumentsForComponent(
                comp_type, comp_name) from type_error
        return comp

    def _init_components(self) -> None:
        for cct_name in self.config.components.comp_order:
            comp = self._init_component(CoreComponentType[cct_name])
            self._components.append(comp)

    def get_doc(self, text: str) -> MutableDocument:
        doc = self._tokenizer(text)
        for comp in self._components:
            doc = comp(doc)
        return doc

    def entity_from_tokens(self, tokens: list[MutableToken]) -> MutableEntity:
        return self._tokenizer.entity_from_tokens(tokens)

    def get_component(self, ctype: CoreComponentType) -> BaseComponent:
        for comp in self._components:
            if comp.get_type() is ctype:
                return comp
        raise ValueError(f"No component found of type {ctype}")


class IncorrectArgumentsForTokenizer(TypeError):

    def __init__(self, provider: str):
        super().__init__(
            f"Incorrect arguments for tokenizer ({provider}). Did you forget "
            "to set `config.general.nlp.init_args` or "
            "`config.general.nlp.init_kwargs`? When using a custom tokenizer, "
            "you need to specify the arguments required for construction "
            "manually.")


class IncorrectArgumentsForComponent(TypeError):

    def __init__(self, comp_type: CoreComponentType, comp_name: str):
        super().__init__(
            f"Incorrect arguments for core component {comp_type.name} "
            f"({comp_name}). Did you forget to set "
            f"`config.components.{comp_type.name}.init_args` and/or "
            f"`config.components.{comp_type.name}.init_kwargs`? "
            "When using a custom component, you need to specify the arguments"
            "required or construction manually.")
