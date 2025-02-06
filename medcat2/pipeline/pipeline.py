from typing import Optional, Any, Type
import logging

from medcat2.tokenizing.tokenizers import BaseTokenizer, create_tokenizer
from medcat2.components.types import (CoreComponentType, create_core_component,
                                      CoreComponent)
from medcat2.components.addons.addons import AddonComponent, create_addon
from medcat2.tokenizing.tokens import (MutableDocument, MutableEntity,
                                       MutableToken)
from medcat2.vocab import Vocab
from medcat2.cdb import CDB
from medcat2.config import Config
from medcat2.config.config import ComponentConfig
from medcat2.utils.default_args import (set_tokenizer_defaults,
                                        set_components_defaults,
                                        set_addon_defaults)


logger = logging.getLogger(__name__)


class DelegatingTokenizer(BaseTokenizer):
    """A delegating tokenizer.

    This can be used to create a tokenizer with some preprocessing
    (i.e components) included.
    """

    def __init__(self, tokenizer: BaseTokenizer,
                 components: list[CoreComponent]):
        self.tokenizer = tokenizer
        self.components = components

    def create_entity(self, doc: MutableDocument,
                      token_start_index: int, token_end_index: int,
                      label: str) -> MutableEntity:
        return self.tokenizer.create_entity(
            doc, token_start_index, token_end_index, label)

    def entity_from_tokens(self, tokens: list[MutableToken]) -> MutableEntity:
        return self.tokenizer.entity_from_tokens(tokens)

    def __call__(self, text: str) -> MutableDocument:
        doc = self.tokenizer(text)
        for comp in self.components:
            doc = comp(doc)
        return doc

    @classmethod
    def get_init_args(cls, config: Config) -> list[Any]:
        return []

    @classmethod
    def get_init_kwargs(cls, config: Config) -> dict[str, Any]:
        return {}

    def get_doc_class(self) -> Type[MutableDocument]:
        return self.tokenizer.get_doc_class()

    def get_entity_class(self) -> Type[MutableEntity]:
        return self.tokenizer.get_entity_class()


class Pipeline:
    """The pipeline for the NLP process.

    This class is responsible to initial creation of the NLP document,
    as well as running through of all the components and addons.
    """

    def __init__(self, cdb: CDB, vocab: Optional[Vocab],
                 model_load_path: Optional[str]):
        # NOTE: this only sets the default arguments if the
        #       default tokenizer is used
        set_tokenizer_defaults(cdb.config)
        self.cdb = cdb
        self.config = self.cdb.config
        self._tokenizer = self._init_tokenizer()
        self._components: list[CoreComponent] = []
        self._addons: list[AddonComponent] = []
        set_components_defaults(cdb, vocab, self._tokenizer, model_load_path)
        set_addon_defaults(cdb, vocab, self._tokenizer, model_load_path)
        # NOTE: this only sets the default arguments if the
        #       a specific default component is used
        self._init_components()

    @property
    def tokenizer(self) -> BaseTokenizer:
        """The raw tokenizer (with no components)."""
        return self._tokenizer

    @property
    def tokenizer_with_tag(self) -> BaseTokenizer:
        """The tokenizer with the tagging component."""
        tag_comp = self.get_component(CoreComponentType.tagging)
        return DelegatingTokenizer(self.tokenizer, [tag_comp])

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

    def _init_component(self, comp_type: CoreComponentType) -> CoreComponent:
        comp_config: ComponentConfig = getattr(self.config.components,
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
        for addon_cnf in self.config.components.addons:
            addon = self._init_addon(addon_cnf)
            self._addons.append(addon)

    def _init_addon(self, cnf: ComponentConfig):
        return create_addon(cnf.comp_name, cnf,
                            *cnf.init_args, **cnf.init_kwargs)

    def get_doc(self, text: str) -> MutableDocument:
        """Get the document for this text.

        This essentially runs the tokenizer over the text.

        Args:
            text (str): The input text.

        Returns:
            MutableDocument: The resulting document.
        """
        doc = self._tokenizer(text)
        for comp in self._components:
            logger.info("Running component %s for %d of text (%s)",
                        comp.full_name, len(text), id(text))
            doc = comp(doc)
        for addon in self._addons:
            doc = addon(doc)
        return doc

    def entity_from_tokens(self, tokens: list[MutableToken]) -> MutableEntity:
        """Get the entity from the list of tokens.

        This effectively turns a list of (consecutive) documents
        into an entity.

        Args:
            tokens (list[MutableToken]): The tokens to use.

        Returns:
            MutableEntity: The resulting entity.
        """
        return self._tokenizer.entity_from_tokens(tokens)

    def get_component(self, ctype: CoreComponentType) -> CoreComponent:
        """Get the core component by the component type.

        Args:
            ctype (CoreComponentType): The core component type.

        Raises:
            ValueError: If no component by that type is found.

        Returns:
            CoreComponent: The corresponding core component.
        """
        for comp in self._components:
            if not comp.is_core() or not isinstance(comp, CoreComponent):
                continue
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
