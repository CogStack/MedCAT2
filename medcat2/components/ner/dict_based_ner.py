from typing import Any, Optional

import logging
from medcat2.tokenizing.tokens import MutableDocument
from medcat2.components.types import CoreComponentType, AbstractCoreComponent
from medcat2.components.ner.vocab_based_annotator import maybe_annotate_name
from medcat2.tokenizing.tokenizers import BaseTokenizer
from medcat2.vocab import Vocab
from medcat2.cdb import CDB

from ahocorasick import Automaton


logger = logging.getLogger(__name__)


class NER(AbstractCoreComponent):
    name = 'cat_dict_ner'

    def __init__(self, tokenizer: BaseTokenizer,
                 cdb: CDB) -> None:
        self.tokenizer = tokenizer
        self.cdb = cdb
        self.config = self.cdb.config
        self.automaton = Automaton()
        self._rebuild_automaton()

    def _rebuild_automaton(self):
        # print("[D] BUILD AUTOMATON")
        # NOTE: every time the CDB changes (is dirtied)
        #       this will be recalculated
        logger.info("Rebuilding NER automaton (Aho-Corasick)")
        self.automaton.clear()
        # NOTE: we do not need name info for NER - only for linking
        for name in self.cdb.name2info.keys():
            clean_name = name.replace(self.config.general.separator, " ")
            if clean_name in self.automaton:
                # no need to duplicate
                continue
            # print("[D] ADD clean word", clean_name)
            self.automaton.add_word(clean_name, clean_name)
        self.automaton.make_automaton()
        # print("[D] AUTOMATON", self.automaton)

    def get_type(self) -> CoreComponentType:
        return CoreComponentType.ner

    def __call__(self, doc: MutableDocument) -> MutableDocument:
        """Detect candidates for concepts - linker will then be able
        to do the rest. It adds `entities` to the doc.entities and each
        entity can have the entity.link_candidates - that the linker
        will resolve.

        Args:
            doc (MutableDocument):
                Spacy document to be annotated with named entities.

        Returns:
            doc (MutableDocument):
                Spacy document with detected entities.
        """
        # print("[D] __call__")
        if self.cdb.has_changed_names:
            # print("UNDIRTY[NAMES]!")
            self.cdb._undirty()
            self._rebuild_automaton()
        text = doc.base.text.lower()
        for end_idx, raw_name in self.automaton.iter(text):
            start_idx = end_idx - len(raw_name) + 1
            # print("[D] FOUND", raw_name, "@", start_idx, end_idx)
            cur_tokens = doc.get_tokens(start_idx, end_idx)
            if not isinstance(cur_tokens, list):
                # NOTE: this shouldn't really happen since
                #       there should be no entities defined
                #       before the NER step.
                #       But we will (at least for now) still handler this
                cur_tokens = list(cur_tokens)
            # print("[D] MAN")
            preprocessed_name = raw_name.replace(
                ' ', self.config.general.separator)
            maybe_annotate_name(self.tokenizer, preprocessed_name, cur_tokens,
                                doc, self.cdb, self.config)
        return doc

    @classmethod
    def get_init_args(cls, tokenizer: BaseTokenizer, cdb: CDB, vocab: Vocab,
                      model_load_path: Optional[str]) -> list[Any]:
        return [tokenizer, cdb]

    @classmethod
    def get_init_kwargs(cls, tokenizer: BaseTokenizer, cdb: CDB, vocab: Vocab,
                        model_load_path: Optional[str]) -> dict[str, Any]:
        return {}
