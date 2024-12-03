from typing import Optional, Union, Any
import os

import shutil
import logging

from medcat2.cdb import CDB
from medcat2.vocab import Vocab
from medcat2.config.config import Config
from medcat2.trainer import Trainer
from medcat2.storage.serialisers import serialise, AvailableSerialisers
from medcat2.storage.serialisers import deserialise
from medcat2.storage.serialisables import AbstractSerialisable
from medcat2.utils.fileutils import ensure_folder_if_parent
from medcat2.platform.platform import Platform
from medcat2.tokenizing.tokens import MutableDocument, MutableEntity
from medcat2.data.entities import Entity, Entities, OnlyCUIEntities


logger = logging.getLogger(__name__)


DEFAULT_PACK_NAME = "medcat2_model_pack"


class CAT(AbstractSerialisable):
    """This is a collection of serialisable model parts.
    """

    def __init__(self,
                 cdb: CDB,
                 vocab: Union[Vocab, None] = None,
                 config: Optional[Config] = None,
                 #  meta_cats: List[MetaCAT] = [],
                 #  addl_ner: Union[TransformersNER, List[TransformersNER]]
                 ) -> None:
        self.cdb = cdb
        self.vocab = vocab
        # ensure  config
        if config is None and self.cdb.config is None:
            raise ValueError("Need to specify a config for either CDB or CAT")
        elif config is None:
            config = cdb.config
        elif config is not None:
            self.cdb.config = config
        self.config = config

        self._trainer: Optional[Trainer] = None
        self._platform = Platform(self.cdb, self.vocab)

    @classmethod
    def get_init_attrs(cls) -> list[str]:
        return ['cdb', 'vocab']

    @classmethod
    def ignore_attrs(cls) -> list[str]:
        return [
            '_trainer',  # recreate if nededed
            '_platform',  # need to recreate regardless
            'config',  # will be loaded along with CDB
        ]

    def __call__(self, text: str) -> Optional[MutableDocument]:
        return self._platform.get_doc(text)

    def get_entities(self,
                     text: str,
                     only_cui: bool = False,
                     # TODO : addl_info
                     ) -> Union[dict, Entities, OnlyCUIEntities]:
        doc = self(text)
        if not doc:
            return {}
        return self._doc_to_out(doc, only_cui=only_cui)

    def _get_entity(self, ent: MutableEntity,
                    doc_tokens: list[str],
                    cui: str) -> Entity:
        context_left = self.config.annotation_output.context_left
        context_right = self.config.annotation_output.context_right

        if context_left > 0 and context_right > 0:
            left_s = max(ent.base.start_index - context_left, 0)
            left_e = ent.base.start_index
            left_context = doc_tokens[left_s:left_e]
            right_s = ent.base.end_index
            right_e = min(ent.base.end_index + context_right, len(doc_tokens))
            right_context = doc_tokens[right_s:right_e]
            ent_s, ent_e = ent.base.start_index, ent.base.end_index
            center_context = doc_tokens[ent_s:ent_e]
        else:
            left_context = []
            right_context = []
            center_context = []

        return {
            'pretty_name': self.cdb.get_name(cui),
            'cui': cui,
            'type_ids': list(self.cdb.cui2info[cui].type_ids),
            'source_value': ent.base.text,
            'detected_name': str(ent.detected_name),
            'acc': ent.context_similarity,
            'context_similarity': ent.context_similarity,
            'start': ent.base.start_char_index,
            'end': ent.base.end_char_index,
            # TODO: add additional info (i.e mappings)
            # for addl in addl_info:
            #     tmp = self.cdb.addl_info.get(addl, {}).get(cui, [])
            #     out_ent[addl.split("2")[-1]] = list(tmp) if type(tmp) is
            # set else tmp
            'id': ent.id,
            # TODO: add met annotations
            # if hasattr(ent._, 'meta_anns') and ent._.meta_anns:
            #     out_ent['meta_anns'] = ent._.meta_anns
            'meta_anns': {},
            'context_left': left_context,
            'context_center': center_context,
            'context_right': right_context,
        }

    def _doc_to_out_entity(self, ent: MutableEntity,
                           doc_tokens: list[str],
                           only_cui: bool,
                           ) -> tuple[int, Union[Entity, str]]:
        cui = str(ent.cui)
        if not only_cui:
            out_ent = self._get_entity(ent, doc_tokens, cui)
            return ent.id, out_ent
        else:
            return ent.id, cui

    def _doc_to_out(self,
                    doc: MutableDocument,
                    only_cui: bool,
                    # addl_info: List[str], # TODO
                    out_with_text: bool = False
                    ) -> Union[Entities, OnlyCUIEntities]:
        out: Union[Entities, OnlyCUIEntities] = {'entities': {},
                                                 'tokens': []}  # type: ignore
        cnf_annotation_output = self.config.annotation_output
        _ents = doc.final_ents

        if cnf_annotation_output.lowercase_context:
            doc_tokens = [tkn.base.text_with_ws.lower() for tkn in list(doc)]
        else:
            doc_tokens = [tkn.base.text_with_ws for tkn in list(doc)]

        for _, ent in enumerate(_ents):
            ent_id, ent_dict = self._doc_to_out_entity(ent, doc_tokens,
                                                       only_cui)
            # NOTE: the types match - not sure why mypy is having issues
            out['entities'][ent_id] = ent_dict  # type: ignore

        if cnf_annotation_output.include_text_in_output or out_with_text:
            out['text'] = doc.base.text
        return out

    @property
    def trainer(self):
        if not self._trainer:
            self._trainer = Trainer(self.cdb, self.__call__, self._platform)
        return self._trainer

    def save_model_pack(
            self, target_folder: str, pack_name: str = DEFAULT_PACK_NAME,
            serialiser_type: Union[str, AvailableSerialisers] = 'dill'
            ) -> None:
        self.config.meta.mark_saved_now()
        # figure out the location/folder of the saved files
        model_pack_path = os.path.join(target_folder, pack_name)
        # ensure target folder and model pack folder exist
        ensure_folder_if_parent(model_pack_path)
        # serialise
        serialise(serialiser_type, self, model_pack_path)
        # zip everything
        shutil.make_archive(model_pack_path, 'zip', root_dir=model_pack_path)

    @classmethod
    def load_model_pack(
            cls, model_pack_path: str,
            serialiser_type: Union[str, AvailableSerialisers] = 'dill'
            ) -> 'CAT':
        if model_pack_path.endswith(".zip"):
            folder_path = model_pack_path.rsplit(".zip", 1)[0]
            if not os.path.exists(folder_path):
                logger.info("Unpacking model pack from %s to %s",
                            model_pack_path, folder_path)
                shutil.unpack_archive(model_pack_path,
                                      folder_path)
            model_pack_path = folder_path
        logger.info("Attempting to load model from file: %s",
                    model_pack_path)
        cat = deserialise(serialiser_type, model_pack_path)
        if not isinstance(cat, CAT):
            raise ValueError(f"Unable to load CAT. Got: {cat}")
        # NOTE: this should only be `True` during training
        #       but some models (especially converted ones)
        #       are saved with it set to True accidentally
        cat.config.components.linking.train = False
        return cat

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CAT):
            return False
        return (self.cdb == other.cdb and
                ((self.vocab is None and other.vocab is None)
                 or self.vocab == other.vocab))
