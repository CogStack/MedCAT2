from typing import Optional, Union
import os

import shutil
import logging

from medcat2.cdb import CDB
from medcat2.vocab import Vocab
from medcat2.config.config import Config, General, Preprocessing, CDBMaker
from medcat2.trainer import Trainer
from medcat2.components.types import CoreComponentType, TrainableComponent
from medcat2.storage.serialisers import serialise, AvailableSerialisers
from medcat2.storage.serialisers import deserialise
from medcat2.utils.fileutils import ensure_folder_if_parent
from medcat2.platform.platform import Platform
from medcat2.tokenizing.tokens import (MutableDocument, MutableEntity,
                                       MutableToken)
from medcat2.data.entities import Entity, Entities, OnlyCUIEntities
from medcat2.preprocessors.cleaners import prepare_name, NameDescriptor


logger = logging.getLogger(__name__)


DEFAULT_PACK_NAME = "medcat2_model_pack"


class CAT:
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
            self._trainer = Trainer(self.cdb, self.__call__,
                                    self.unlink_concept_name,
                                    self.add_and_train_concept)
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
        return cat

    @property
    def _pn_configs(self) -> tuple[General, Preprocessing, CDBMaker]:
        return (self.config.general, self.config.preprocessing,
                self.config.cdb_maker)

    def unlink_concept_name(self, cui: str, name: str,
                            preprocessed_name: bool = False) -> None:
        """Unlink a concept name from the CUI (or all CUIs if full_unlink),
        removes the link from the Concept Database (CDB). As a consequence
        medcat will never again link the `name` to this CUI - meaning the
        name will not be detected as a concept in the future.

        Args:
            cui (str):
                The CUI from which the `name` will be removed.
            name (str):
                The span of text to be removed from the linking dictionary.
            preprocessed_name (bool):
                Whether the name being used is preprocessed.

        Examples:

            >>> # To never again link C0020538 to HTN
            >>> cat.unlink_concept_name('C0020538', 'htn', False)
        """

        cuis = [cui]
        if preprocessed_name:
            names: dict[str, NameDescriptor] = {
                name: NameDescriptor([], set(), name, name.isupper())}
        else:
            names = prepare_name(name, self._platform.tokenizer, {},
                                 self._pn_configs)

        # If full unlink find all CUIs
        if self.config.general.full_unlink:
            logger.warning("In the config `full_unlink` is set to `True`. "
                           "Thus removing all CUIs linked to the specified "
                           "name (%s)", name)
            for n in names:
                if n not in self.cdb.name2info:
                    continue
                cuis.extend(self.cdb.name2info[n].cuis)

        # Remove name from all CUIs
        for c in cuis:
            self.cdb._remove_names(cui=c, names=names.keys())

    def add_and_train_concept(self,
                              cui: str,
                              name: str,
                              mut_doc: Optional[MutableDocument] = None,
                              mut_entity: Optional[
                                  Union[list[MutableToken],
                                        MutableEntity]] = None,
                              ontologies: set[str] = set(),
                              name_status: str = 'A',
                              type_ids: set[str] = set(),
                              description: str = '',
                              full_build: bool = True,
                              negative: bool = False,
                              devalue_others: bool = False,
                              do_add_concept: bool = True) -> None:
        r"""Add a name to an existing concept, or add a new concept, or do not
        do anything if the name or concept already exists. Perform training if
        spacy_entity and spacy_doc are set.

        Args:
            cui (str):
                CUI of the concept.
            name (str):
                Name to be linked to the concept (in the case of MedCATtrainer
                this is simply the selected value in text, no preprocessing or
                anything needed).
            mut_doc (Optional[MutableDocument]):
                Spacy representation of the document that was manually
                annotated.
            mut_entity (mut_entity: Optional[Union[list[MutableToken],
                                                   MutableEntity]]):
                Given the spacy document, this is the annotated span of text -
                list of annotated tokens that are marked with this CUI.
            ontologies (Set[str]):
                ontologies in which the concept exists (e.g. SNOMEDCT, HPO)
            name_status (str):
                One of `P`, `N`, `A`
            type_ids (Set[str]):
                Semantic type identifier (have a look at TUIs in UMLS or
                SNOMED-CT)
            description (str):
                Description of this concept.
            full_build (bool):
                If True the dictionary self.addl_info will also be populated,
                contains a lot of extra information about concepts, but can be
                very memory consuming. This is not necessary for normal
                functioning of MedCAT (Default Value `False`).
            negative (bool):
                Is this a negative or positive example.
            devalue_others (bool):
                If set, cuis to which this name is assigned and are not `cui`
                will receive negative training given that negative=False.
            do_add_concept (bool):
                Whether to add concept to CDB.
        """
        names = prepare_name(name, self._platform.tokenizer, {},
                             self._pn_configs)
        if (not names and cui not in self.cdb.cui2info and
                name_status == 'P'):
            logger.warning(
                "No names were able to be prepared in "
                "CAT.add_and_train_concept method. As such no preferred name "
                "will be able to be specifeid. The CUI: '%s' and raw name: "
                "'%s'", cui, name)
        # Only if not negative, otherwise do not add the new name if in fact
        # it should not be detected
        if do_add_concept and not negative:
            self.cdb._add_concept(cui=cui, names=names, ontologies=ontologies,
                                  name_status=name_status, type_ids=type_ids,
                                  description=description,
                                  full_build=full_build)

        if mut_entity is None or mut_doc is None:
            return
        linker = self._platform.get_component(
            CoreComponentType.linking)
        if not isinstance(linker, TrainableComponent):
            logger.warning(
                "Linker cannot be trained during add_and_train_concept"
                "because it has no train method: %s", linker)
        else:
            # Train Linking
            if isinstance(mut_entity, list):
                mut_entity = self._platform.entity_from_tokens(mut_entity)
            linker.train(cui=cui, entity=mut_entity, doc=mut_doc,
                         negative=negative, names=names)

            if not negative and devalue_others:
                # Find all cuis
                cuis = set()
                for n in names:
                    if n in self.cdb.name2info:
                        info = self.cdb.name2info[n]
                        cuis.update(info.cuis)
                # Remove the cui for which we just added positive training
                if cui in cuis:
                    cuis.remove(cui)
                # Add negative training for all other CUIs that link to
                # these names
                for _cui in cuis:
                    linker.train(cui=_cui, entity=mut_entity, doc=mut_doc,
                                 negative=True)
