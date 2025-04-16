from typing import Iterable, Any, Collection

from medcat2.storage.serialisables import AbstractSerialisable
from medcat2.cdb.concepts import CUIInfo, NameInfo, TypeInfo
from medcat2.cdb.concepts import get_new_cui_info, get_new_name_info
from medcat2.cdb.concepts import reset_cui_training
from medcat2.utils.defaults import default_weighted_average, StatusTypes as ST
from medcat2.utils.hasher import Hasher
from medcat2.preprocessors.cleaners import NameDescriptor
from medcat2.config import Config

import logging


logger = logging.getLogger(__name__)


class CDB(AbstractSerialisable):

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cui2info: dict[str, CUIInfo] = {}
        self.name2info: dict[str, NameInfo] = {}
        self.type_id2info: dict[str, TypeInfo] = {}
        self.token_counts: dict[str, int] = {}
        self.addl_info: dict[str, Any] = {}
        self._subnames: set[str] = set()
        self.is_dirty = False
        self.has_changed_names = False

    @classmethod
    def get_init_attrs(cls) -> list[str]:
        return ['config']

    def _reset_subnames(self):
        logger.info("Resetting subnames")
        self._subnames.clear()
        for info in self.cui2info.values():
            self._subnames.update(info['subnames'])
        self.has_changed_names = False

    def has_subname(self, name: str) -> bool:
        """Whether the CDB has the specified subname.

        Args:
            name (str): The subname to check.

        Returns:
            bool: Whether the subname is present in this CDB.
        """
        if (self.has_changed_names or
                len(self._subnames) < len(self.name2info)):
            self._reset_subnames()
        return name in self._subnames

    def get_name(self, cui: str) -> str:
        """Returns preferred name if it exists, otherwise it will return
        the longest name assigned to the concept.

        Args:
            cui (str):
                Concept ID or unique identifier in this database.

        Returns:
            str: The name of the concept.
        """
        name = cui
        sep = self.config.general.separator

        if cui not in self.cui2info:
            return name
        cui_info = self.cui2info[cui]
        pref_name = cui_info['preferred_name']
        names = cui_info['names']
        if pref_name:
            name = pref_name
        elif names:
            # longest name
            raw_name = max(names, key=len)
            name = " ".join(str(raw_name).split(sep)).title()

        return name

    def weighted_average_function(self, step: int) -> float:
        """Get the weighted average for steop.

        Args:
            step (int): The steop.

        Returns:
            float: The weighted average.
        """
        return default_weighted_average(step)

    def add_types(self, types: Iterable[tuple[str, str]]) -> None:
        """Add type info to CDB.

        Args:
            types (Iterable[tuple[str, str]]): The raw type info.
        """
        for type_id, name in types:
            self.type_id2info[type_id] = TypeInfo(type_id, name)

    def add_names(self, cui: str, names: dict[str, NameDescriptor],
                  name_status: str = ST.AUTOMATIC, full_build: bool = False
                  ) -> None:
        """Adds a name to an existing concept.

        Args:
            cui (str):
                Concept ID or unique identifier in this database, all concepts
                that have the same CUI will be merged internally.
            names (dict[str, NameDescriptor]):
                Names for this concept, or the value that if found in free
                text can be linked to this concept. Names is an dict like:
                `{name: {'tokens': tokens, 'snames': snames,
                         'raw_name': raw_name}, ...}`
                Names should be generated by helper function
                'medcat.preprocessing.cleaners.prepare_name'
            name_status (str):
                One of `P`, `N`, `A`. Defaults to 'A'.
            full_build (bool):
                If True the dictionary self.addl_info will also be populated,
                contains a lot of extra information about concepts, but can be
                very memory consuming. This is not necessary for normal
                functioning of MedCAT (Default value `False`).
        """
        name_status = name_status.upper()
        if name_status not in ST.ALLOWED_STATUS:
            # Name status must be one of the three
            name_status = ST.AUTOMATIC

        self._add_concept(cui=cui, names=names, ontologies=set(),
                          name_status=name_status, type_ids=set(),
                          description='', full_build=full_build)

    def _add_concept_names(self, cui: str, names: dict[str, NameDescriptor],
                           name_status: str) -> None:
        cui_info = self.cui2info[cui]
        for name, in_name_info in names.items():
            # add name and synonyms
            cui_info['names'].add(name)
            cui_info['subnames'].update(in_name_info.snames)

            if name not in self.name2info:
                self.name2info[name] = get_new_name_info(name=name)
            # Add whether concept is uppercase
            name_info = self.name2info[name]
            name_info['is_upper'] = in_name_info.is_upper
            status_map = name_info['per_cui_status']
            if cui not in status_map:
                status_map[cui] = name_status
            elif name_status == ST.PRIMARY_STATUS_NO_DISAMB:
                # if this is primary, overwrite old status
                status_map[cui] = name_status
            # if already in status map and other status, leave it be

            # Add tokens to token counts
            for token in in_name_info.tokens:
                if token in self.token_counts:
                    self.token_counts[token] += 1
                else:
                    self.token_counts[token] = 1
            self._subnames.update(cui_info['subnames'])
            self.is_dirty = True

    def _add_full_build(self, cui: str, names: dict[str, NameDescriptor],
                        ontologies: set[str], description: str,
                        type_ids: set[str]) -> None:
        cui_info = self.cui2info[cui]
        # Use original_names as the base check because they must be added
        orig_names: set[str] = set([v.raw_name for v in names.values()])
        if cui_info['original_names'] is None:
            if ontologies:
                cui_info['in_other_ontology'] = ontologies
            cui_info['original_names'] = orig_names
        else:
            # Update existing ones
            if ontologies:
                ontos = cui_info['in_other_ontology']
                if ontos is None:
                    ontos = cui_info['in_other_ontology'] = set()
                ontos.update(ontologies)
            cui_info['original_names'].update(orig_names)
        if description:
            cui_info['description'] = description

        for type_id in type_ids:
            # Add type_id2cuis link
            if type_id not in self.type_id2info:
                # NOTE: most type IDs should be added at CDB creation and
                #       their names should be (manually?) added alongside
                self.type_id2info[type_id] = TypeInfo(type_id=type_id,
                                                      name='N/A')
            type_info = self.type_id2info[type_id]
            type_info.cuis.add(cui)

    def _add_concept(self,
                     cui: str,
                     names: dict[str, NameDescriptor],
                     ontologies: set[str],
                     name_status: str,
                     type_ids: set[str],
                     description: str,
                     full_build: bool = False) -> None:
        """Add a concept to internal Concept Database (CDB). Depending on what
        you are providing this will add a large number of properties for each
        concept.

        Args:
            cui (str):
                Concept ID or unique identifier in this database, all concepts
                that have the same CUI will be merged internally.
            names (dict[str, NameDescriptor]):
                Names for this concept, or the value that if found in free
                text can be linked to this concept. Names is a dict like:
                `{name: {'tokens': tokens, 'snames': snames,
                         'raw_name': raw_name}, ...}`
                Names should be generated by helper function
                'medcat.preprocessing.cleaners.prepare_name'
            ontologies (set[str]):
                ontologies in which the concept exists (e.g. SNOMEDCT, HPO)
            name_status (str):
                One of `P`, `N`, `A`
            type_ids (set[str]):
                Semantic type identifier (have a look at TUIs in UMLS or
                SNOMED-CT)
            description (str):
                Description of this concept.
            full_build (bool):
                If True the dictionary self.addl_info will also be populated,
                contains a lot of extra information about concepts, but can be
                very memory consuming. This is not necessary for normal
                functioning of MedCAT (Default Value `False`).
        """
        if not names:
            logger.warning("Passed an empty names dict for CUI '%s'. "
                           "This could be caused by the preprocessor "
                           "not picking up names due to the value of "
                           "'config.cdb_maker.min_letters_required' "
                           "(currently %d) being too small for this "
                           "particular name", cui,
                           self.config.cdb_maker.min_letters_required)
            return
        # Add CUI to the required dictionaries
        if cui not in self.cui2info:
            # Create placeholders
            cui_info = get_new_cui_info(
                cui=cui, preferred_name='', type_ids=type_ids)
            self.cui2info[cui] = cui_info
        else:
            cui_info = self.cui2info[cui]
            # If the CUI is already in update the type_ids
            cui_info['type_ids'].update(type_ids)

        # Add names to the required dictionaries
        self._add_concept_names(cui, names, name_status)

        if name_status == 'P' and not cui_info['preferred_name']:
            raw_names = [ini.raw_name for ini in names.values()]
            # TODO: which raw name?
            # previous implementation used the somewhat arbitrary last raw name
            rni = -1
            cui_info['preferred_name'] = raw_names[rni]

        # Add other fields if full_build
        if full_build:
            self._add_full_build(cui, names, ontologies, description, type_ids)
        self.is_dirty = True

    def reset_training(self) -> None:
        """Will remove all training efforts - in other words all embeddings
        that are learnt for concepts in the current CDB. Please note that this
        does not remove synonyms (names) that were potentially added during
        supervised/online learning.
        """
        for cui_info in self.cui2info.values():
            reset_cui_training(cui_info)
        for name_info in self.name2info.values():
            name_info['count_train'] = 0
        self._subnames.clear()
        # clear config entries as well
        self.config.meta.unsup_trained.clear()
        self.config.meta.sup_trained.clear()
        self.is_dirty = True

    def filter_by_cui(self, cuis_to_keep: Collection[str]) -> None:
        """Subset the core CDB fields (dictionaries/maps).

        Note that this will potenitally keep a bit more CUIs
        then in cuis_to_keep. It will first find all names that
        link to the cuis_to_keep and then find all CUIs that
        link to those names and keep all of them.

        This also will not remove any data from cdb.addl_info -
        as this field can contain data of unknown structure.

        Args:
            cuis_to_keep (Collection[str]):
                CUIs that will be kept, the rest will be removed
                (not completely, look above).

        Raises:
            Exception: If no snames and subsetting is not possible.
        """
        # First get all names/snames that should be kept based on this CUIs
        names_to_keep = set()
        snames_to_keep = set()
        for cui in cuis_to_keep:
            if cui not in self.cui2info:
                logger.warning(
                    "While filtering for CUIs asked to keep CUI '%s'"
                    "which is not a part of the existing CDB", cui)
                continue
            ci = self.cui2info[cui]
            names_to_keep.update(ci['names'])
            snames_to_keep.update(ci['subnames'])

        # Based on the names get also the indirect CUIs that have to be kept
        all_cuis_to_keep: set[str] = set()
        for name in names_to_keep:
            # NOTE: since this was based on the cui2info they
            #       should all have a name info
            ni = self.name2info[name]
            all_cuis_to_keep.update(ni['per_cui_status'].keys())

        new_cui2info: dict[str, CUIInfo] = {}
        new_name2info: dict[str, NameInfo] = {}

        # get kept
        for cui in all_cuis_to_keep:
            if cui not in self.cui2info:
                # NOTE: already warned above
                continue
            new_cui2info[cui] = self.cui2info[cui]

        for name in names_to_keep:
            # NOTE: should all be in name2info since got from cui2info
            new_name2info[name] = self.name2info[name]

        # set filtered dicts
        self.cui2info = new_cui2info
        self.name2info = new_name2info
        # redo all subnames
        self._reset_subnames()
        self.is_dirty = True

    def _remove_names(self, cui: str, names: Iterable[str]) -> None:
        """Remove names from an existing concept - effect is this name will
        never again be used to link to this concept. This will only remove the
        name from the linker (namely name2cuis and name2cuis2status), the name
        will still be present everywhere else. Why? Because it is bothersome
        to remove it from everywhere, but could also be useful to keep the
        removed names in e.g. cui2names.

        Args:
            cui (str):
                Concept ID or unique identifier in this database.
            names (Iterable[str]):
                Names to be removed (e.g list, set, or even a dict (in which
                case keys will be used)).
        """
        for name in names:
            if name in self.name2info:
                info = self.name2info[name]
                if cui in info['per_cui_status']:
                    del info['per_cui_status'][cui]
                if len(info['per_cui_status']) == 0:
                    del self.name2info[name]

            # Remove from per_cui_status
            if name in self.name2info:
                info = self.name2info[name]
                cuis2status = info['per_cui_status']
                if cui in cuis2status:
                    _ = cuis2status.pop(cui)
                if len(cuis2status) == 0:
                    # TODO: does this make sense?
                    del self.name2info[name]

            # Set to disamb always if name2cuis2status is now only one CUI
            if name in self.name2info:
                info = self.name2info[name]
                cuis2status = info['per_cui_status']
                if len(cuis2status) == 1:
                    for _cui in cuis2status:
                        if cuis2status[_cui] == 'A':
                            cuis2status[_cui] = 'N'
                        elif cuis2status[_cui] == 'P':
                            cuis2status[_cui] = 'PD'
        self.is_dirty = True
        self.has_changed_names = True

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CDB):
            return False
        # NOTE: Using config.model_dump since
        #       some parts of the config should not be considered.
        #       This refers to (mostly) the init args stored within there
        #       for various components
        return (self.config.model_dump() == other.config.model_dump() and
                self.cui2info == other.cui2info and
                self.name2info == other.name2info and
                self.type_id2info == other.type_id2info and
                self.token_counts == other.token_counts)

    def get_cui2count_train(self) -> dict[str, int]:
        return {
            cui: ct for cui, ci in self.cui2info.items()
            if (ct := ci['count_train'])
        }

    def get_name2count_train(self) -> dict[str, int]:
        return {
            cui: ct for cui, ni in self.name2info.items()
            if (ct := ni['count_train'])
        }

    def get_hash(self) -> str:
        hasher = Hasher()
        # only length for number of cuis/names/subnames
        hasher.update(len(self.cui2info))
        hasher.update(len(self.name2info))
        hasher.update(len(self._subnames))
        # the entirety of trained stuff
        hasher.update(self.get_cui2count_train())
        hasher.update(self.get_name2count_train())
        return hasher.hexdigest()
