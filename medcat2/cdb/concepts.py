from typing import Optional, Any, TypedDict
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from medcat2.utils.defaults import StatusTypes as ST


class CUIInfo(TypedDict):
    cui: str  # NOTE: we _could_ get away without to save on memory
    preferred_name: str
    names: set[str]
    subnames: set[str]
    type_ids: set[str]
    # optional parts start here
    description: Optional[str]
    original_names: Optional[set[str]]
    tags: list[str]
    group: Optional[str]
    in_other_ontology: dict[str, Any]
    # stuff related to training starts here
    # TODO: separate supervised and unsupervised
    count_train: int
    context_vectors: Optional[dict[str, np.ndarray]]
    average_confidence: float


def get_new_cui_info(cui: str, preferred_name: str,
                     names: set[str] = set(), subnames: set[str] = set(),
                     type_ids: set[str] = set(),
                     description: Optional[str] = None,
                     original_names: Optional[set[str]] = None,
                     tags: list[str] = list(), group: Optional[str] = None,
                     in_other_ontology: dict[str, Any] = dict(),
                     count_train: int = 0,
                     context_vectors: Optional[dict[str, np.ndarray]] = None,
                     average_confidence: float = 0.0) -> CUIInfo:
    return {
        'cui': cui,
        'preferred_name': preferred_name,
        'names': names or names.copy(),
        'subnames': subnames or subnames.copy(),
        'type_ids': type_ids or type_ids.copy(),
        'description': description,
        'original_names': original_names,
        'tags': tags or tags.copy(),
        'group': group,
        'in_other_ontology': in_other_ontology or in_other_ontology.copy(),
        'count_train': count_train,
        'context_vectors': context_vectors,
        'average_confidence': average_confidence
    }


def reset_cui_training(cui_info: CUIInfo) -> None:
    cui_info['context_vectors'] = None
    cui_info['count_train'] = 0
    cui_info['average_confidence'] = 0


def get_defdict():
    return defaultdict(lambda: ST.AUTOMATIC)


class NameInfo(TypedDict):
    name: str  # NOTE: we _could_ get away without to save on memory
    cuis: set[str]
    per_cui_status: defaultdict[str, str]
    is_upper: bool
    # stuff related to training starts here
    count_train: int


def get_new_name_info(name: str, cuis: set[str] = set(),
                      per_cui_status: defaultdict[str, str] = get_defdict(),
                      is_upper: bool = False,
                      count_train: int = 0) -> NameInfo:
    return {
        'name': name,
        'cuis': cuis or cuis.copy(),
        'per_cui_status': per_cui_status or per_cui_status.copy(),
        'is_upper': is_upper,
        'count_train': count_train
    }


@dataclass
class TypeInfo:
    """Represents all the info regarding a type ID."""
    type_id: str  # NOTE: we _could_ get away without to save on memory
    name: str
    cuis: set[str] = field(default_factory=set)
