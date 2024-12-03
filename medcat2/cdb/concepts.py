from typing import Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from medcat2.utils.defaults import StatusTypes as ST
from medcat2.storage.serialisables import SerialisingStrategy


@dataclass
class CUIInfo:
    cui: str  # NOTE: we _could_ get away without to save on memory
    preferred_name: str
    names: set[str] = field(default_factory=set)
    subnames: set[str] = field(default_factory=set)
    type_ids: set[str] = field(default_factory=set)
    # optional parts start here
    description: Optional[str] = None
    original_names: Optional[set[str]] = None
    tags: list[str] = field(default_factory=list)
    group: Optional[str] = None
    in_other_ontology: dict[str, Any] = field(default_factory=dict)
    # stuff related to training starts here
    count_train: int = 0  # TODO: separate supervised and unsupervised
    context_vectors: Optional[dict[str, np.ndarray]] = None
    average_confidence: float = 0.0

    def reset_training(self) -> None:
        self.context_vectors = None
        self.count_train = 0
        self.average_confidence = 0

    def __eq__(self, other) -> bool:
        if not isinstance(other, CUIInfo):
            return False
        for ann_key in self.__annotations__:
            v1, v2 = getattr(self, ann_key), getattr(other, ann_key)
            if ann_key != 'context_vectors':
                if v1 != v2:
                    return False
                continue
            if v1 is None and v2 is None:
                continue
            if v1.keys() != v2.keys():
                return False
            for k in v1:
                sv1, sv2 = v1[k], v2[k]
                if not np.all(sv1 == sv2):
                    return False
        return True

    # Serialisable

    def get_strategy(self) -> SerialisingStrategy:
        # NOTE: has no serialisable
        return SerialisingStrategy.DICT_ONLY

    @classmethod
    def get_init_attrs(cls) -> list[str]:
        return list(__annotations__.keys())

    @classmethod
    def ignore_attrs(cls) -> list[str]:
        return []


@dataclass
class NameInfo:
    name: str  # NOTE: we _could_ get away without to save on memory
    cuis: set[str]  # = field(default_factory=set)
    per_cui_status: dict[str, str] = field(
        default_factory=lambda: defaultdict(lambda: ST.AUTOMATIC))
    is_upper: bool = False
    # stuff related to training starts here
    count_train: int = 0

    # Serialisable

    def get_strategy(self) -> SerialisingStrategy:
        # NOTE: has no serialisable
        return SerialisingStrategy.DICT_ONLY

    @classmethod
    def get_init_attrs(cls) -> list[str]:
        return list(__annotations__.keys())

    @classmethod
    def ignore_attrs(cls) -> list[str]:
        return []


@dataclass
class TypeInfo:
    type_id: str  # NOTE: we _could_ get away without to save on memory
    name: str
    cuis: set[str] = field(default_factory=set)
