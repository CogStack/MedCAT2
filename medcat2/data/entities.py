from typing import TypedDict, Optional


class MetaAnnotation(TypedDict):
    value: str
    confidence: float
    name: str


class Entity(TypedDict):
    pretty_name: str
    cui: str
    type_ids: list[str]
    # types: List[str]# TODO: add back
    source_value: str
    detected_name: str
    acc: float
    context_similarity: float
    start: int
    end: int
    # icd10: List[str]# TODO: add back
    # ontologies: List[str]# TODO: add back
    # snomed: List[str]# TODO: add back
    id: int
    meta_anns: dict[str, MetaAnnotation]
    # optionals:
    context_left: list[str]
    context_center: list[str]
    context_right: list[str]


class Entities(TypedDict):
    # We use int as the key type, but a generic dictionary with int keys
    # is defined because the keys in entities are dynamic and can change.
    entities: dict[int, Entity]
    tokens: list[str]  # TODO - do we need this
    text: Optional[str]


class OnlyCUIEntities(TypedDict):
    entities: dict[int, str]
    tokens: list[str]  # TODO - do we need this
    text: Optional[str]
