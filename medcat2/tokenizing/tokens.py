from typing import Protocol, Optional, Iterator, overload


class BaseToken(Protocol):

    @property
    def raw_text(self) -> str:
        pass

    @property
    def lower(self) -> str:
        pass

    @property
    def text_versions(self) -> list[str]:
        pass

    @property
    def is_upper(self) -> bool:
        pass


class MutableToken(Protocol):

    @property
    def base(self) -> BaseToken:
        pass

    @property
    def is_punctuation(self) -> bool:
        pass

    @is_punctuation.setter
    def is_punctuation(self, val: bool) -> None:
        pass

    @property
    def to_skip(self) -> bool:
        pass

    @to_skip.setter
    def to_skip(self, new_val: bool) -> None:
        pass

    @property
    def lemma(self) -> str:
        pass

    @property
    def tag(self) -> Optional[str]:
        pass

    def should_include(self) -> bool:
        pass


class BaseEntity(Protocol):

    @property
    def start_index(self) -> int:
        pass

    @property
    def end_index(self) -> int:
        pass

    @property
    def label(self) -> int:
        pass

    @property
    def text(self) -> str:
        pass

    def __iter__(self) -> Iterator[BaseToken]:
        pass

    def __len__(self) -> int:
        pass


class MutableEntity(Protocol):

    @property
    def base(self) -> BaseEntity:
        pass

    @property
    def detected_name(self) -> str:
        pass

    @detected_name.setter
    def detected_name(self, name: str) -> None:
        pass

    @property
    def link_candidates(self) -> list[str]:
        pass

    @link_candidates.setter
    def link_candidates(self, candidates: list[str]) -> None:
        pass

    @property
    def context_similarity(self) -> float:
        pass

    @context_similarity.setter
    def context_similarity(self, val: float) -> None:
        pass

    @property
    def confidence(self) -> float:
        pass

    @confidence.setter
    def confidence(self, val: float) -> None:
        pass

    @property
    def cui(self) -> str:
        pass

    @cui.setter
    def cui(self, value: str) -> None:
        pass

    @property
    def id(self) -> int:
        pass

    @id.setter
    def id(self, value: int) -> None:
        pass

    def __iter__(self) -> Iterator[MutableToken]:
        pass

    def __len__(self) -> int:
        pass


class BaseDocument(Protocol):

    @property
    def text(self) -> str:
        pass

    @overload
    def __getitem__(self, index: int) -> BaseToken:
        pass

    @overload
    def __getitem__(self, index: slice) -> BaseEntity:
        pass

    def __iter__(self) -> Iterator[BaseToken]:
        pass

    def isupper(self) -> bool:
        pass


class MutableDocument(Protocol):

    @property
    def base(self) -> BaseDocument:
        pass

    @property
    def final_ents(self) -> list[MutableEntity]:
        pass

    def __iter__(self) -> Iterator[MutableToken]:
        pass
