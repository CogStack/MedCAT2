from typing import Iterator, Union, Optional, overload, cast
import logging

from spacy.tokens import Token as SpacyToken
from spacy.tokens import Span as SpacySpan
from spacy.tokens import Doc as SpacyDoc

from medcat2.tokenizing.tokens import (BaseToken, MutableToken,
                                       BaseEntity, MutableEntity,
                                       BaseDocument)


logger = logging.getLogger(__name__)


class Token:

    def __init__(self, delegate: SpacyToken) -> None:
        self._delegate = delegate
        # defaults
        self.norm = ''
        self.to_skip = False
        self.is_punctuation = self._delegate.is_punct

    @property
    def base(self) -> BaseToken:
        return cast(BaseToken, self)

    @property
    def text(self) -> str:
        return self._delegate.text

    @property
    def text_versions(self) -> list[str]:
        return [self.text, self.norm]

    # @property
    # def norm(self) -> bool:
    #     return self._delegate._.norm

    # @norm.setter
    # def norm(self, value: Union[bool, str]) -> None:
    #     self._delegate._.norm = value

    @property
    def lower(self) -> str:
        return self._delegate.lower_

    # @property
    # def to_skip(self) -> bool:
    #     return self._delegate._.to_skip

    # @to_skip.setter
    # def to_skip(self, new_val: bool) -> None:
    #     self._delegate._.to_skip = new_val

    @property
    def is_stop(self) -> bool:
        return self._delegate.is_stop

    @property
    def is_digit(self) -> bool:
        return self._delegate.is_digit

    # @property
    # def is_punct(self) -> bool:
    #     return self._delegate.is_punct

    # @is_punct.setter
    # def is_punct(self, new_val: bool) -> None:
    #     self._delegate.is_punct = new_val

    @property
    def is_upper(self) -> bool:
        return self._delegate.is_upper

    @property
    def tag(self) -> Optional[str]:
        return self._delegate.tag_

    @property
    def lemma(self) -> str:
        return self._delegate.lemma_

    @property
    def text_with_ws(self) -> str:
        return self._delegate.text_with_ws

    @property
    def index(self) -> int:
        return self._delegate.i

    def should_include(self) -> bool:
        return (not self.to_skip and not self.is_stop and
                not self.is_digit and not self.is_punctuation)


class Entity:

    def __init__(self, delegate: SpacySpan) -> None:
        self._delegate = delegate
        # defaults
        self.link_candidates: list[str] = []
        self.context_similarity: float = 0.0
        self.confidence: float = 0.0
        self.cui = ''
        self.id = -1  # TODO - what's the default?
        self.detected_name = ''

    @property
    def base(self) -> BaseEntity:
        return cast(BaseEntity, self)

    @property
    def text(self) -> str:
        return self._delegate.text

    @property
    def label(self) -> int:
        return self._delegate.label

    # @property
    # def link_candidates(self) -> Optional[list[str]]:
    #     return self._delegate._.link_candidates

    # @link_candidates.setter
    # def link_candidates(self, candidates: list[str]) -> None:
    #     self._delegate._.link_candidates = candidates

    # @property
    # def context_similarity(self) -> float:
    #     return self._delegate._.context_similarity

    # @context_similarity.setter
    # def context_similarity(self, val: float) -> None:
    #     self._delegate._.context_similarity = val

    # @property
    # def confidence(self) -> float:
    #     return self._delegate._.confidence

    # @confidence.setter
    # def confidence(self, val: float) -> None:
    #     self._delegate._.confidence = val

    # @property
    # def cui(self) -> str:
    #     return self._delegate._.cui

    # @cui.setter
    # def cui(self, value: str) -> None:
    #     self._delegate._.cui = value

    # @property
    # def id(self) -> int:
    #     return self._delegate._._id

    # @id.setter
    # def id(self, value: int) -> None:
    #     self._delegate._._id = value

    @property
    def start_index(self) -> int:
        return self._delegate[0].i

    @property
    def end_index(self) -> int:
        return self._delegate[-1].i

    # @property
    # def detected_name(self) -> str:
    #     return self._delegate._.detected_name

    # @detected_name.setter
    # def detected_name(self, name: str) -> None:
    #     self._delegate._.detected_name = name

    def __iter__(self) -> Iterator[MutableToken]:
        for tkn in self._delegate:
            yield Token(tkn)

    def __len__(self) -> int:
        return len(self._delegate)


class Document:

    def __init__(self, delegate: SpacyDoc) -> None:
        self._delegate = delegate
        self.all_ents: list[MutableEntity] = []

    @property
    def base(self) -> BaseDocument:
        return cast(BaseDocument, self)

    @property
    def final_ents(self) -> list[MutableEntity]:
        return [Entity(span) for span in self._delegate.ents]

    @final_ents.setter
    def final_ents(self, value: list[MutableEntity]) -> None:
        self._delegate.ents = [entity._delegate for entity in value
                               if isinstance(entity, Entity)]
        # if some of the mare not Entity (i.e not Spacy-based)
        if len(self._delegate.ents) != len(value):
            raise ValueError("All entities were expected to be Spacy-based. "
                             f"Got: {value}")

    @property
    def text(self) -> str:
        return self._delegate.text

    @overload
    def __getitem__(self, index: int) -> MutableToken:
        pass

    @overload
    def __getitem__(self, index: slice) -> MutableEntity:
        pass

    def __getitem__(self, index: Union[int, slice]
                    ) -> Union[MutableToken, MutableEntity]:
        delegated = self._delegate[index]
        if isinstance(delegated, SpacyToken):
            return Token(delegated)
        return Entity(delegated)

    def __iter__(self) -> Iterator[MutableToken]:
        for tkn in iter(self._delegate):
            yield Token(tkn)

    def isupper(self) -> bool:
        return self._delegate.text.isupper()
