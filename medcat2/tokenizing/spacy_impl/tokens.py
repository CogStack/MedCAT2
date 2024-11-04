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

    @property
    def lower(self) -> str:
        return self._delegate.lower_

    @property
    def is_stop(self) -> bool:
        return self._delegate.is_stop

    @property
    def is_digit(self) -> bool:
        return self._delegate.is_digit

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

    @property
    def start_index(self) -> int:
        return self._delegate[0].i

    @property
    def end_index(self) -> int:
        return self._delegate[-1].i

    @property
    def start_char_index(self) -> int:
        return self._delegate.start_char

    @property
    def end_char_index(self) -> int:
        return self._delegate.end_char

    def __iter__(self) -> Iterator[MutableToken]:
        for tkn in self._delegate:
            yield Token(tkn)

    def __len__(self) -> int:
        return len(self._delegate)


class Document:

    def __init__(self, delegate: SpacyDoc) -> None:
        self._delegate = delegate
        self.all_ents: list[MutableEntity] = []
        self.final_ents: list[MutableEntity] = []

    @property
    def base(self) -> BaseDocument:
        return cast(BaseDocument, self)

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
