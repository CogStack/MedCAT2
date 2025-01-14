from typing import Protocol, Optional, Iterator, overload, Union, Any, Type


class BaseToken(Protocol):

    @property
    def text(self) -> str:
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

    @property
    def is_stop(self) -> bool:
        pass

    @property
    def char_index(self) -> int:
        pass

    @property
    def index(self) -> int:
        pass

    @property
    def text_with_ws(self) -> str:
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

    @property
    def norm(self) -> str:
        pass

    @norm.setter
    def norm(self, value: str) -> None:
        pass


class BaseEntity(Protocol):

    @property
    def start_index(self) -> int:
        pass

    @property
    def end_index(self) -> int:
        pass

    @property
    def start_char_index(self) -> int:
        pass

    @property
    def end_char_index(self) -> int:
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

    def set_addon_data(self, path: str, val: Any) -> None:
        """Used to add arbitray data to the entity.

        This is generally used by addons to keep track of their data.

        NB! The path used needs to be registered using the
        `register_addon_path` class method.

        Args:
            path (str): The data ID / path.
            val (Any): The value to be added.
        """
        pass

    def get_addon_data(self, path: str) -> Any:
        """Get data added to the entity.

        See `add_data` for details.

        Args:
            path (str): The data ID / path.

        Returns:
            Any: The stored value.
        """
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

    @classmethod
    def register_addon_path(cls, path: str, def_val: Any = None,
                            force: bool = True) -> None:
        """Register a custom/arbitrary data path.

        This can be used to store arbitrary data along with the entity for
        use in an addon (e.g MetaCAT).

        PS: If using this, it is important to use paths namespaced to the
        component you're using in order to avoid conflicts.

        Args:
            path (str): The path to be used. Should be prefixed by component
                name (e.g `meta_cat_id` for an ID tied to the `meta_cat` addon)
            def_val (Any): Default value. Defaults to `None`.
            force (bool): Whether to forcefully add the value.
                Defaults to True.
        """
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

    @property
    def all_ents(self) -> list[MutableEntity]:
        pass

    def __iter__(self) -> Iterator[MutableToken]:
        pass

    @overload
    def __getitem__(self, index: int) -> MutableToken:
        pass

    @overload
    def __getitem__(self, index: slice) -> MutableEntity:
        pass

    def get_tokens(self, start_index: int, end_index: int
                   ) -> Union[MutableEntity, list[MutableToken]]:
        pass

    def set_addon_data(self, path: str, val: Any) -> None:
        """Used to add arbitray data to the entity.

        This is generally used by addons to keep track of their data.

        NB! The path used needs to be registered using the
        `register_addon_path` class method.

        Args:
            path (str): The data ID / path.
            val (Any): The value to be added.
        """
        pass

    def get_addon_data(self, path: str) -> Any:
        """Get data added to the entity.

        See `add_data` for details.

        Args:
            path (str): The data ID / path.

        Returns:
            Any: The stored value.
        """
        pass

    @classmethod
    def register_addon_path(cls, path: str, def_val: Any = None,
                            force: bool = True) -> None:
        """Register a custom/arbitrary data path.

        This can be used to store arbitrary data along with the entity for
        use in an addon (e.g MetaCAT).

        PS: If using this, it is important to use paths namespaced to the
        component you're using in order to avoid conflicts.

        Args:
            path (str): The path to be used. Should be prefixed by component
                name (e.g `meta_cat_id` for an ID tied to the `meta_cat` addon)
            def_val (Any): Default value. Defaults to `None`.
            force (bool): Whether to forcefully add the value.
                Defaults to True.
        """
        pass


class UnregisteredDataPathException(ValueError):

    def __init__(self, cls: Type, path: str):
        super().__init__(
            f"Unregistered path {path} for class: {cls}")
        self.cls = cls
        self.path = path
