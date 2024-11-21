from typing import Optional
import os

from medcat2.storage.serialisables import Serialisable, AbstractSerialisable
from medcat2.storage import serialisers

import unittest
import tempfile


class DummyClassWithDefValues(AbstractSerialisable):

    def __init__(self,
                 attr1: Optional[Serialisable] = None,
                 attr2: Optional[int] = None,
                 attr3: Optional[str] = None,
                 attr4: Optional[Serialisable] = None,
                 ):
        self.attr1 = attr1
        self.attr2 = attr2
        self.attr3 = attr3
        self.attr4 = attr4

    @classmethod
    def get_default(cls) -> 'DummyClassWithDefValues':
        return cls(
            attr1=AbstractSerialisable(),
            attr2=-1,
            attr3='some string',
            attr4=AbstractSerialisable(),
        )

    def __str__(self):
        return (f"<attr1: {self.attr1}, attr2: {self.attr2}"
                f"attr3: {self.attr3}, attr4: {self.attr4}>")

    def __repr__(self):
        return f"{{{self}}}"

    def __eq__(self, other: 'DummyClassWithDefValues') -> bool:
        if not isinstance(other, DummyClassWithDefValues):
            return False
        # NOTE: only the serialisable bits can eb checked since
        #       the rest of it doesn't get loaded
        return (self.attr1 == other.attr1 and self.attr4 == other.attr4)


class DummyClassWithMissingDefValues(DummyClassWithDefValues):

    def __init__(self, attr5: bool,
                 attr1: Optional[Serialisable] = None,
                 attr2: Optional[int] = None,
                 attr3: Optional[str] = None,
                 attr4: Optional[Serialisable] = None):
        super().__init__(attr1, attr2, attr3, attr4)
        self.attr5 = attr5

    @classmethod
    def get_default(cls, extra: bool = True
                    ) -> 'DummyClassWithMissingDefValues':
        return cls(
            attr5=extra,
            attr1=AbstractSerialisable(),
            attr2=-1,
            attr3='some string',
            attr4=AbstractSerialisable(),
        )


class SerialiserWorksTests(unittest.TestCase):
    SERIALISER_TYPE = serialisers.AvailableSerialisers.dill
    SERIALISABLE_INSTANCE = DummyClassWithDefValues.get_default()
    TARGET_CLASS = DummyClassWithDefValues

    def setUp(self):
        self._temp_folder = tempfile.TemporaryDirectory()
        self.temp_folder = self._temp_folder.name
        serialisers.serialise(self.SERIALISER_TYPE,
                              self.SERIALISABLE_INSTANCE,
                              self.temp_folder)

    def tearDown(self):
        self._temp_folder.cleanup()

    def assert_has_files_in_folder(self):
        self.assertTrue(os.listdir(self.temp_folder))

    def test_can_serialise_to_file(self):
        self.assert_has_files_in_folder()

    def test_can_deserialise_from_file(self):
        got = serialisers.deserialise(self.SERIALISER_TYPE, self.temp_folder)
        self.assertIsInstance(got, self.TARGET_CLASS)

    def test_deserialised_instance_same(self):
        got = serialisers.deserialise(self.SERIALISER_TYPE, self.temp_folder)
        self.assertEqual(got, self.SERIALISABLE_INSTANCE)


class SerialiserFailsTests(SerialiserWorksTests):
    NON_SERIALISABLE_INSTANCE = DummyClassWithMissingDefValues.get_default()
