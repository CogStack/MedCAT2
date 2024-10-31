from medcat2.utils import iterutils

import unittest


class CallbackIterableTests(unittest.TestCase):
    EXP_LEN_LIST = 12
    EXP_LEN_ITER = 18
    ID_LEN = 'with length'
    ID_NO_LEN = 'no length'

    @classmethod
    def setUpClass(cls):
        cls._w_len = list(range(cls.EXP_LEN_LIST))

    def setUp(self):
        self._no_len = ('a' * nr for nr in range(self.EXP_LEN_ITER))
        self.last_id = 'N/A'
        self.last_count = -1

    def init_w_len(self):
        self.w_len = iterutils.callback_iterator(self.ID_LEN,
                                                 self._w_len, self.callback)

    def init_no_len(self):
        self.no_len = iterutils.callback_iterator(self.ID_NO_LEN,
                                                  self._no_len, self.callback)

    def callback(self, identifier: str, count: str):
        self.last_id = identifier
        self.last_count = count

    def consume(self, iterable):
        for _ in iterable:
            pass

    def test_gets_len_of_list_before_full_iter(self):
        self.init_w_len()
        for _ in self.w_len:
            # still need to iter over the first element
            break
        self.assertEqual(self.last_id, self.ID_LEN)
        self.assertEqual(self.last_count, self.EXP_LEN_LIST)

    def test_list_len_correct_after_iter(self):
        self.init_w_len()
        self.consume(self.w_len)
        self.assertEqual(self.last_id, self.ID_LEN)
        self.assertEqual(self.last_count, self.EXP_LEN_LIST)

    def test_no_len_of_gen_before_iter(self):
        self.init_no_len()
        self.assertNotEqual(self.last_id, self.ID_NO_LEN)
        self.assertNotEqual(self.last_count, self.EXP_LEN_ITER)

    def test_gets_len_of_gen_after_iter(self):
        self.init_no_len()
        self.consume(self.no_len)
        self.assertEqual(self.last_id, self.ID_NO_LEN)
        self.assertEqual(self.last_count, self.EXP_LEN_ITER)
