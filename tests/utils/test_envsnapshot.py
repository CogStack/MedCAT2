from medcat2.utils import envsnapshot

import unittest


class EnvSnapshotTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = envsnapshot.get_environment_info()

    def test_gets_environment(self):
        self.assertIsInstance(self.env, envsnapshot.Environment)

    def test_has_deps(self):
        self.assertGreater(len(self.env.dependencies), 0)

    def test_has_trans_deps(self):
        self.assertGreater(len(self.env.transitive_deps), 0)

    def test_has_os(self):
        self.assertTrue(self.env.os)

    def test_has_arch(self):
        self.assertTrue(self.env.cpu_arcitecture)

    def test_has_py_version(self):
        self.assertTrue(self.env.python_version)

    def test_has_py3(self):
        self.assertIn("3.", self.env.python_version)
        self.assertTrue(self.env.python_version.startswith("3."))
