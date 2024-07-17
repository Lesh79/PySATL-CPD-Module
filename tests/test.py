import unittest


class Test(unittest.TestCase):
    def testABC(self):
        a = 1
        self.assertEqual(a, 1, "Messagee")
