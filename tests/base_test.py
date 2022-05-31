import sys, os
import logging
import unittest
from emlopt.utils import set_seed

def create_logger(name):
    test_logger = logging.getLogger(name)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    test_logger.addHandler(stream)
    test_logger.setLevel(logging.DEBUG)
    return test_logger

class BaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(BaseTest, cls).setUpClass()
        cls.test_logger = create_logger('emlopt-test')
        cls.test_logger.setLevel(logging.DEBUG)
        set_seed()

    def setUp(self):
        set_seed()
