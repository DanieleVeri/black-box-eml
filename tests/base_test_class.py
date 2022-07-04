import sys, os
import logging
import unittest
from emlopt.utils import set_seed

def create_logger(name):
    test_logger = logging.getLogger(name)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    if not test_logger.handlers:
        test_logger.addHandler(stream)
    test_logger.setLevel(logging.DEBUG)
    test_logger.propagate = False
    return test_logger

class BaseTest(unittest.TestCase):
    loglevel = None

    @classmethod
    def setUpClass(cls):
        super(BaseTest, cls).setUpClass()
        cls.test_logger = create_logger(cls.__name__)
        if cls.loglevel is not None:
            cls.test_logger.setLevel(cls.loglevel)
        set_seed()

    @classmethod
    def tearDownClass(cls):
        cls.test_logger.handlers.clear()
        logging._handlers.clear()

    def setUp(self):
        set_seed()
