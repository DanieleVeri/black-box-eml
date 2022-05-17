import sys, os
import logging

def create_logger(name):
    test_logger = logging.getLogger(name)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    test_logger.addHandler(stream)
    test_logger.setLevel(logging.DEBUG)
    return test_logger
