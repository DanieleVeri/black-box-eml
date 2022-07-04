import unittest
import logging
import sys
sys.path.append('./tests')

def main(args):

    LOGLEVEL = logging.INFO

    def test_class(test_cls):
        test_cls.loglevel = LOGLEVEL
        suite = unittest.TestLoader().loadTestsFromTestCase(test_cls)
        results = unittest.TextTestRunner(verbosity=2).run(suite)
        if len(results.failures) > 0 or len(results.errors) > 0:
            exit(1)

    from bound_propagation_test import BoundPropagationTest
    test_class(BoundPropagationTest)

    from convergence_test import SearchConvergenceTest
    test_class(SearchConvergenceTest)

    from datapoints_initialization_test import InitTest
    test_class(InitTest)

    from dist_methods_test import DistMethodsTest
    test_class(DistMethodsTest)

    from emllib_backend_test import EMLBackendTest
    test_class(EMLBackendTest)

    from solver_method_test import SolverMethodTest
    test_class(SolverMethodTest)

    from surrogate_method_test import SurrogateMethodTest
    test_class(SurrogateMethodTest)

    exit(0)

if __name__ == '__main__':
    main(sys.argv)
