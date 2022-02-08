from setuptools import find_packages, setup


setup(
    name='emlopt',
    packages=find_packages(include=['emlopt']),
    version='0.1.0',
    description='Empirical Model Learning black box OPTimization with constrains',
    author='danver',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests'
)
