from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import os
import sys

import neural_network

here = os.path.abspath(os.path.dirname(__file__))


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read("docs/README.txt")


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


setup(
    name="Neural Network",
    version=neural_network.__version__,
    url="https://github.com/Havilash/Neural-Network/",
    author="Gregory Reiter, Havilash Sivaratnam",
    tests_require=["pytest"],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "jupyter",
        "keras",
    ],
    cmdclass={"test": PyTest},
    description="own Neural Network",
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    test_suite="tests",
    classifiers=[
        "Programming Language :: Python",
    ],
    extras_require={
        "testing": ["pytest"],
    },
)
