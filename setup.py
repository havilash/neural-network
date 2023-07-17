from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import os
import sys

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
requirements = read("requirements.txt").split('\n')


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
    name="neural_network",
    version='1.0.0',
    url="https://github.com/Havilash/Neural-Network/",
    author="Nicolas Thöni Castillo, Ensar Korkmaz, Gregory Reiter, Havilash Sivaratnam",
    license="MIT",
    keywords=["neural network", "MNIST", "classification"],
    tests_require=["pytest"],
    install_requires=requirements,
    extras_require={
        "dev": ["pytest"],
    },
    cmdclass={"test": PyTest},
    description="own Neural Network",
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    test_suite="tests",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
},
)
