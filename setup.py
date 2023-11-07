""" project setup """
from setuptools import setup, find_packages

VERSION="1.0.0rc1"

setup(
    name='TEG',
    version=VERSION,
    packages=[package for package in find_packages() if package.startswith("TEG")],
    zip_safe=True,
)
