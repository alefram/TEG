""" project setup """
import pathlib
from setuptools import setup, find_packages

CWD = pathlib.Path(__file__).absolute().parent

VERSION="0.0.1"

def get_version():
    """Gets TEG version."""
    path = CWD / "TEG" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")

setup(
    name='TEG',
    version=VERSION,
    packages=[package for package in find_packages() if package.startswith("TEG")],
    zip_safe=True,
)
