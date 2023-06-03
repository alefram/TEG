""" project setup """
from setuptools import setup, find_packages

VERSION="0.0.1"

setup(
    name='TEG',
    version=VERSION,
    description='Reinforcement Learning Environments for train robot arms agents',
    author='Alexis Fraudita',
    author_email='cuatroalejandro@gmail.com',
    license='Apache License 2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache  Software License'
    ],
    install_requires=[
        "numpy >=1.23.5",
        "mujoco >=2.3.1.post1",
        "gymnasium>=0.27.1"
    ],
    packages=[package for package in find_packages() if package.startswith("TEG")],
    package_data={
        "TEG": [
            "*.stl",
            "*.urdf",
            "*.xml"
        ]
    },
    python_requires='>=3.7',
    zip_safe=False,
    setup_requires=['pytest-runner', 'black'],
    tests_require=['pytest'],
)
