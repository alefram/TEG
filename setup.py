from setuptools import setup

setup(
    name='RobotEnv',
    version='0.0.1',
    description='Environments and tools for develop smart controllers',
    author='Alexis Fraudita',
    license='Apache License 2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache  Software License'
    ],
    install_requires=['gym', 'mujoco_py'],
    python_requires='>=3.7',
    py_modules=[]
)
