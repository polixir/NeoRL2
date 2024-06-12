import os
import sys
from setuptools import setup
from setuptools import find_packages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neorl2'))

assert sys.version_info.major == 3, \
    "This repo is designed to work with Python 3." \
    + "Please install it before proceeding."

setup(
    name='neorl2',
    author='Polixir Technologies Co., Ltd.',
    py_modules=['neorl2'],
    version="0.0.2",
    url="https://github.com/polixir/NeoRL2",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scipy',
        'torch',
        'onnxruntime',
        'matplotlib',
        'gymnasium[all]==0.29.1',
        'numpy==1.26.4',
    ],
    extras_require={
        'mujoco': ['mujoco-py']
    },

)
