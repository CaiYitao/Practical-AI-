import setuptools
# import os
# os.environ['CC'] = 'gcc'
# os.environ['CXX'] = 'clang++'


if __name__ == "__main__":

    # export CXX= g++
    setuptools.setup(
        name="confgf",
        version="0.1.0",
        packages=setuptools.find_packages(include=["confgf"]),
        python_requires=">=3.5",
    )

import sys
sys.executable

import torch
print(torch.cuda.is_available())