from setuptools import setup, find_packages
import sys

with open('README.md', encoding="utf8") as f:
    # strip the header and badges etc
    readme = f.read().split('--------------------')[-1]

setup(
    name="spe-encoder",
    version="0.0.2",
    author="David Herel, Hugo Cisneros, Daniela Hradilova, Tomas Mikolov",
    description="Sentence embedding technique for textual adversarial attacks",
    packages=find_packages(include=['spe'], exclude=['projects', ]),
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'fasttext==0.9.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English",
    ],

)
