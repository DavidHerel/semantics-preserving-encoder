import setuptools

setuptools.setup(
    name="spe",
    version="0.0.1",
    author="David Herel, Hugo Cisneros, Daniela Hradilova, Tomas Mikolov",
    description="Sentence embedding technique for textual adversarial attacks",
    packages=find_packages(include=['spe'], exclude=['projects', ])
    install_requires=['<numpy>', '<fasttext>'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English",
    ],

)