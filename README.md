## About
[Semantics Preserving Encoder](https://arxiv.org/abs/2211.04205) is a simple, fully supervised sentence embedding technique for textual adversarial attacks.

## Setup
You should be able to run this package with Python 3.6+. To use Semantics Preserving Encoder simply run pip with command:
``` sh
pip install spe
```

## Usage
This package is easy to use or integrate into any Python project as follows:
``` sh
from spe import spe

input_sentences = input_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"]

output_vectors = spe(input_sentences)
```
### Possible modifications
You can utilise the default classifiers specified in paper [Semantics Preserving Encoder](https://arxiv.org/abs/2211.04205) or extend/ replace with your own classifiers by placing them in the "classifiers" project folder. The script will auto detect these changes. *Note*: Currently, only fastText classifiers are supported.

You can also define your own vector dimension for the output vectors through an optional second parameter of 'spe' method. Otherwise, 10 is used as a default value.
``` sh
my_vector_dimensions = 20
output_vectors = spe(input_sentences, my_vector_dimensions)
```

## Citation

If you use Semantics Preserving Encoder for your research, please cite [Preserving Semantics in Textual Adversarial Attacks](https://arxiv.org/abs/2211.04205).

```bibtex
@misc{https://doi.org/10.48550/arxiv.2211.04205,
  doi = {10.48550/ARXIV.2211.04205},
  url = {https://arxiv.org/abs/2211.04205},
  author = {Herel, David and Cisneros, Hugo and Mikolov, Tomas},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Preserving Semantics in Textual Adversarial Attacks},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```
