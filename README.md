## Semantics Preserving Encoder
[Semantics Preserving Encoder](https://arxiv.org/abs/2211.04205) is a simple, fully supervised sentence embedding technique for textual adversarial attacks.

## How to use

``` sh
from spe import spe

input_sentences = input_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"]

output_vectors = spe(input_sentences)

```

## Citing SPE

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
```

###
