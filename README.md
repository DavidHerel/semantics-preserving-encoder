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
