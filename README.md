# Semantics Preserving Encoder

This repository contains the source code for reproducing the experiments of the
paper: "Preserving Semantics in Textual Adversarial Attacks" by David Herel,
Hugo Cisneros and Tomas Mikolov.

## Reproducing experiments

For a task `$TASK` and an attack `$ATTACK`, you can run the corresponding
experiment with:
``` sh
python run_attacks.py --task $TASK --attack $ATTACK
```

A list of available tasks and attacks can be found by running

``` sh
python run_attacks.py --help
```

To run all experiments at once and write the output to log files do:

``` sh
for attack in  TextFooler TFAdjusted TextFoolerBERT TFAdjusted USE SPE  SPEAdjusted
 Genetic CLARE SPECLARE
do
    for task in hate offensive imdb yelp rotten
    do
        python run_attacks.py --task $TASK --attack $ATTACK
    done
done
```
