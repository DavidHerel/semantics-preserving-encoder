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
for attack in TextFooler SPE TextFoolerBERT TFAdjusted TFAdjustedUSE SPEAdjusted Genetic CLARE SPECLARE
do
    for task in hate offensive imdb yelp rotten
    do
        python run_attacks.py --task $TASK --attack $ATTACK
    done
done
```

## Human evaluation
The results of the human evaluation can be seen in the `attack_results` folder. The file `attack_results_summary` contains the result of the human evaluation.

## Classifiers modifiability
Classifiers selected for our SPE method can be modified according to the user needs. If the user wants to have a better performance in e.g. tweet domain he can add more fasttext or other classifiers to the `classifiers_fasttext` folder, which are trained on a suitable task. This will result in even bettter performance.
