import argparse
import datetime
import pickle as pkl
import time
from enum import Enum

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from textattack import AttackArgs, Attacker
from textattack.attack_recipes.clare_li_2020 import CLARE2020
from textattack.attack_recipes.genetic_algorithm_alzantot_2018 import (
    GeneticAlgorithmAlzantot2018,
)
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from recipes.spe_clare_attack_recipe import SPECLARE
from recipes.spe_textfooler_attack_recipe import SPETextFooler
from recipes.textfooler_bert import TextFoolerJin2019BERT
from recipes.textfooler_jin_2019_adjusted import (
    TextFoolerJin2019Adjusted,
    TextFoolerJin2019AdjustedUSE,
)
from recipes.textfooler_spe_adjusted import SPETFAdjusted


class Attack(Enum):
    """All the possible attacks to run experiments with. This enum is used to
    ensure we don't select an impossible option from the CLI.

    """

    TextFooler = "TextFooler"
    TFAdjusted = "TFAdjusted"
    TextFoolerBERT = "TextFoolerBERT"
    TFAdjustedUSE = "TFAdjustedUSE"
    Genetic = "Genetic"
    SPETextFooler = "SPE"
    SPEAdjusted = "SPEAdjusted"
    CLARE = "CLARE"
    SPECLARE = "SPECLARE"

    def __str__(self):
        return self.value


class Task(Enum):
    """All the possible datasets to run experiments with. This enum is used to
    ensure we don't select an impossible option from the CLI.

    """

    Hate = "hate"
    Offensive = "offensive"
    IMDB = "imdb"
    YELP = "yelp"
    ROTTEN_TOMATOES = "rotten"

    def __str__(self):
        return self.value


# A dict with the options used to load the model weights from HF for each target
# dataset.
tasks_to_models = {
    "hate": "cardiffnlp/twitter-roberta-base-hate",
    "offensive": "cardiffnlp/twitter-roberta-base-offensive",
    "imdb": "lvwerra/distilbert-imdb",
    "yelp": "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "rotten": "RJZauner/distilbert_rotten_tomatoes_sentiment_classifier",
}

# A dict with the options used to load the datasets from HF.
tasks_to_dataset = {
    "hate": {"dataset": "tweet_eval", "subset": "hate", "split": "train"},
    "offensive": {"dataset": "tweet_eval", "subset": "offensive", "split": "train"},
    "imdb": {"dataset": "imdb", "subset": None, "split": "train"},
    "yelp": {"dataset": "yelp_polarity", "subset": None, "split": "train"},
    "rotten": {"dataset": "rotten_tomatoes", "subset": None, "split": "train"},
}


# A mapping from the attack chosen in CLI to actual attack recipe class to use.
all_attack_classes = {
    Attack.TextFooler: TextFoolerJin2019,
    Attack.TFAdjusted: TextFoolerJin2019Adjusted,
    Attack.TextFoolerBERT: TextFoolerJin2019BERT,
    Attack.TFAdjustedUSE: TextFoolerJin2019AdjustedUSE,
    Attack.SPETextFooler: SPETextFooler,
    Attack.SPEAdjusted: SPETFAdjusted,
    Attack.Genetic: GeneticAlgorithmAlzantot2018,
    Attack.CLARE: CLARE2020,
    Attack.SPECLARE: SPECLARE,
}

if __name__ == "__main__":
    all_results = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=Attack, choices=list(Attack))
    parser.add_argument("--task", type=Task, choices=list(Task))
    args = parser.parse_args()

    # CLI arguments
    task = str(args.task)
    all_results[task] = {}
    attack_class = all_attack_classes[args.attack]
    print(f"Running task {task} with attack {args.attack}")

    # Load the model to attack
    model_str = tasks_to_models[task]

    tokenizer = AutoTokenizer.from_pretrained(model_str)

    model = AutoModelForSequenceClassification.from_pretrained(model_str)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # Import the dataset
    dataset_meta = tasks_to_dataset[task]
    hg_dataset = load_dataset(
        dataset_meta["dataset"], dataset_meta["subset"], dataset_meta["split"]
    )
    if isinstance(hg_dataset, DatasetDict) or isinstance(
        hg_dataset, IterableDatasetDict
    ):
        subset = hg_dataset["train"]
    else:
        subset = hg_dataset

    dataset = HuggingFaceDataset(
        subset,
        dataset_columns=(["text"], "label"),
    )

    # Create the attack
    attack = attack_class.build(model_wrapper)
    attack_args = AttackArgs(
        num_examples=1000,
        shuffle=False,
    )
    attacker = Attacker(attack, dataset, attack_args)
    start_time = time.time()
    attack_results = attacker.attack_dataset()
    end_time = time.time()

    # Store attack results for export
    all_results[task][str(attack_class)] = (
        (end_time - start_time),
        attack_results,
    )

    # Export results
    current_datetime = datetime.datetime.now().isoformat()
    ck_name = f"checkpoints/results_{task}_{args.attack}_{current_datetime}.pkl"
    pkl.dump(all_results, open(ck_name, "wb"))

    print(f"Done running task {task} with attack {args.attack}")
