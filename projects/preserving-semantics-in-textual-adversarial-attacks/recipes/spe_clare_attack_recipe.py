"""
SPE + CLARE Recipe
=============

(Semantics Preserving Encoder + Contextualized Perturbation for Textual
Adversarial Attack)
"""

import transformers
from textattack import Attack
from textattack.attack_recipes import AttackRecipe
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedySearch
from textattack.transformations import (
    CompositeTransformation,
    WordInsertionMaskedLM,
    WordMergeMaskedLM,
    WordSwapMaskedLM,
)

from spe import SemanticsPreservingEncoder


class SPECLARE(AttackRecipe):
    """This class is an attack recipe adapted from:

    Li, Zhang, Peng, Chen, Brockett, Sun, Dolan. "Contextualized Perturbation
    for Textual Adversarial Attack" (Li et al., 2020)
    https://arxiv.org/abs/2009.07502

    This method uses greedy search with replace, merge, and insertion
    transformations that leverage a pretrained language model.

    Instead of the USE similarity constraint, we use the SPE similarity
    constraint.

    """

    @staticmethod
    def build(model_wrapper, classifiers=None, cosine_thresh=0.95):
        shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained(
            "distilroberta-base"
        )
        shared_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "distilroberta-base"
        )
        transformation = CompositeTransformation(
            [
                WordSwapMaskedLM(
                    method="bae",
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=5e-4,
                ),
                WordInsertionMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=0.0,
                ),
                WordMergeMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=5e-3,
                ),
            ]
        )

        #
        # Don't modify the same word twice or stopwords.
        #
        constraints = [RepeatModification(), StopwordModification()]

        # "A common choice of sim(·,·) is to encode sentences using neural
        # networks, and calculate their cosine similarity in the embedding space
        # (Jin et al., 2020)." The original implementation uses similarity of
        # 0.7.
        use_constraint = SemanticsPreservingEncoder(
            threshold=cosine_thresh,
            metric="cosine",
            compare_against_original=True,
            window_size=None,
            skip_text_shorter_than_window=False,
            classifiers=classifiers,
        )
        constraints.append(use_constraint)

        # Goal is untargeted classification. "The score is then the negative
        # probability of predicting the gold label from f, using [x_{adv}] as
        # the input"
        goal_function = UntargetedClassification(model_wrapper)

        # "To achieve this, we iteratively apply the actions, and first select
        #  those minimizing the probability of outputting the gold label y from
        #  f."
        #
        # "Only one of the three actions can be applied at each position, and we
        # select the one with the highest score."
        #
        # "Actions are iteratively applied to the input, until an adversarial
        # example is found or a limit of actions T is reached. Each step selects
        # the highest-scoring action from the remaining ones."
        #
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
