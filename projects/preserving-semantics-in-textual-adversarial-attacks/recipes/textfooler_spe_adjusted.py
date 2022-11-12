"""
TFAdjusted + SPE Recipe
=============

Adapted from
https://github.com/QData/Reevaluating-NLP-Adversarial-Examples/blob/master/section_6_adjusted_attacks/recipes/textfooler_jin_2019_adjusted.py

"""
from textattack import Attack
from textattack.attack_recipes import AttackRecipe
from textattack.constraints.grammaticality import LanguageTool
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding

from spe import SemanticsPreservingEncoder


class SPETFAdjusted(AttackRecipe):
    """
    This is an attack recipe for the TFAdjusted attack using the SPE
    (Semantics Preserving Encoder) as semantic similarity metric.

    """

    @staticmethod
    def build(model, se_thresh=0.98, classifiers=None, concatenate=False):
        #
        # Swap words with their embedding nearest-neighbors.
        #
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        #
        # 50 nearest-neighbors with a cosine similarity of at least 0.5.
        # (The paper claims 0.7, but analysis of the code and some empirical
        # results show that it's definitely 0.5.)
        #
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # Minimum word embedding cosine similarity of 0.9.
        #
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.9))
        #
        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.7.
        #

        se_constraint = SemanticsPreservingEncoder(
            threshold=se_thresh,
            metric="cosine",
            compare_against_original=True,
            window_size=None,
            skip_text_shorter_than_window=False,
            classifiers=classifiers,
            concatenate=concatenate,
        )

        constraints.append(se_constraint)
        #
        # Do grammar checking
        #
        constraints.append(LanguageTool(0))

        #
        # Untargeted attack
        #
        goal_function = UntargetedClassification(model)

        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR()

        return Attack(goal_function, constraints, transformation, search_method)
