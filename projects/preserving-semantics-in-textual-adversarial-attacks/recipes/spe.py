import numpy as np
import fasttext
from scipy import spatial
from textattack.constraints.semantics.sentence_encoders import SentenceEncoder

ALL_CLASSIFIER_NAMES = [
    "ag_news",
    "cyberbully",
    "sogou_news",
    "amazon_review_full",
    "dbpedia",
    "stack_overflow",
    "amazon_review_polarity",
    "emotion",
    "trip_advisor",
    "yahoo_answers",
    "clothing",
    "poem",
    "yelp_review_full",
    "sarcastic",
    "yelp_review_polarity",
    "imdb",
    "rotten",
    "snli",
    "paws",
    "trec",
    "cola",
    "rte",
    "mrpc",
    "sst2",
]

DEFAULT_CLS = [
    "stack_overflow",
    "emotion",
    "yelp_review_polarity",
    "snli",
    "cola",
    "rte",
    "sst2",
]


class SemanticsPreservingEncoder(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder."""

    def __init__(self, threshold=0.8, metric="angular", classifiers=DEFAULT_CLS, **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        selected_classifiers_names = classifiers
        if classifiers is None:
            selected_classifiers_names = DEFAULT_CLS
        assert type(selected_classifiers_names) == list
        unknown_classifiers = [
            c for c in selected_classifiers_names if c not in ALL_CLASSIFIER_NAMES
        ]
        if unknown_classifiers:
            raise ValueError(f"The classifiers {unknown_classifiers} are not known")
        # loading fasttext classifiers into the array from the folder

        self.classifiers = []
        for name in selected_classifiers_names:
            self.classifiers.append(
                fasttext.load_model("classifiers_fasttext/" + name + ".ftz")
            )

    # concatenate
    def spe_concatenate(self, original_text):
        concat_vector_arr = []

        # get all vectors from classifiers and put them in an array
        for i, model in enumerate(self.classifiers):
            sentence_vector = model.get_sentence_vector(original_text)
            concat_vector_arr.append(sentence_vector)

        final_average_vector = np.array(concat_vector_arr)

        # return averaged vector
        return final_average_vector.flatten()

    # average on 10 dimensions
    def spe_average(self, original_text):
        averaged_vector_arr = np.zeros(shape=(len(self.classifiers), 10))

        # get all vectors from classifiers and put them in an array
        for i, model in enumerate(self.classifiers):
            sentence_vector = model.get_sentence_vector(original_text)
            averaged_vector_arr[i] = sentence_vector

        final_average_vector = averaged_vector_arr.mean(axis=0)

        # return averaged vector
        return final_average_vector

    def spe(self, original_sentences):
        arr = []
        for i in original_sentences:
            arr.append(self.spe_concatenate(i))

        return arr

    def encode(self, sentences):
        return self.spe(sentences)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = None
