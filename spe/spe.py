import numpy as np
import fasttext
import os

def spe_sentence(input_sentence, classifiers, vector_dimension):
    """
    Processes the input sentence for each classifier into an averaged vector of given dimensions.
    This process is described in detail in paper https://arxiv.org/abs/2211.04205.

    :param input_sentence: sentence as a string
    :param classifiers: list of loaded classifiers
    :param vector_dimension: dimension of the averaged output vector as int
    :return: averaged output vector for given sentence
    """

    max_vector_size = min([len(i.get_sentence_vector("Test")) for i in classifiers])
    if vector_dimension > max_vector_size:
        raise Exception("ERROR: Vector dimension is higher than the model dimension, which is: " + str(max_vector_size))

    averaged_vector_arr = np.zeros(shape=(len(classifiers), vector_dimension))
    # get all vectors from classifiers and put them in an array
    for i, model in enumerate(classifiers):
        sentence_vector = model.get_sentence_vector(input_sentence)
        averaged_vector_arr[i] = sentence_vector[:vector_dimension]

    final_average_vector = averaged_vector_arr.mean(axis=0)
    return final_average_vector


def load_classifiers():
    """
    Loads all the fastText classifiers that are in classifiers_folder_path.

    :return: list of loaded fastText classifiers
    """
    classifiers_folder_path = os.path.join(os.path.dirname(__file__), 'classifiers')

    classifiers = []
    classifier_files = [f for f in os.listdir(classifiers_folder_path) if
                        os.path.isfile(os.path.join(classifiers_folder_path, f)) and f.endswith(".ftz")]

    if len(classifier_files) == 0:
        raise Exception("ERROR: There were not fasttext classifiers located in the classifiers folder")

    try:
        # silence the deprecation warnings as the package does not properly use the python 'warnings' package
        # see https://github.com/facebookresearch/fastText/issues/1056
        fasttext.FastText.eprint = lambda *args, **kwargs: None

        for name in classifier_files:
            classifiers.append(
                fasttext.load_model(os.path.join(classifiers_folder_path, name))
            )
    except:
        pass

    return classifiers


def spe(input_sentences, vector_dimension = 10):
    """
    Processes the list of input sentences into a corresponding array of vectors.

    :param input_sentences: list of input sentences as list of strings
    :param vector_dimension (optional): dimension of the output sentence vectors as int, default value is 10
    :return: array of vector sentences, one for each sentence
    """
    classifiers = load_classifiers()
    vector_sentences = []
    for sentence in input_sentences:
        vector_sentence = spe_sentence(sentence, classifiers, vector_dimension)
        vector_sentences.append(vector_sentence)
    return vector_sentences


if __name__ == '__main__':
    input_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"]
    output = spe(input_sentences, 10)
    print("SPE output:")
    print(output)
