import numpy as np
import fasttext
from scipy import spatial
import os

def spe(input_sentences):
    my_path = "classifiers"
    classifiers = []
    classifier_files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path, f)) and f.endswith(".ftz")]
    for name in classifier_files:
        classifiers.append(
            fasttext.load_model(my_path+ "/" + name)
        )

    averaged_vector_arr = np.zeros(shape=(len(classifiers), 10))

    # get all vectors from classifiers and put them in an array
    for i, model in enumerate(classifiers):
        sentence_vector = model.get_sentence_vector(input_sentences)
        averaged_vector_arr[i] = sentence_vector

    final_average_vector = averaged_vector_arr.mean(axis=0)

    return final_average_vector


if __name__ == '__main__':
    input_sentences = "bla bla"
    output = spe(input_sentences)
    print("SPE output:")
    print(output)