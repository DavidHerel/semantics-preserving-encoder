import os

import fasttext
from sklearn.model_selection import train_test_split

file_names = [
    "snli",
    "cola",
    "rte",
    "sst2",
    "stack_overflow",
    "emotion",
    # "banking",
    # "clothing",
    # "cooking",
    # "cyberbully",
    # "movies",
    # "poem",
    # "sarcastic",
    # "trip_advisor",
    # "imdb",
    # "rotten",
    # 'paws',
    # 'trec',
    # 'mrpc'
]

file_names = ["stsb"]
vector_size = "_100"

for file_name in file_names:
    print("Training model on " + file_name + " dataset")
    # train the model with autotune parameters with size max 2MB (, autotuneModelSize="2M")
    # default training is 300 seconds. - 5 minutes, can be extended (, autotuneDuration=600)
    model = fasttext.train_supervised(
        input="datasets_fasttext/" + file_name + ".train",
        autotuneValidationFile="datasets_fasttext/" + file_name + ".valid",
        autotuneDuration=600,
        autotuneModelSize="20M",
        dim=100,
    )
    # save the model
    model.save_model("classifiers_fasttext/" + file_name + vector_size + ".ftz")
    print("Testing model precision and recall at one")
    print(model.test("datasets_fasttext/" + file_name + ".valid"))

    print()
    print(file_name + " hyper-parameters:")
    args_obj = model.f.getArgs()
    for hparam in dir(args_obj):
        if not hparam.startswith("__"):
            print(f"{hparam} -> {getattr(args_obj, hparam)}")
