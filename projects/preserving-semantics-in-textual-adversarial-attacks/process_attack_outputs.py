import pathlib
import pickle as pkl

import pandas as pd

def get_type(st):
    if "SuccessfulAttackResult" in str(type(st)):
        return "Successful"
    elif "SkippedAttackResult" in str(type(st)):
        return "Skipped"
    else:
        return "Failed"

results = {}
for c in pathlib.Path("checkpoints/").glob("results*.pkl"):
    dic = pkl.load(open(c, "rb"))
    for key in dic:
        if not key in results:
            results[key] = {}
        results[key] = {**results[key], **dic[key]}

results_table = []
for dataset in results:
    for attack in results[dataset]:
        (time_taken, sentences) = results[dataset][attack]
        attack_name = attack.split(".")[-1].strip("'>")
        print(f"Processing dataset {dataset} with attack {attack_name}")

        output_fname = f"attack_results/output_{dataset}_{attack_name}.csv"
        text_comp_df = pd.DataFrame(
            [
                [st.original_text(), st.perturbed_text(), get_type(st)]
                for st in sentences
            ]
        )
        text_comp_df.columns = ["original", "perturbed", "type"]
        print(f"\tWriting to {output_fname}")
        text_comp_df.to_csv(
            output_fname, index=False
        )

        time_per_sent = time_taken / len(sentences)

        num_successful = sum(
            1 for st in sentences if "SuccessfulAttackResult" in str(type(st))
        )

        num_skipped = sum(
            1 for st in sentences if "SkippedAttackResult" in str(type(st))
        )

        num_failed = sum(1 for st in sentences if "FailedAttackResult" in str(type(st)))

        success_rate = num_successful / (num_failed + num_successful)

        percent_skipped = num_skipped / len(sentences)

        num_words_per_inp = sum(
            len(st.original_text().split()) for st in sentences
        ) / len(sentences)

        original_accuracy = (num_successful + num_failed) / len(sentences)

        attacked_accuracy = num_failed / len(sentences)

        modified_words = (
            sum(
                sum(
                    1
                    for (a, b) in zip(
                        st.perturbed_text().split(), st.original_text().split()
                    )
                    if a != b
                )
                for st in sentences
                if "SuccessfulAttackResult" in str(type(st))
            )
            / num_successful
        )

        results_table.append(
            [
                dataset,
                attack_name,
                num_successful,
                num_skipped,
                num_failed,
                success_rate,
                percent_skipped,
                num_words_per_inp,
                original_accuracy,
                attacked_accuracy,
                modified_words,
                time_per_sent,
            ]
        )

column_names = [
    "dataset",
    "attack",
    "successful_attacks",
    "skipped",
    "failed_attacks",
    "success_rate",
    "percent_skipped",
    "avg_words_per_input",
    "original_accuracy",
    "attacked_accuracy",
    "avg_num_modified_words_per_sent",
    "time_per_sent",
]

results_df = pd.DataFrame(results_table)
results_df.columns = column_names
results_df.to_csv("attack_results/attack_results_summary.csv", index=False)
print("Writing summary to attack_results/attack_results_summary.csv")
