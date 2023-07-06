"""Saves the results of a surface-level decontextualization to the format useful for upworkers.

Upworkers were asked to write questions about each snippet.
"""

import json
import random
from argparse import ArgumentParser

import pandas as pd

decontext_data_file = (
    "results/decontext/surface/oatext-davinci-003s1fsns4XEe-trn-qaspar-upwork-decontext-"
    "surface_1000-2000/val-qaspar-upwork-decontext-surface_1000-2000-t-0.7_topp-1.0_mgl-"
    "150/predictions.json"
)


def main():
    argp = ArgumentParser()
    argp.add_argument("--outfile", type=str, default=None)
    argp.add_argument("--seed", type=int, default=-1)
    args = argp.parse_args()

    with open("data/qaspar/fields/val.json") as f:
        val_fields = [json.loads(line) for line in f]

    with open(decontext_data_file) as f:
        decontext_data = [json.loads(line) for line in f]
        decontext_data = {d["idx"]: d for d in decontext_data}

    output = {"idx": [], "title": [], "question": [], "sentence": []}
    if args.seed >= 0:
        random.seed(args.seed)
        random.shuffle(val_fields)

    for sample in val_fields:
        sentence = sample["sentence"]
        if sentence.startswith("FLOAT SELECTED"):
            continue

        try:
            decontext_sent = decontext_data[sample["idx"]]["y_hat"].strip()
        except KeyError:
            continue

        output["idx"].append(sample["idx"])
        output["title"].append(sample["title"])
        output["question"].append(sample["question"])
        output["sentence"].append(decontext_sent)

        # if len(output["idx"]) == 100:
        #     break

    out_df = pd.DataFrame(output)
    if args.outfile is not None:
        out_df.to_csv(args.outfile, index=False)
    else:
        print(out_df.head())


if __name__ == "__main__":
    main()
