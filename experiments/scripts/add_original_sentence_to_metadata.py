"""Adds the original snippet to metadata.

In order to calculate SARI scores, the original snippet before decontextualization is needed. However, this
snippet is not necesarily stored anywhere easily accessible in the experiment code. (For instance, the "x"
value is sometimes the prompt to the model which includes the snippet along with other information.)
Instead of trying to parse out the original snippet, which can be difficult, this script reads it from the
given `original_dataset` and adds it to the metadata.json file saved at the results path. If this file does
not exist, it is created. If it does exist, it's duplicated and then edited to add the original snippet.

If post_process_preds is True, there is additional post-processing performed on the predictions. Namely, the
brackets are removed and [REF0] is replaced with "the authors". This was used in earlier experiments to fix
a formatting mismatch between predictions and references due to a change in instructions.
"""

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path


def main():
    argp = ArgumentParser()
    argp.add_argument(
        "original_data",
        type=str,
        help="Probably should be the 'data/emnlp23/science/all_data.jsonl",
    )
    argp.add_argument("metadata", type=str)
    argp.add_argument("--post_process_preds", action="store_true")
    args = argp.parse_args()

    with open(args.original_data) as f:
        data = [json.loads(line) for line in f]
        try:
            data = {sample["idx"]: sample for sample in data}
        except KeyError:
            # for wiki
            data = {sample["example_id"]: sample for sample in data}
    # first preserve the metadata
    old_metadata = Path(args.metadata).with_stem("old_metadata")
    # print(Path(args.metadata).exists)
    if not Path(args.metadata).exists():
        # create from predictions.json file
        with open(Path(args.metadata).parent / "predictions.json") as f:
            metadata = [{"idx": json.loads(line.strip())["idx"]} for line in f]
    else:
        shutil.copy(args.metadata, old_metadata)

        with open(args.metadata) as f:
            metadata = [json.loads(line) for line in f]

    # add the x_no_parse field to the metadata
    for example in metadata:
        if "sentence" in data[example["idx"]]:
            example["x_no_parse"] = data[example["idx"]]["sentence"]
        else:
            # for wiki
            example["x_no_parse"] = data[example["idx"]]["original_sentence"]
    with open(args.metadata, "w") as f:
        for example in metadata:
            f.write(json.dumps(example) + "\n")

    if args.post_process_preds:
        with open(Path(args.metadata).parent / "predictions.json") as f:
            predictions = [json.loads(line) for line in f]

        for pred in predictions:
            pred["y_hat"] = pred["y_hat"].replace("<add>", "[")
            pred["y_hat"] = pred["y_hat"].replace("</add>", "]")

        with open(Path(args.metadata).parent / "predictions.json", "w") as f:
            for line in predictions:
                f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
