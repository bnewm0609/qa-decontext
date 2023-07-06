"""This script creates derived datasets by splitting and transforming all_data.jsonl files.

The script is run like:
`python propagate_data_changes.py data/path/to/all_data.jsonl`

The script searches for data/path/to/.conf.yaml, which defines how to generate the derived datasets
from all_data.json. The .conf.yaml file starts with the following keys:

```yaml
version: <verion_number>
seed: <seed>
name: <dataset_name>
```

`version` defines the version. This is automatically incremented every time the script is run. If the `--breaking`
    flag is used, the major version is updated, otherwise only the minor version is updated.
The `seed` is used for generating splits.
The `name` is used in the git commit message. After running, the script automatically creates a commit with the
    newest version of the dataset. This can be disabled by using the `--disable_git` flag.

After this, there is a `dirs` key which describes how to derive datasets. There are two types of derived datasets:
splits and transformations.

Splits
------
There is only one split. It defines how to divide up all_data.jsonl into train/val/test sets. It looks like:
```yaml
- name: <name>
  splits:
    train: <num_train>
    val: <num_val>
    test: <num_test>
```
This creates a directory called `<name>` and puts a `train.jsonl`, `val.jsonl`, and `test.jsonl` file in this
directory with <num_train>, <num_val>, and <num_test> lines from `all_data.jsonl` in each.

Transformations
--------------
There are multiple, chained transformations. They look like:
```yaml
- name: <name>
  parent: <parent_name>
  fn: <transformation_fn_name>
  args:
    <arg1>: <arg1_val>
```
This creates a directory called <name> and reads all of the files from <parent_name>. It applies the function
associated with <transformation_fn_name> to each sample (along with any args) and saves them in new files with
the same name in then new <name> directory. (E.g. parent_name/train.jsonl -> name/train.jsonl).

The script creates a DAG to ensure that all of the parents have been created before their children. Right now
this DAG is a tree, but in the future, it would be good to add support for multiple parents.
(Right now that's implemented in a janky way where some transformation functions just open other files.)
"""

import glob
import json
import random
import shlex
import subprocess
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from transformers import GPT2Tokenizer


def generate_splits(
    all_data_file: Path, out_files_with_splits: Dict[Path, float], seed: int
) -> None:
    """
    Args:
        all_data_file: the path to the file containing all of the data we use to generate splits
            (e.g. train,val,test).
        out_files_with_splits: ordered map from the paths to save the split examples to the number
            or proportion of the example devoted to that split
        seed: used to seed the rng for shuffling the data. Doesn't shuffle if seed is < 0

    Raises:
        AssertionError if the not all of the data in all_data_file is used.
    """

    # loads in all of the data
    data: List[dict[str, str]] = []
    with open(all_data_file) as f:
        data = [json.loads(line.strip()) for line in f.readlines()]

    # sets the seed to deterministically shuffle the data
    if seed >= 0:
        random.seed(seed)
        random.shuffle(data)

    # generates the splits -  note that since python 3.7 `dict` is ORDERED, and we want to keep the splits in
    # the same order they are in the config.
    # https://stackoverflow.com/questions/63075036/safe-to-assume-and-teach-that-a-python-dict-will-stay-ordered

    split_start_idx = 0
    for i, (split_data_file, split_num_or_pct) in enumerate(
        out_files_with_splits.items()
    ):
        # converts the number of examples or percentage into number of examples
        if split_num_or_pct > 1:
            split_num_examples = int(split_num_or_pct)
        else:
            split_num_examples = int(len(data) * split_num_or_pct)

        # turns the number of examples into the end index of the data
        split_end_idx = split_start_idx + split_num_examples

        # If this is the last split, make sure all of the data has been sorted into a split
        if i == len(out_files_with_splits) - 1:
            assert (
                split_end_idx == len(data) or split_end_idx == len(data) - 1
            ), "Make sure the number of samples in each split add up to include all of the data!"
            split_end_idx = len(data)

        # generates the split data and save it in the file for the split
        split_data = data[split_start_idx:split_end_idx]

        if seed >= 0:
            print(seed)
            split_data = sorted(split_data, key=lambda x: x["idx"])

        split_data_file.parent.mkdir(exist_ok=True)
        with open(split_data_file, "w") as f:
            # sorted(split_data, key=lambda x: x["idx"])
            f.write(
                "\n".join([json.dumps(line) for line in split_data]) + "\n"
            )

        split_start_idx = split_end_idx


# DATA FUNCTIONS SPECIFICALLY FOR EMNLP
def science_pipeline(
    in_data: List[dict], start: Optional[int] = 0, limit: Optional[int] = None
) -> List[dict]:
    limit = len(in_data) if limit is None else limit
    out_data = []
    for sample in in_data[start:]:
        out_data.append(
            {
                "idx": sample["idx"],
                "paper_id": sample["paper_id"],
                "title": sample["title"],
                "abstract": sample["abstract"],
                "context_section_header": sample["context_section_header"],
                "context_paragraph": sample["context_paragraph"],
                "sentence": sample["sentence"],
                "cited_ids": sample["cited_ids"],
                "y": sample["synthesis_paren"],
            }
        )
        if len(out_data) >= limit:
            break

    return out_data


def science_pipeline_gold_questions(
    in_data: List[dict], start: Optional[int] = 0, limit: Optional[int] = None
) -> List[dict]:
    # with open("data/full_text_db_with_missing_cite_ids_extended.json") as f:
    limit = len(in_data) if limit is None else limit
    out_data = []

    for sample in in_data[start:]:
        out_data.append(
            {
                "idx": sample["idx"],
                "paper_id": sample["paper_id"],
                "title": sample["title"],
                "abstract": sample["abstract"],
                "context_section_header": sample["context_section_header"],
                "context_paragraph": sample["context_paragraph"],
                "sentence": sample["sentence"],
                "cited_ids": sample["cited_ids"],
                "y": sample["synthesis_paren"],
                "snippet_surface": sample["snippet_rewrite"],
                "questions": {
                    question["question_id"]: question["question"]
                    for question in sample["questions"]
                },
            }
        )
        if len(out_data) >= limit:
            break

    return out_data


def science_pipeline_gold_qae(
    in_data: List[dict], start: Optional[int] = 0, limit: Optional[int] = None
) -> List[dict]:
    limit = len(in_data) if limit is None else limit
    out_data = []

    for sample in in_data[start:]:
        out_data.append(
            {
                "idx": sample["idx"],
                "paper_id": sample["paper_id"],
                "title": sample["title"],
                "abstract": sample["abstract"],
                "context_section_header": sample["context_section_header"],
                "context_paragraph": sample["context_paragraph"],
                "sentence": sample["sentence"],
                "cited_ids": sample["cited_ids"],
                "y": sample["synthesis_paren"],
                "snippet_surface": sample["snippet_rewrite"],
                "questions": {
                    question["question_id"]: question["question"]
                    for question in sample["questions"]
                },
                "answers": {
                    question["question_id"]: question["answer_text"]
                    for question in sample["questions"]
                },
                "evidence": {
                    question["question_id"]: question["evidence"]
                    for question in sample["questions"]
                },
            }
        )
        if len(out_data) >= limit:
            break

    return out_data


def science_pipeline_pred_questions(
    in_data: List[dict], start: Optional[int] = 0, limit: Optional[int] = None
) -> List[dict]:
    # with open("data/full_text_db_with_missing_cite_ids_extended.json") as f:
    limit = len(in_data) if limit is None else limit
    out_data = []

    pred_questions = {}
    snippet_files = glob.glob(
        "/net/nfs.cirrascale/s2-research/benjaminn/s2-contrastive-tldrs-internal/results/pipeline/Q"
        "GEN-gpt3-curie-gpt3-u9M0ueygQz-gldFalse_QA-gpt3-fewshot-qaspar-qa-gpt3-dense-lIK9KPrm0N-gld"
        "eFalse-gldaFalse_SYNTH-gpt3-curie-gpt3-endtoend-CtFXSxKsj9-trn-science-pipeline_gold_qae/*/"
        "paper_snippet.json"
    )
    for q_file_path in snippet_files:
        with open(q_file_path) as f:
            snippet = json.load(f)
            pred_questions[snippet["idx"]] = snippet["questions"]

    for sample in in_data[start:]:
        pred_qs = pred_questions.get(sample["idx"])
        if not pred_qs:
            print(f"Missing predicted questions for {sample['idx']}")
        out_data.append(
            {
                "idx": sample["idx"],
                "paper_id": sample["paper_id"],
                "title": sample["title"],
                "abstract": sample["abstract"],
                "context_section_header": sample["context_section_header"],
                "context_paragraph": sample["context_paragraph"],
                "sentence": sample["sentence"],
                "cited_ids": sample["cited_ids"],
                "y": sample["synthesis_paren"],
                "snippet_surface": sample["snippet_rewrite"],
                "questions": pred_qs,
            }
        )
        if len(out_data) >= limit:
            break

    return out_data


def science_filter_for_human_eval(
    in_data: List[dict], limit: int = None
) -> List[dict]:
    human_eval_idxs = {
        "104377",
        "113060",
        "114795",
        "119638",
        "135732",
        "137696",
        "138193",
        "154425",
        "1604.00400.1.1.1",
        "1604.00400.5.1.1",
        "1604.00727.1.1.1",
        "1606.03676.1.1.1",
        "1609.00425.1.1.1",
        "1609.00425.1.2.2",
        "1611.03599.4.1.1",
        "1611.03599.5.2.2",
        "1612.03226.1.1.1",
        "164331",
        "168296",
        "168810",
        "1702.03856.1.2.1",
        "1703.10344.1.1.2",
        "1703.10344.4.1.1",
        "1703.10344.6.2.1",
        "1704.06194.1.1.1",
        "171837",
        "173414",
        "176206",
        "176260",
        "1604.00400.2.1.1",
    }
    result = []
    for sample in in_data:
        if sample["idx"] in human_eval_idxs:
            result.append(sample)
    return result


def science_endtoend(in_data: List[dict], limit: int = None) -> List[dict]:
    """
    Pull out the fields that we're going to need for the end-to-end task.
    We'll use templates, so the templates can pull out the fields we need.
    """
    if limit is None:
        limit = len(in_data)

    with open("data/full_text_db_with_missing_cite_ids_extended.json") as f:
        full_text_db = json.load(f)

    out_data = []
    for sample in in_data:
        # most of the fields we need are pre-computed, but here we compute the ones that aren't

        # Introduction
        full_text = full_text_db[sample["paper_id"]]["full_text"]
        introduction = ""
        for section in full_text:
            if section["section_name"].lower().startswith("intro"):
                introduction = "\n".join(section["paragraphs"])
                break

        # Evidence paragraphs
        evidences = []
        questions = []
        for question in sample["questions"]:
            questions.append(question["question"])
            evidences.extend(
                [evidence["paragraph"] for evidence in question["evidence"]]
            )
        evidences = list(set(evidences))

        out_data.append(
            {
                "idx": sample["idx"],
                "sentence": sample["sentence"],
                "y": sample["synthesis_paren"],
                "context_section_header": sample["context_section_header"],
                "context_paragraph": sample["context_paragraph"],
                "gold_evidence": evidences,
                "gold_questions": questions,
                "title": sample["title"],
                "abstract": sample["abstract"],
                "introduction": introduction,
                "full_text": full_text,
            }
        )

        if len(out_data) >= limit:
            break

    return out_data


def science_endtoend_tsp(in_data: List[dict], limit: int = None) -> List[dict]:
    limit = len(in_data) if limit is None else limit
    out_data = []
    for sample in in_data:
        context = f"Title: \"{sample['title']}\"\n"
        context += "Paragraph with the snippet:\n"
        if sample["context_section_header"].strip():
            context += sample["context_section_header"].strip() + "\n"
        context += sample["context_paragraph"].strip() + "\n"
        context += "\n"
        context += f"Text snippet: \"{sample['sentence']}\""

        out_data.append(
            {
                "idx": sample["idx"],
                "context": context,
                "sentence": sample["sentence"],
                "y": sample["y"],
            }
        )

        if len(out_data) >= limit:
            break

    return out_data


def science_endtoend_tasp(
    in_data: List[dict], limit: int = None
) -> List[dict]:
    limit = len(in_data) if limit is None else limit
    out_data = []
    for sample in in_data:
        context = f"Title: \"{sample['title']}\"\n"
        context += f"Abstract: \"{sample['abstract']}\"\n"
        context += "Paragraph with the snippet:\n"
        if sample["context_section_header"].strip():
            context += sample["context_section_header"].strip() + "\n"
        context += sample["context_paragraph"].strip() + "\n"
        context += "\n"
        context += f"Text snippet: \"{sample['sentence']}\""

        out_data.append(
            {
                "idx": sample["idx"],
                "context": context,
                "sentence": sample["sentence"],
                "y": sample["y"],
            }
        )

        if len(out_data) >= limit:
            break

    return out_data


def science_endtoend_taisp(
    in_data: List[dict], limit: int = None
) -> List[dict]:
    limit = len(in_data) if limit is None else limit
    out_data = []
    for sample in in_data:
        context = f"Title: \"{sample['title']}\"\n"
        context += f"Abstract: \"{sample['abstract']}\"\n"
        context += f"Introduction: \"{sample['introduction']}\"\n"
        context += "Paragraph with the snippet:\n"
        if sample["context_section_header"].strip():
            context += sample["context_section_header"].strip() + "\n"
        context += sample["context_paragraph"].strip() + "\n"
        context += "\n"
        context += f"Text snippet: \"{sample['sentence']}\""

        out_data.append(
            {
                "idx": sample["idx"],
                "context": context,
                "sentence": sample["sentence"],
                "y": sample["y"],
            }
        )

        if len(out_data) >= limit:
            break

    return out_data


def science_endtoend_paper(
    in_data: List[dict], limit: int = None
) -> List[dict]:
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    limit = len(in_data) if limit is None else limit
    out_data = []
    for sample in in_data:
        full_text = {
            "title": sample["title"],
            "abstract": sample["abstract"],
            "full_text": sample["full_text"],
        }

        out_data.append(
            {
                "idx": sample["idx"],
                # "context": context,
                "sentence": sample["sentence"],
                "title": sample["title"],
                "abstract": sample["abstract"],
                "full_text": full_text,
                "y": sample["y"],
            }
        )

        if len(out_data) >= limit:
            break

    return out_data


def science_endtoend_qc(in_data: List[dict], limit: int = None) -> List[dict]:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    limit = len(in_data) if limit is None else limit
    out_data = []
    for sample in in_data:

        evidence = "\n\n".join([f'"{g}"' for g in sample["gold_evidence"]])

        context = f"Title: \"{sample['title']}\"\n"
        # context += f"Abstract: \"{sample['abstract']}\"\n"
        context += "Additional Context:\n"
        context += evidence
        context += "\n\n"
        context += "Paragraph with the snippet:\n"
        if sample["context_section_header"].strip():
            context += sample["context_section_header"].strip() + "\n"
        context += sample["context_paragraph"].strip() + "\n"
        context += "\n"
        context += f"Text snippet: \"{sample['sentence']}\""

        if len(tokenizer.tokenize(context)) > 4097 - 150:
            print(sample["idx"])

        out_data.append(
            {
                "idx": sample["idx"],
                "context": context,
                "sentence": sample["sentence"],
                "y": sample["y"],
            }
        )

        if len(out_data) >= limit:
            break

    return out_data


def science_endtoend_ablations(
    in_data: List[dict], start: int = 0, limit: int = None
) -> List[dict]:
    limit = len(in_data) if limit is None else limit
    out_data = []
    for sample in in_data[start:]:

        # get a list of unique evidences
        all_evidence = [
            evidence
            for question in sample["questions"]
            for evidence in question["evidence"]
        ]
        unique_evidence = []
        evidence_tracker = set()
        # if sample["idx"] == "103595":
        #     breakpoint()
        for evidence in all_evidence:
            if evidence["section"] + evidence["paragraph"] in evidence_tracker:
                continue
            else:
                evidence_tracker.add(
                    evidence["section"] + evidence["paragraph"]
                )
                unique_evidence.append(evidence)

        out_data.append(
            {
                "idx": sample["idx"],
                "paper_id": sample["paper_id"],
                "title": sample["title"],
                "abstract": sample["abstract"],
                "context_section_header": sample["context_section_header"],
                "context_paragraph": sample["context_paragraph"],
                "sentence": sample["sentence"],
                "snippet_rewrite": sample["snippet_rewrite"],
                "cited_ids": sample["cited_ids"],
                "questions": sample["questions"],
                "unique_evidence": unique_evidence,
                "y": sample["synthesis_paren"],
            }
        )
        if len(out_data) >= limit:
            break

    return out_data


def apply_and_save_jsonl_data(
    func: Callable[[List[dict]], List[dict]],
    in_file: Path,
    out_file: Path,
    **kwargs,
):
    """Wrapper function for reading and writing jsonl files."""
    with open(in_file) as f:
        try:
            in_data = [json.loads(line.strip()) for line in f]
        except json.decoder.JSONDecodeError:
            print("ERROR LOADING:")
            print(in_file)

    result = func(in_data, **kwargs)

    with open(out_file, "w") as f:
        for sample in result:
            f.write(json.dumps(sample) + "\n")


class Node:
    """Class for storing information about the order of constructing datasets."""

    def __init__(self, props: DictConfig) -> None:
        self.props = props
        self.children: List[Node] = []
        self.out_files: List[Path] = []

    def __repr__(self) -> str:
        return f"Node({self.props.name}, children={[child.props.name for child in self.children]})"


"""
These functions are only passed the input and output file names, so they read the input files and write
to the output files.
"""
FNS = {
    "science_endtoend": partial(apply_and_save_jsonl_data, science_endtoend),
    "science_endtoend_tsp": partial(
        apply_and_save_jsonl_data, science_endtoend_tsp
    ),
    "science_pipeline": partial(apply_and_save_jsonl_data, science_pipeline),
    "science_endtoend_paper": partial(
        apply_and_save_jsonl_data, science_endtoend_paper
    ),
    "science_endtoend_tasp": partial(
        apply_and_save_jsonl_data, science_endtoend_tasp
    ),
    "science_endtoend_taisp": partial(
        apply_and_save_jsonl_data, science_endtoend_taisp
    ),
    "science_endtoend_qc": partial(
        apply_and_save_jsonl_data, science_endtoend_qc
    ),
    "science_pipeline_gold_questions": partial(
        apply_and_save_jsonl_data, science_pipeline_gold_questions
    ),
    "science_endtoend_ablations": partial(
        apply_and_save_jsonl_data, science_endtoend_ablations
    ),
    "science_pipeline_gold_qae": partial(
        apply_and_save_jsonl_data, science_pipeline_gold_qae
    ),
    "science_filter_for_human_eval": partial(
        apply_and_save_jsonl_data, science_filter_for_human_eval
    ),
    "science_pipeline_pred_questions": partial(
        apply_and_save_jsonl_data, science_pipeline_pred_questions
    ),
}


# Also ignores nested directories and anything that starts with "_."
BLACKLIST = ["annotations.jsonl", "round_1_ben_results.jsonl"]


def main(all_data_file, breaking, should_use_git):
    """Propagate the data changes."""

    # Get the data config:
    all_data_file = Path(all_data_file)
    data_dir = all_data_file.parent
    conf = OmegaConf.load(data_dir / ".conf.yaml")

    # create a DAG for creating the datasets (though it's just a tree)
    dags = []
    all_nodes = {}
    for out_dir in conf.dirs:
        parent = out_dir.get("parent")
        if parent is None:
            node = Node(out_dir)
            dags.append(node)
        else:
            node = Node(out_dir)
            all_nodes[parent].children.append(node)

        all_nodes[out_dir.name] = node

    print("all tree nodes", all_nodes)

    # run the dag
    run_queue = dags.copy()
    while len(run_queue) > 0:
        to_run = run_queue.pop(0)
        print(f"Running: {to_run.props.name}")
        fn_name = to_run.props.get("fn")
        print("fn_name", fn_name)
        if fn_name is None:
            # generate new splits if needed
            if to_run.props.get("splits"):
                splits = {
                    data_dir
                    / to_run.props.name
                    / f"{split_name}{all_data_file.suffix}": num_or_pct
                    for split_name, num_or_pct in to_run.props.splits.items()
                }
                generate_splits(all_data_file, splits, conf.seed)
                to_run.out_files = [split_file for split_file in splits]
            # otherwise, just load in the splits that are already there
            else:
                to_run.out_files = [
                    fname
                    for fname in (data_dir / to_run.props.name).iterdir()
                    if Path(fname).name not in BLACKLIST
                    and not Path(fname).name.startswith("_.")
                ]
                print(to_run.out_files)
        else:
            # run the fn on the parent's results to generate the new files
            fn = FNS[fn_name]
            out_files = []
            for in_file in all_nodes[to_run.props.parent].out_files:
                out_file = data_dir / to_run.props.name / in_file.name
                suffix = to_run.props.get("out_file_suffix", out_file.suffix)
                out_file = out_file.with_suffix(suffix)

                (data_dir / to_run.props.name).mkdir(exist_ok=True)
                if (fn_args := to_run.props.get("args")) is not None:
                    fn(in_file, out_file, **fn_args)
                else:
                    fn(in_file, out_file)
                out_files.append(out_file)

            to_run.out_files = out_files

        print("\tFinished. Generated:")
        for fn in to_run.out_files:
            print(f"\t  {fn}")

        run_queue.extend(to_run.children)

    # update the version in the config
    old_version = conf.version
    major, minor = map(int, old_version.split("."))
    if breaking:
        major += 1
        minor = 0
    else:
        minor += 1
    new_version = f"{major}.{minor}"
    conf.version = new_version

    OmegaConf.save(conf, data_dir / ".conf.yaml")
    print(f"Updated version from {old_version} -> {new_version}")

    # make a new git commit
    if should_use_git:
        print("Saving to git")
        ga_cmd = shlex.split(f"git add {data_dir}")
        gc_cmd = shlex.split(
            f'git commit -m "Updates dataset {conf.name} from {old_version}->{new_version}"'
        )
        subprocess.run(ga_cmd, check=True)
        subprocess.run(gc_cmd, check=True)


if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument("all_data_file")
    breaking_help_str = (
        "Use flag if the dataset update breaks comprability between models."
        " Setting this flag will update the major version number."
    )
    argp.add_argument(
        "--breaking", action="store_true", help=breaking_help_str
    )
    argp.add_argument(
        "--disable_git", action="store_true", help=breaking_help_str
    )
    args = argp.parse_args()

    main(args.all_data_file, args.breaking, not args.disable_git)
