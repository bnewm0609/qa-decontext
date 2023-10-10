# qa-decontextualization

This repository contains the code for running experiments from the paper: __A Question Answering Framework for Decontextualizing User-facing Snippets from Scientific Documents__. This code enables running
1. Models running locally (with a combination of pytorch-lightning and huggingface).
2. Models behind APIs (OpenAI, Claude).
3. End-to-end methods and a pipeline-based method (in which we generate clarifying questions, answer them and rewrite the snippets to include the answers).

For a pip-installable package to run decontextualization on your own data, please see the [`main`](https://github.com/bnewm0609/qa-decontextualization) branch.

## Installation
Install by running:
```
git clone [repo-url]
sh scripts/setup_env.sh
```
(This assumes that you have a version of pytorch installed that matches with your cuda version.)

## Usage

The main entrypoint into training all of these models is the `decontext_exp/run.py` script, which can be run like so:

```bash
python decontext_exp/run.py mode={train,predict,evaluate} task=<task_name> model=<model_name> data=<data_name> generation=<generation_params_name>
```


* `mode` has three options:
    1. `train`: trains a new model for the task (this is not used in the experiments in the paper).
    2. `predict`: uses an already trained model (the result of `train`, a pretrained model or a model behind an API) to generate outputs.
    3. `evaluate`: uses existing predictions (the result of `predict`) to calculate scores based on outputs.
* `task_name` is the task. It is used for specifying where the results are saved. Currently, the tasks are `endtoend` for end-to-end methods and `pipeline` for the pipeline ones.
* `model` points to a config in the `configs/model` directory that contains information about the model. The name of the results directory depends on the parameters in `model`.
* `data` points to a config in the `configs/data` directory that contains information about the data used to train, validate, and evaluate the model. It contains the paths to the datasets. It also contains any templates that are used.
* `generation` points to a config in the `configs/generation` directory that contains hyperparameters for how to generate outputs and metrics for evaluation. The generation and metrics are both used for validation when `mode=train`, only generation when `mode=predict` and only metrics when `mode=evaluate`.

Configs are managed with [`hydra`](https://hydra.cc/).

### Running End-to-end Experiments

_Set Up_
* OpenAI: Run `export OPENAI_API_KEY='yourkey'`
* Claude: Run `export ANTHROPIC_API_KEY='yourkey'`
* Tülu:
    * Download The Tülu-30B model from huggingface [here](https://huggingface.co/allenai/tulu-30b)
    * Install [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and run `python -m bitsandbytes` and ensure the cuda version is correct.

_Predict_
```bash
python decontext_exp/run.py mode=predict task=decontext/endtoend model=<model> data=data/emnlp/endtoend_ablations data.template=<template> generation=<generation>
```
Running prediction generates a `predictions.jsonl` file. Each line in this file is a json dictionary with an `idx`, `x`, `y_hat`, and `y_gold` key. To quickly compare the predictions to the references, you can run:

```bash
sh scripts/preview_predictions.sh <results_path>
```
which will use [`jq`](https://jqlang.github.io/jq/download/) to show only the `y_hat` and `y_gold` fields.

_Evaluate_
```bash
python scripts/add_original_sentence_to_metadata.py data/emnlp23/science/all_data.jsonl {results_path}/metadata.jsonl
python decontext_exp/run.py mode=evaluate ...
```

Running the first script adds the original snippet to the metadata for computing SARI scores. The second script evaluates the predictions.


Running the evaluation generates a `scores.json` file. This file has many scores in it, both averages over the entire dataset as well as sentence or document level scores, so to preview just the summary statistics, you can run:
```bash
sh scripts/preview_scores.sh <results_path>
```

which will use [`jq`](https://jqlang.github.io/jq/download/) to output a high-level summary of the scores.


Each of the cells in Table 1 has a slightly different command. See `scripts/run_ablations.sh` for the commands to run to get each of these values. The predict and evaluate scripts are both included. (For the Tülu results, the commands assume the Tülu model is in a project-level directory called `big_models`.)

For an example of generating some decontextualizations, you can run:
```python
python decontext_exp/run.py mode=predict model=pipeline_better_qgen data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_qae_human_eval/ task=pipeline generation=gpt3-endtoend model.qgen.use_gold=True model.qa.use_gold_evidence=True model.qa.use_gold_answers=True model.qa.retriever=dense
```
This will create ~30 decontextualizations which you can explore.

### Running Pipeline Experiments
Running the pipeline experiments (to reproduce Table 3) is quite similar, but there are a few additional overrides to be aware of.

```bash
python decontext_exp/run.py mode={predict,evaluate} task=pipeline model=<model> data=pipeline data.base_dir=<data_dir> model.qa.retriever={dense,null} model.qgen.use_gold={True,False} model.qa.use_gold_evidence={True,False}
```
Unlike in the end-to-end experiments, there is **no** need to run `scripts/add_original_sentence_to_metadata.py`---the original sentences are automatically added to the metadata.
If `model.qgen.use_gold=True`, then gold questions are used. If `model.qa.use_gold_evidence=True` then human-selected evidence paragraphs are used.

See `scripts/run_pipeline.sh` for the commands to run to reprocduce the values in Table 3 of the paper.

## Data
There are four datasets that are used, all in `data/emnlp/science/`:
* `endtoend_ablations` - used for end-to-end experiments
* `pipeline_gold_questions` - used for pipeline experiments
* `pipeline_pred_questions` - used for pipeline experiments
* `pipeline_gold_qae` - used for pipeline experiments
* `pipeline_gold_qae_human_eval` - used for pipeline experiments (performing human eval on a small set of examples)

If you make any changes to `data/emnlp/science/all_data.json`, you can propagate them to rest of the datasets by running
```bash
python scripts/propagate_data_changes.py data/emnlp/science/all_data.json
```

## Configs and Templates
Expermint config files can be found under the `configs` directory. The directory `data` holds the dataset configs, `model` holds the model ones, and `generation` holds the generation/evaluation ones.

Templates are in the `configs/templates` directory. They are a field of the dataset (specifically the `configs/data/template*` datasets). Some of the dataset configs include the template directly and others include a path to the template yaml file (the latter is useful for overriding at the command line). The choice is somewhat arbitrary. The templates in `configs/templates` are for the end-to-end experiments while the templates that are included in the yaml files in `configs/data/template*` tend to be for the pipeline experiments.

The templates themselves are written in yaml files with jinja syntax. For OpenAI chat models, the messages are specified in yaml (see `configs/data/template-chatgpt3.yaml` for an example). For the completion models, Claude, and Tülu, the templates are just strings.
