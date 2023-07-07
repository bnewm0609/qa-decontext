import dataclasses
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from metrics import load_metrics
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

import decontext_exp.pipeline as pipeline
from decontext_exp.clarification_metric import jaccard_sentence
from decontext_exp.data import Dataset
from decontext_exp.experiment.experiment_runner import ExperimentRunner
from decontext_exp.utils import Predictions, get_prediction_save_dir


class PipelineExperimentRunner(ExperimentRunner):
    """Wrapper for running pipeline experiments.

    Supports experiments that combines question-generation, question-answering, and rewriting to perform
    decontextualization. This wrapper also makes it easy to swap out generated data for gold data to use
    for subsequent pipeline components.

    Attributes:
        [super class's attributes]
        steps_to_run: Which steps of the pipeline have to be run. If gold data is being used in place of
            earlier steps, then those setps will be missing from this list.
    """

    def initialize_experiment(self) -> None:
        """Initialize experiment and determine which steps of the pipeline need to be run."""

        super().initialize_experiment()

        self.steps_to_run = pipeline.load_steps_to_run(self.args)

    def _predict(self, args: DictConfig, dataset: Dataset, model: LightningModule) -> Predictions:
        """Run inference on a set of examples.

        Run each step in `self.steps_to_run` on each of the examples. Currently, the examples are not
        batched, so this can take a while.

        Args:
            args (DictConfig): Experiment configuration arguments.
            data (DatasetDict): The dataset.
            model (Union[ApiModel, pl.LightningModule]): Model to use for inference.

        Returns:
            Predictions: A wrapper for the model predictions. There is no metadata
                associated with these predictions.
        """

        # Results for each snippet are stored under a sub-directory for that snippet
        results_path_base = args.results_path
        for paper_snippet in dataset:
            args.results_path = results_path_base + f"/id_{paper_snippet.idx}"
            os.makedirs(args.results_path, exist_ok=True)
            if "qgen" in self.steps_to_run:
                pipeline.question_generation(args, paper_snippet)

            # Retriever is missing if e.g. we're inputting the entire paper to GPT4
            if "qa-retrieval" in self.steps_to_run and args.model.qa.retriever is not None:
                pipeline.run_retrieval(args, paper_snippet)

            if "qa-answers" in self.steps_to_run:
                pipeline.question_answering(args, paper_snippet)

            # "synth" stands for synthesis. It's the rewriting step that synthesizes the answers and the
            # original snippet
            if "synth" in self.steps_to_run:
                pipeline.synthesize(args, paper_snippet)

            # Save the final paper snippet:
            with open(Path(args.results_path) / "paper_snippet.json", "w") as f:
                # dump the whole dataclass
                json.dump(dataclasses.asdict(paper_snippet), f, ensure_ascii=False)

        # collect all of the results into one list[str]
        prediction_paths = glob.glob(results_path_base + "/*/paper_snippet.json")
        all_predictions: list[str] = []
        for prediction_path in prediction_paths:
            with open(prediction_path) as f:
                all_predictions.extend([json.loads(line)["decontextualized_snippet"] for line in f])

        return Predictions(all_predictions)

    def evaluate(self, args: DictConfig, split: str) -> None:
        """Run evaluation on a set of predictions.

        Run evaluations based on the given metrics on both the final outputs AND on the intermediate outputs.
        For example, if we perform:
        * question generation: calculate question precision/recall/F1
        * retrieval: calculate Recall@k and mean reciprocal rank
        * question answering: calculate ROUGE/BERTScore against gold answers
        Additionally, evaluate the final outputs. The metrics for evaluating final outputs and question answering
        are specified by the `args`. The metrics for question generation and retrieval are currently implemented
        in this method but in the future should be brought out.

        Args:
            args (DictConfig): Experiment configuration arguments.
        """

        save_dir = get_prediction_save_dir(args, split)

        # load in the gold data
        target_snippets = []
        with open(args.data.val.path) as f:
            target_snippets = [json.loads(line) for line in f]

        # We want to evaluate both:
        #  - the final result
        #  - the intermediate part directly after the gold data was used
        results = {}

        if not args.model.qgen.use_gold:
            # evaluate question generation
            paper_snippet_files = glob.glob(f"{args.results_path}/*/paper_snippet.json")
            paper_snippets = {}
            for paper_snippet in paper_snippet_files:
                with open(paper_snippet) as f:
                    snippet = json.load(f)
                    paper_snippets[snippet["idx"]] = snippet

            recall = []
            precision = []
            f1 = []
            # calculate the metrics
            for tgt_snippet in target_snippets:
                pred_snippet = paper_snippets.get(tgt_snippet["idx"])
                if pred_snippet is None:
                    continue

                tgt_q_matches = [False for _ in tgt_snippet["questions"].values()]

                num_targets = len(tgt_snippet["questions"].values())
                num_preds = len(pred_snippet["questions"].values())
                for pred_i, pred_q in enumerate(pred_snippet["questions"].values()):
                    for tgt_i, tgt_q in enumerate(tgt_snippet["questions"].values()):
                        if jaccard_sentence(pred_q, tgt_q) > 0.5:
                            tgt_q_matches[tgt_i] = True

                r = sum(tgt_q_matches) / num_targets
                p = sum(tgt_q_matches) / num_preds
                recall.append(r)
                precision.append(p)
                f1.append(2 * p * r / (p + r) if any(tgt_q_matches) else 0)

            results["qgen_prf1"] = {
                "p_mean": np.mean(precision),
                "r_mean": np.mean(recall),
                "f1_mean": np.mean(f1),
                "p": precision,
                "r": recall,
                "f1": f1,
            }
        elif args.model.qgen.use_gold and not args.model.qa.use_gold_evidence:
            # evaluate retrieval (bc we're using the gold questions)

            # load in the retrieval results
            retrieval_result_files = glob.glob(f"{args.results_path}/*/qa/retrieval_results.jsonl")

            # calculate recall @3 here rather than in metrics
            docs_by_qid = defaultdict(list)
            # extract the snippet id from the file path
            pattern = r"\/id_((?:\d+)|(?:\d+\.\d+\.\d+\.\d+\.\d+))\/"
            for retrieval_result_file in retrieval_result_files:
                try:
                    idx = re.findall(pattern, retrieval_result_file)[0]
                except IndexError:
                    breakpoint()
                with open(retrieval_result_file) as f:
                    # group by docs by qid
                    docs = [json.loads(line) for line in f]
                    for doc in docs:
                        docs_by_qid[f'{idx}.{doc["qid"]}'].append(doc["text"])

            # now calculate recall@k
            ks: dict[int, list[float]] = {3: [], 5: [], 10: []}
            ks_no_context_or_abstract: dict[int, list[float]] = {3: [], 5: [], 10: []}
            mrrs: list[float] = []
            mrrs_no_ctx_abs: list[float] = []
            for target_snippet in target_snippets:
                gold_docs = target_snippet["evidence"]
                for qid, evidence in gold_docs.items():
                    pred_evidence = docs_by_qid[f'{target_snippet["idx"]}.{qid}']

                    # calculate recall@k AND recall@k where we ignore retrieving the abstract
                    # or context snippet because these are always included in the QA prompt anyway.
                    gold_evidence = [e["paragraph"] for e in evidence]

                    gold_evidence_not_context_or_abstract = [
                        e["paragraph"]
                        for e in evidence
                        if e["section"].lower() != "abstract"
                        and e["paragraph"] != target_snippet["context_paragraph"]
                    ]

                    for k in ks:
                        r_k = (
                            len(set(pred_evidence[:k]) & set(gold_evidence))
                            / len(set(gold_evidence))
                            if gold_evidence
                            else 1
                        )
                        ks[k].append(r_k)

                        if gold_evidence_not_context_or_abstract:
                            r_k_noctxabs = len(
                                set(pred_evidence[:k]) & set(gold_evidence_not_context_or_abstract)
                            ) / len(set(gold_evidence_not_context_or_abstract))
                            ks_no_context_or_abstract[k].append(r_k_noctxabs)

                    # Also calculate MRR
                    for gold_e in gold_evidence:
                        try:
                            rank = pred_evidence.index(gold_e) + 1
                        except ValueError:
                            print("skipping gold evidence in mrr")
                            continue
                        mrrs.append(1.0 / rank)

                    for gold_e in gold_evidence_not_context_or_abstract:
                        try:
                            rank = pred_evidence.index(gold_e) + 1
                        except ValueError:
                            continue
                        mrrs_no_ctx_abs.append(1.0 / rank)

            results["retrieval_r@k"] = {str(k): np.mean(rs) for k, rs in ks.items()}
            results["retrieval_r@k_no_ctx_abs"] = {
                str(k): np.mean(rs) for k, rs in ks_no_context_or_abstract.items()
            }
            results["retrieval_mrr"] = {
                "mrr_mean": np.mean(mrrs),
                "mrr_mean_no_ctx_abs": np.mean(mrrs_no_ctx_abs),
                "rrs": mrrs,
                "rrs_no_ctx_abs": mrrs_no_ctx_abs,
            }

        # elif args.model.qa.use_gold_evidence and not args.model.qa.use_gold_answers:
        if not args.model.qa.use_gold_answers:  # and args.model.qgen.use_gold:
            # evaluate qa
            # load in the predicted snippets
            paper_snippet_files = glob.glob(f"{args.results_path}/*/paper_snippet.json")
            paper_snippets = {}
            for paper_snippet in paper_snippet_files:
                with open(paper_snippet) as f:
                    snippet = json.load(f)
                    paper_snippets[snippet["idx"]] = snippet

            # calculate the metrics
            predictions_qa = []
            targets_qa = []
            for tgt_snippet in target_snippets:
                pred_snippet = paper_snippets[tgt_snippet["idx"]]

                for qid in pred_snippet["answers"]:
                    predictions_qa.append(pred_snippet["answers"][qid])
                    targets_qa.append(tgt_snippet["answers"][qid])

            metrics = load_metrics(args.model.qa.get("metrics", []))
            for metric in metrics:
                results[f"qa_{metric.name}"] = metric.evaluate(predictions_qa, targets_qa, None)

        # evaluate final synthesis regardless of what else we evaluate
        predictions: list[str]
        targets: list[str]
        idxs: list[str]
        with open(save_dir / "predictions.json") as f:
            preds_json = [json.loads(line.strip()) for line in f]
            predictions = [entry["y_hat"] for entry in preds_json]
            targets = [entry["y_gold"] for entry in preds_json]
            idxs = [entry["idx"] for entry in preds_json]

        metadata: Optional[list[dict]]
        try:
            with open(save_dir / "metadata.jsonl") as f:
                metadata = [json.loads(line.strip()) for line in f]
        except FileNotFoundError:
            metadata = [
                {"idx": idx, "x_no_parse": tgt_snippet["sentence"]}
                for idx, tgt_snippet in zip(idxs, target_snippets)
            ]

        metrics = load_metrics(args.generation.metrics)
        for metric in metrics:
            # Some metrics require metadata
            if metric.requires_metadata:
                score = metric.evaluate(predictions, targets, metadata)
            else:
                score = metric.evaluate(predictions, targets)
            results[metric.name] = score

        with open(save_dir / "scores.json", "w") as f:
            json.dump(results, f)
