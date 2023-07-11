import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from decontext.experiments.data import Dataset, DatasetDict, load_dataset
from decontext.experiments.metrics import load_metrics
from decontext.experiments.model import ApiModel, FewShotModel, load_model
from decontext.experiments.utils import (
    Predictions,
    RunMode,
    convert_paths_to_absolute,
    get_prediction_save_dir,
)


class ExperimentRunner(object):
    """Wrapper for running experiments.

    This class is subclassed for running experiments locally, using an API, baselines,
    retrieval systems, and pipelined systems.

    Attributes:
        args (DictConfig): Experiment configuration arguments.
        model (transformers.PretrainedModel): The model to run the experiment with.
        tokenizer (transformers.PretrainedModel): The tokenizer associated with the model.
        data: (DatasetDict): Object with the dataset splits.
    """

    def __init__(self, args: DictConfig):
        """Initialize the Experiment Runner

        Args:
            args (DictConfig): Experiment configuration arguments.
        """

        self.args = args
        self.initialize_experiment()

    def initialize_experiment(self) -> None:
        """Initialize the experiment.

        Initialize the experiment by doing the following:
            - setting the random seed
            - setting the device
            - converting all relative paths in config to absolute paths
            - creating the results directory
            - saving the experiment arguments in the new results directory
            - setting up logging
        """

        # Set the seed
        random.seed(self.args.model.seed)
        np.random.seed(self.args.model.seed)
        torch.manual_seed(self.args.model.seed)
        torch.cuda.manual_seed(self.args.model.seed)

        # Set the timestamp
        self.args.timestamp = str(datetime.now())

        # Set the device
        self.args.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert all paths to relative paths (in-place)
        convert_paths_to_absolute(self.args)

        # Create the results directory
        # the format is going to be `[task]/[model+training_dataset]/[eval_dataset]`
        # results_dir = os.path.join("results", self.args.task, self.model._id + self.data.train._id)
        if self.args.notes is not None:
            self.args.results_path += f"_note-{self.args.notes}"

        os.makedirs(
            self.args.results_path, exist_ok=True
        )  # TODO log if this already exists?

        # Save the experiment config (if training)
        if self.args.mode == RunMode.TRAIN:
            OmegaConf.save(
                config=self.args,
                f=os.path.join(self.args.results_path, "config.yaml"),
            )

        # Set up logging (TODO)

        # Set up model
        if self.args.mode != "evaluate" or self.args.model.interface == "api":
            self.model, self.tokenizer = load_model(self.args)
        else:
            self.model, self.tokenizer = None, None

        # Set up data
        self.data: DatasetDict = load_dataset(self.args, self.tokenizer)

    def train(
        self,
        args: DictConfig,
        data: DatasetDict,
        model: Union[ApiModel, pl.LightningModule],
    ) -> None:
        """Train the model.

        This method is called to start training a model. How this training starts will depend on the subclass.

        Args:
            args (DictConfig): Experiment configuration arguments.
            data (DatasetDict): The dataset with train and val dataloaders.
            model (Union[ApiModel, pl.LightningModule]): Model to train.

        Raises:
            NotImplementedError.
        """

        raise NotImplementedError("Should be overridden by subclass")

    def predict(
        self,
        args: DictConfig,
        data: DatasetDict,
        model: Union[ApiModel, pl.LightningModule],
        split: str,
    ) -> None:
        """Runs inference and post-processes predictions.

        This method is called to start running inference using a given model. This method then saves the
        results from the run along with any relevant metadata.

        Args:
            args (DictConfig): Experiment configuration arguments.
            data (DatasetDict): The dataset.
            model (Union[ApiModel, pl.LightningModule]): Model to run inference with.
            split (str): one of "val" or "test" indicating whether to run on the validation or test set of the
                given dataset.

        Raises:
            NotImplementedError.
        """

        save_dir = get_prediction_save_dir(args, split)

        # save the prediction args separately
        OmegaConf.save(config=self.args, f=Path(save_dir) / "config.yaml")

        if self.args.model.get("few_shot") is not None:
            # if self.args.model.name == "gpt-4":
            #     model, dataset, _ = FewShotModel.create_messages_dataset(args, model, data)
            # else:
            # pass
            model, dataset, fs_examples = FewShotModel.convert_to_few_shot(
                args, model, data
            )
            with (Path(save_dir) / "few_shot_train.jsonl").open("w") as f:
                for instance in fs_examples:
                    f.write(json.dumps(instance) + "\n")
        else:
            dataset = data.val_dataset
        predictions = self._predict(args, dataset, model)

        if args.model.interface != "retrieval":
            predictions_json: list[dict[str, str]] = []

            # mypy isn't able to recognize that dataset is an iterable, so it complains
            # about using `zip`
            for gold, pred in zip(dataset, predictions.predictions):  # type: ignore
                pred_sample = {
                    "idx": gold["idx"],
                    "x": gold[dataset.x_label],
                    "y_gold": gold[dataset.y_label],
                    "y_hat": pred,
                }
                predictions_json.append(pred_sample)

            with (save_dir / "predictions.json").open("w") as f:
                for line in predictions_json:
                    f.write(
                        json.dumps(line, default=dict, ensure_ascii=False)
                        + "\n"
                    )

            # Save the metadata if it exists:
            if predictions.metadata is not None:
                metadata_json = []
                for gold, pred, metadata in zip(
                    dataset, predictions.predictions, predictions.metadata
                ):
                    metadata_sample = {
                        "idx": gold["idx"],
                        **metadata,
                    }
                    metadata_json.append(metadata_sample)

                with (save_dir / "metadata.jsonl").open("w") as f:
                    for line in metadata_json:
                        f.write(json.dumps(line) + "\n")

            predictions = "\n".join(predictions.predictions) + "\n"  # type: ignore
            with (save_dir / "predictions.txt").open("w") as f:
                f.write(predictions)

        print("Predictions saved to:", save_dir)

    def _predict(
        self,
        args: DictConfig,
        dataset: Dataset,
        model: Union[ApiModel, pl.LightningModule],
    ) -> Predictions:
        """Run inference on a set of examples.

        In contrast to `predict`, this method only runs inference and does not handle saving predictions to disk.
        It is called by `predict`. The exact implementation will depend on what type of model is being used.

        Args:
            args (DictConfig): Experiment configuration arguments.
            data (DatasetDict): The dataset.
            model (Union[ApiModel, pl.LightningModule]): Model to use for inference.

        Returns:
            Predictions: A wrapper for the model predictions and any metadata.

        Raises:
            NotImplementedError: if the sublcasses don't override it.
        """

        raise NotImplementedError("Should be overridden by subclass")

    def evaluate(self, args: DictConfig, split: str) -> None:
        """Run evaluation on a set of predictions.

        Given a results path in the args, load in the predictions and metadata and use them to
        evaluate the system against the gold data.
        The metrics used for evaluation are also specfied by the args.

        Args:
            args (DictConfig): Experiment configuration arguments.
        """

        save_dir = get_prediction_save_dir(args, split)

        predictions: list[str]
        targets: list[str]
        idxs: list[str]
        with open(save_dir / "predictions.json") as f:
            preds_json = [json.loads(line.strip()) for line in f]
            predictions = [entry["y_hat"] for entry in preds_json]
            targets = [entry["y_gold"] for entry in preds_json]
            idxs = [entry["idx"] for entry in preds_json]

        # If there is no metadata, just set the indices to be the metadata
        metadata: Optional[list[str]]
        try:
            with open(save_dir / "metadata.jsonl") as f:
                metadata = [json.loads(line.strip()) for line in f]
        except FileNotFoundError:
            metadata = idxs

        metrics = load_metrics(args.generation.metrics)
        results = {}
        for metric in metrics:
            # Some metrics require metadata, so include it if necessary
            if metric.requires_metadata:
                score = metric.evaluate(predictions, targets, metadata)
            else:
                score = metric.evaluate(predictions, targets)
            results[metric.name] = score

        with open(save_dir / "scores.json", "w") as f:
            json.dump(results, f)
