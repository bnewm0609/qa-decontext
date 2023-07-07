from typing import Union

import pytorch_lightning as pl
from omegaconf import DictConfig

from decontext_exp.data import Dataset, DatasetDict
from decontext_exp.experiment.experiment_runner import ExperimentRunner
from decontext_exp.model import ApiModel
from decontext_exp.utils import Predictions


class BaselineExperimentRunner(ExperimentRunner):
    """Wrapper for running baseline experiments

    Baseline experiments are ones whose outputs are specified in the data file.
    """

    def train(
        self, args: DictConfig, data: DatasetDict, model: Union[ApiModel, pl.LightningModule]
    ) -> None:
        """Train the model.

        Raises NotImplementedError because Baseline models can't be trained.

        Args:
            args (DictConfig): Experiment configuration arguments.
            data (DatasetDict): The dataset.
            model (Union[ApiModel, pl.LightningModule]): Model to train.

        Raises:
            NotImplementedError.
        """

        raise NotImplementedError("Cannot train the Baseline Experiment Runner")

    def _predict(
        self, args: DictConfig, dataset: Dataset, model: Union[ApiModel, pl.LightningModule]
    ) -> Predictions:
        """Run inference

        Run inference by outputing the target specified in the data file.

        Args:
            args (DictConfig): Experiment configuration arguments.
            data (DatasetDict): The dataset.
            model (Union[ApiModel, pl.LightningModule]): Model to use for inference.

        Returns:
            Predictions: A wrapper for the model predictions.
        """

        predictions = []
        for inpt in dataset.data:
            prediction = model(inpt)
            predictions.append(prediction)

        return Predictions(predictions=predictions)
