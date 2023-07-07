import os
from pathlib import Path
from typing import Any, Optional, Union, cast

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from decontext_exp.callbacks import (
    SMATrainLossLoggerCallback,
    ValidationReportingCallback,
)
from decontext_exp.data import Dataset, DatasetDict, FilterDataset
from decontext_exp.experiment.experiment_runner import ExperimentRunner
from decontext_exp.model import (
    GalacticaModel,
    LocalClassificationModel,
    LocalModel,
    LocalPEFTModel,
)
from decontext_exp.utils import Predictions, get_last_checkpoint_path


class LocalExperimentRunner(ExperimentRunner):
    """Wrapper for running local experiments

    Local experiments are ones whose models are run locally, using a combination of pytorch-lightening and
    Huggingface (as opposed to models behind APIs).

    Attributes:
        [super class's attributes]
        _hf_model: the underlying transormers.PreTrainedModel that is wrapped in a PytorchLightning trainer.
        trainer: the pl.Trainer that wraps the Huggingface model and allows for easy training and inference.
    """

    def initialize_experiment(self) -> None:
        """Initialize the experiment.

        In addition to the initialization in the superclass, set up the pytorch-lightning Trainer with the
        relevant loggers and callbacks. The Trainer is initialized using the experiment arguments.
        """

        super().initialize_experiment()

        if self.args.mode == "evaluate":
            return

        # initialize the trainer, overwrite self.model to be the pytorch lightning module
        self._hf_model = self.model  # type: ignore
        if self.args.model.prediction_type == "sequence":
            if self.args.model.get("peft") is not None:
                local_model_cls = LocalPEFTModel
            else:
                local_model_cls = LocalModel
        elif self.args.model.prediction_type == "sequence-decoder-only":
            local_model_cls = GalacticaModel
        else:
            local_model_cls = LocalClassificationModel

        num_train_batches = (
            len(self.data.train_dataset.dataloader) if self.data.train_dataset.dataloader else None
        )
        self.model = local_model_cls(
            self.args,
            self._hf_model,
            self.tokenizer,
            num_training_batches=num_train_batches,
        )

        # Set up logging
        loggers = [
            CSVLogger(
                save_dir=os.path.join(self.args.results_path, "logs"),
                name=None,
                version=None,
                flush_logs_every_n_steps=self.args.model.get("flush_logs_every_n_steps", 100),
            )
        ]

        if self.args.wandb:
            print("setting up wandb...")
            loggers.append(
                WandbLogger(
                    name=Path(self.args.results_path).stem,
                    project="contrastive-tldrs",
                    save_dir=self.args.results_path,
                )
            )

        # Set up checkpointing
        # TODO: if doing classification, we have to change what we're monitoring
        monitor, mode = ("val_loss", "min")
        filename = "ckpt-step={step:02d}-val_loss={val_loss:.2f}"
        if "rouge" in self.args.generation.metrics:
            monitor, mode = "val/avg_rouge12L", "max"
            filename += "-rouge={val/avg_rouge12L:.2f}"
        callbacks = [
            ModelCheckpoint(
                save_top_k=3,
                monitor=monitor,
                mode=mode,
                dirpath=Path(self.args.results_path) / "checkpoints",
                filename=filename,
                auto_insert_metric_name=False,
            ),
            ValidationReportingCallback(save_dir=os.path.join(self.args.results_path, "logs")),
            LearningRateMonitor(logging_interval="step"),
            SMATrainLossLoggerCallback(window_size=25),
        ]

        self.trainer = pl.Trainer(
            accelerator="auto",
            accumulate_grad_batches=self.args.model.accumulate_grad_batches,
            precision=self.args.model.precision,
            devices=torch.cuda.device_count()
            if torch.cuda.is_available()
            else None,  # smnth like torch.cuda.device_count()
            max_epochs=self.args.model.max_epochs,
            check_val_every_n_epoch=self.args.model.check_val_every_n_epoch,
            default_root_dir=os.path.join(self.args.results_path, "checkpoints"),
            log_every_n_steps=self.args.model.log_every_n_steps,
            val_check_interval=self.args.model.val_check_interval,
            callbacks=callbacks,
            logger=loggers,
            # Args for development/testing
            fast_dev_run=self.args.model.fast_dev_run,
            overfit_batches=self.args.model.overfit_batches,
            # strategy="deepspeed_stage_3_offload",
        )  # can we defer intializing these if we don't need them (e.g. if we are just evaluating?)

        print("[LocalExperimentRunner.initialize_experiment]", self.trainer._logger_connector)
        print("[LocalExperimentRunner.initialize_experiment]", self.trainer._loggers)

    def train(self, args: DictConfig, data: DatasetDict, model: pl.LightningModule) -> None:
        """Train or finetune the model.

        This method is called to start training a model. How this training starts will depend on the subclass.

        Args:
            args (DictConfig): Experiment configuration arguments.
            data (DatasetDict): The dataset with train and val dataloaders.
            model (Union[ApiModel, pl.LightningModule]): Model to train.
        """
        self.trainer.fit(
            model=model,
            train_dataloaders=data.train_dataset.dataloader,
            val_dataloaders=data.val_dataset.dataloader,
        )

    def _predict(
        self, args: DictConfig, dataset: Dataset, model: pl.LightningModule
    ) -> Predictions:
        """Run inference.

        This method generates predictions and returns them using a Predictions object, which
        wraps the actual strings as well as any metadata that should be saved.

        This method is only responsible for returning the Predictions and shouldn't
        save anything to disk.

        Args:
            args: config for the run
            dataloader: either the dev or test dataloader from a Dataset object
            model: the model to use to generate the predictions

        Returns:
            Predictions: A wrapper for the model predictions.
        """
        # The model checkpoint to use to generate the predictions can either be specified by the config
        # (and therefore at the command line) or automatically determined (which is the default).
        ckpt_path = args.model.get(
            "ckpt_path", get_last_checkpoint_path(args, key="rouge", best="max")
        )
        # Generate the predictions
        print("Prompt:")
        print(dataset.dataloader.dataset[0][dataset.x_label])
        want_to_continue = input("Do you want to continue with this prompt? y/N> ")
        if want_to_continue != "y":
            import sys

            sys.exit(0)

        predictions: Union[
            list[torch.tensor], list[dict[str, torch.tensor]]
        ] = self.trainer.predict(
            model=model,
            dataloaders=dataset.dataloader,
            return_predictions=True,
            ckpt_path=ckpt_path,
        )

        metadata: Optional[list[dict[str, Any]]] = None
        if self.args.model.prediction_type in ("sequence", "sequence-decoder-only"):
            prediction_outputs = [
                prediction
                for prediction_batch in predictions
                for prediction in self.tokenizer.batch_decode(
                    prediction_batch, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            ]
        elif self.args.model.prediction_type == "classification":
            dataset = cast(FilterDataset, dataset)
            predictions = cast(list[dict[str, torch.tensor]], predictions)
            metadata = [
                {
                    "probs": prob.tolist(),
                    "label": dataset.id2label[label.item()],
                    "label_map": dataset.id2label,
                }
                for prediction_batch in predictions
                for prob, label in zip(prediction_batch["probs"], prediction_batch["labels"])
            ]

            prediction_outputs = [
                dataset.id2label[label.item()]
                for prediction_batch in predictions
                for label in prediction_batch["labels"]
            ]

        return Predictions(predictions=prediction_outputs, metadata=metadata)
