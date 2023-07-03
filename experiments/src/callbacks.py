"""Pytorch-lightning callbacks for training/finetuning models"""

import json
import os
from typing import Any

import numpy as np

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT


class ValidationReportingCallback(Callback):
    """Report validation scores here because the default CSVLogger combines all statistics in one file.
    
    Attributes:
        val_losses (list[float]): the validation losses for each batch.
        save_dir (str): the path to the directory to save the validation results in.
    """

    def __init__(self, save_dir: str) -> None:
        """Initialize the ValidatioReportingCallback.
        
        Args:
            save_dir (str): the directory to save the validation results.
        """

        super().__init__()
        self.val_losses: list[float] = []
        self.save_dir = save_dir

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.tensor,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Track validation losses for each batch.

        Args:
            outputs (torch.tensor): the average val loss for the batch.
            batch (Any): the batch that just finished.
        """

        # save the outputs
        batch_val_losses = [outputs.item()] * batch.input_ids.shape[0]
        self.val_losses.extend(batch_val_losses)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save validation results after the validation epoch is over.
        
        Args:
            trainer (Trainer): the trainer object controlling the training process.
        """

        avg_val_loss = np.mean(self.val_losses)
        val_results_dict = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "val_loss": avg_val_loss,
        }

        with open(os.path.join(self.save_dir, "val.json"), "a") as f:
            f.write(json.dumps(val_results_dict) + "\n")

        self.val_losses = []


class SMATrainLossLoggerCallback(Callback):
    """Callback for calculating a Simple Moving Average of the training loss.
    
    Attributes:
        window_size (int): how many batches to average over.
        last_losses (list[float]): the losses over the last at most window_size batches.
        curr_idx (int): the index of the loss in last_losses to overwrite.
    """

    def __init__(self, window_size: int = 50):
        """Initialize the callback.

        Args:
            window_size (int): how many batches to average over
        """

        super().__init__()
        if window_size <= 0:
            raise ValueError("Window size must be > 0.")
        self.window_size: int = window_size
        self.last_losses: list[float] = []
        self.curr_idx: int = 0

    def next_simple_moving_avg(self, loss: float) -> float:
        """Calculate the simple moving average given the loss.

        This function is defined as a separate function for easier testing.

        Args:
            loss (float): the most recent calculated loss.
        """
        # overwrite the oldest loss with the newest
        if len(self.last_losses) == self.curr_idx:
            self.last_losses.append(loss)
        else:
            self.last_losses[self.curr_idx] = loss
        
        # calculate the average and update the index of the oldest loss
        avg = np.mean(self.last_losses)
        self.curr_idx = (self.curr_idx + 1) % self.window_size
        return avg

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log the average loss after each batch.

        Args:
            outputs (STEP_OUTPUT): output from the most recent batch (which contains the loss).
        """
        if isinstance(outputs, torch.Tensor):
            loss = outputs.detach().item()
        else:
            # assume there's a key in the dictionary "loss"
            loss = outputs["loss"].detach().item()

        avg = self.next_simple_moving_avg(loss)
        self.log("sma_train_loss", avg)
