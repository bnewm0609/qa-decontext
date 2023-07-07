"""Entrypoint for running experiments."""

import sys

import hydra
from decontext_exp.experiment.api import ApiExperimentRunner
from decontext_exp.experiment.baseline import BaselineExperimentRunner
from decontext_exp.experiment.experiment_runner import ExperimentRunner
from decontext_exp.experiment.local import LocalExperimentRunner
from decontext_exp.experiment.pipeline import PipelineExperimentRunner
from decontext_exp.model import BASELINE_MODELS
from decontext_exp.utils import RunMode, hash_strs
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(args: DictConfig) -> None:
    """Entrypoint for running experiments.

    Registers resolvers, and sets up the experiment runner based on the passed configuration arguments.

    Args:
        args (DictConfig): The experiment configuration arguments.
    """
    OmegaConf.register_new_resolver("esc_slash", lambda x: x.replace("/", "-"))
    OmegaConf.register_new_resolver(
        "esc_period", lambda x: str(x).replace(".", "-")
    )
    # ignores trailing "/"
    OmegaConf.register_new_resolver(
        "extract_path",
        lambda path, num_segments: "-".join(
            path.strip("/").split("/")[-num_segments:]
        ),
    )
    OmegaConf.register_new_resolver("hash", hash_strs)

    if args.mode not in {RunMode.TRAIN, RunMode.PREDICT, RunMode.EVALUATE}:
        print(
            f'Mode {args.mode} is not recognized. Please choose among {{"train", "predict", "evaluate"}}'
        )
        sys.exit(0)

    # Use different experiment model functions if we're using a local model
    # (e.g. BART from Huggingface) versus a model behind an API (e.g. GPT3)
    experiment_runner: ExperimentRunner
    if args.model.interface == "api":
        experiment_runner = ApiExperimentRunner(args)
    elif args.model.interface == "local":
        if args.model.name in BASELINE_MODELS:
            experiment_runner = BaselineExperimentRunner(args)
        else:
            experiment_runner = LocalExperimentRunner(args)
    elif args.model.interface == "pipeline":
        experiment_runner = PipelineExperimentRunner(args)
    else:
        raise ValueError(
            f"Unknown model interface type: {args.model.interface}. Must be one of ['api', 'local', 'pipeline']."
        )

    # Run training, prediction, or evaluation based on user specification
    if args.mode == RunMode.TRAIN:
        print("Results will save to:", args.results_path)
        experiment_runner.train(
            args, experiment_runner.data, experiment_runner.model
        )
        print("Results saved to:", args.results_path)
    elif args.mode == RunMode.PREDICT:
        experiment_runner.predict(
            args, experiment_runner.data, experiment_runner.model, "val"
        )
    elif args.mode == RunMode.EVALUATE:
        experiment_runner.evaluate(args, "val")


if __name__ == "__main__":
    main()
