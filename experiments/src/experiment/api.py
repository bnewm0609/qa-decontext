import math
import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional

import anthropic
import openai
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from contrastive_tldrs.data import Dataset, DatasetDict
from contrastive_tldrs.experiment.experiment_runner import ExperimentRunner
from contrastive_tldrs.model import ApiModel
from contrastive_tldrs.utils import (
    Predictions,
    get_openai_price_estimate,
    run_subprocess,
)


class ApiExperimentRunner(ExperimentRunner):
    """Wrapper for running experiments with API models
    
    API models are ones from companies like Anthropic and OpenAI. Right now inference for
    Anthropic and OpenAI are supported, but only training for OpenAI is supported.
    """
    
    def initialize_experiment(self) -> None:
        """Initialize the experiment"""

        super().initialize_experiment()

    def build_train_cmd(
        self,
        train_file: Path,
        val_file: Path,
        base_model: str,
        n_epochs: int,
        learning_rate_multiplier: float,
        suffix: str,
    ) -> list[str]:
        """Create the cli command for training OpenAI models.

        See here for more information: https://platform.openai.com/docs/guides/fine-tuning

        Args:
            train_file (Path): The path to training data.
            val_file (Path): The path to validation data.
            base_model (str): The name of the model to finetune.
            n_epochs (int): The number of epochs to finetune for.
            learning_rate_multiplier (float): Controls the learning rate used for finetuning.
            suffix (str): An identifier for the run to append to the model name.

        Returns:
            list[str]: cli command for training OpenAI models tokenized using shlex.split
        """

        train_cmd = f"""openai api fine_tunes.create \
        -t {train_file} \
        -v {val_file} \
        -m {base_model} \
        --n_epochs {n_epochs} \
        --learning_rate_multiplier {learning_rate_multiplier} \
        --suffix {suffix}"""

        return shlex.split(train_cmd)

    def get_suffix(self, model_id: str, train_id: str) -> Optional[str]:
        """Get an identifier for a run.

        The identifier is appended to the model name when saved on OpenAI. There is a maximum suffix length of
        40 characters.
        
        Args:
            model_id (str): An identifier for the base model.
            train_id (str): An identifier for the training args.
        
        Returns:
            str: if the suffix is 40 or fewer characters.
            None: if the suffix is too long.
        """
        suffix = model_id + "_" + train_id

        # if the suffix is too long, remove the model name from the beginning because this will appear elsewhere
        # in the fine-tuned model name.
        if len(suffix) > 40:
            _, suffix = suffix.split("_", 1)

        # if the suffix is *still* too long, then just return None - we are probably not trying
        # to use a fine-tuned model here and the suffixes only matter for the fine-tuned models.
        if len(suffix) > 40:
            print(f"WARNING!!! Suffix is too long at {len(suffix)} chars. Suffix is: {suffix}")
            return None

        assert len(suffix) <= 40, f"Suffix is too long at {len(suffix)} chars. Suffix is: {suffix}"

        # openai doesn't allow "_" in their suffixes for some reason...
        suffix = suffix.replace("_", "-")
        return suffix

    def train(self, args: DictConfig, data: DatasetDict, model: ApiModel) -> None:
        """Call the Openai Fine-tuning endpoint to train a model on our dataset.

        Args:
            args (DictConfig): contains the configuration variables for the run.
            data (DatasetDict): Data with train and val dataloaders.
            model (ApiModel): A dummy variable that represents the model we will call the API to train.
        """

        # Turns out that this happens at the command line
        # Guide is here: https://beta.openai.com/docs/guides/fine-tuning/advanced-usage

        # For generation they suggest fewer batches and lower learning rate, so I'll start with
        # lr = 0.05 and n_epochs = 1

        # first confirm that the data is in a valid format
        # data_validate_cmd = "openai tools fine_tunes.prepare_data -f train.jsonl"
        # Maybe it's better to put this in testing? This should be possible to do
        # by using the openai cli module: i.e.
        # https://github.com/openai/openai-python/blob/3edecbee24102299dd6e4a35af031780e9ad0f9a/openai/cli.py#L521

        # then train the model
        suffix = self.get_suffix(
            self.args.model._id,
            self.args.data.train._id,
        )

        assert suffix is not None, "Suffix is too long!"

        train_cmd = self.build_train_cmd(
            train_file=self.args.data.train.gpt3_path,
            val_file=self.args.data.val.gpt3_path,
            base_model=self.args.model.name,
            n_epochs=self.args.model.max_epochs,
            learning_rate_multiplier=self.args.model.lr,
            suffix=suffix,
        )

        log_lines = []
        results_file_id: str
        if args.model.get("for_sure_run_train_command_and_spend_dollars", False):
            for line in run_subprocess(train_cmd):
                print(line)
                log_lines.append(line)
                results_file_match = re.search(r"Created fine-tune: (ft-\w+)", line)
                if results_file_match is not None:
                    results_file_id = results_file_match.group(1)
        else:
            print(train_cmd)
            import sys

            sys.exit(0)

        # now save the logs
        with open(Path(self.args.results_path) / "logs" / "stdout.txt", "w") as f:
            f.write("\n".join(log_lines) + "\n")

        # and save the training metrics
        get_results_file_cmd = shlex.split(f'openai api fine_tunes.results -i "{results_file_id}"')
        with open(self.args.results_path / "logs" / "metrics.csv", "w") as f:
            subprocess.run(get_results_file_cmd, stdout=f)


    def _predict(self, args: DictConfig, dataset: Dataset, model: LightningModule) -> Predictions:
        """Run inference using the specified model

        The model might be a fine-tuned GPT3 model, a base GPT3 model or a Claude model. Use the model to generate
        predictions and track the cost. Before running, print the first prompt for manual verification.

        Args:
            args (DictConfig): Config for the run.
            dataloader (Dataset): Either the dev or test dataloader from a DataSet object.
            model (LightningModule): The model to use to generate the predictions (unused in this case)
        
        Returns:
            Predictions: A wrapper for the model predictions and metadata.
        """
        # First, get the model name
        # e.g. "curie:ft-semantic-scholar:curie-e1-lr0-05-rewrite-2022-10-18-17-31-12"

        suffix = self.get_suffix(args.model._id, args.data.train._id)
        ft_model_name = None
        if suffix is not None:
            model_prefix = f"{args.model.name}:ft-semantic-scholar:{suffix}"
            models = openai.FineTune.list()["data"]
            # breakpoint()
            for ft_model in models:
                if ft_model["fine_tuned_model"] is not None and ft_model[
                    "fine_tuned_model"
                ].startswith(model_prefix):
                    ft_model_name = ft_model["fine_tuned_model"]
        else:
            model_prefix = (
                "NO PREFIX SPECIFIED (SUFFIX IS PROBABLY TOO LONG)."
                " THIS IS OK IF YOU'RE USING A NOT FINETUNED MODEL."
            )
            models = []

        if ft_model_name is None:
            print("Unable to find model with name:", model_prefix)
            print("Found models:", [ft_model["fine_tuned_model"] for ft_model in models])

            price_estimate = get_openai_price_estimate(
                args.model.name, dataset=dataset, max_gen_len=args.generation.max_tokens
            )
            price_estimate_per_example = price_estimate / len(dataset)
            print(
                f"Estimated price of prompts (total): ${math.ceil(price_estimate * 100) / 100.0 :.2f}"
            )
            print(f"Estimated price per prompt: ${price_estimate_per_example}")

            # Print a sample prompt
            if self.model.is_chat_model:
                print("Example prompt:")
                for ex in dataset.data[0][dataset.x_label]:
                    print(ex.role.upper(), end=":\n")
                    if len(ex.content) > 5000:
                        print(ex.content[:5000] + " [... omitted for length]")
                    else:
                        print(ex.content)
                    print("\t" + "-" * 50)
            else:
                print(f"Example prompt: {dataset.data[0][dataset.x_label]}")

            # Warn and ask for confirmation before continuing
            response = "y"
            if args.get("warn_before_hitting_api", True):
                response = input(
                    "[WARNING]: Unable to find the finetuned model name. Loading the NOT fine-tuned version"
                    " instead. Confirm this is what you want. (y/N) > "
                )

            if response != "y":
                print("Quitting...")
                import sys

                sys.exit(0)
            ft_model_name = args.model.name

        # set the model name to update the default parameters
        self.model.name = ft_model_name

        # now use it to predict with
        predictions = []
        metadata = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for example in dataset.data:  # [:10]:
            params, response = self.model(example[dataset.x_label])
            # params["prompt"] = example["prompt"]
            # response = self.prompt_with_cache(params)
            if self.model.is_anthropic_model:
                predictions.append(response["completion"])  # type: ignore
            elif self.model.is_chat_model:
                predictions.append(response["choices"][0]["message"]["content"])  # type: ignore
            else:
                predictions.append(response["choices"][0]["text"])  # type: ignore
            metadata.append(
                {
                    "params": params,
                    "response": response,
                }
            )
            # total_token_usage += response["usage"]["total_tokens"]  # type: ignore

            if self.model.is_anthropic_model:
                total_prompt_tokens += anthropic.count_tokens(example[dataset.x_label])
                total_prompt_tokens += anthropic.count_tokens(response["completion"])  # type: ignore
            else:
                total_prompt_tokens += response["usage"]["prompt_tokens"]  # type: ignore
                total_completion_tokens += response["usage"]["completion_tokens"]  # type: ignore

        # TODO figure out what to do for claude here...
        price_actual = get_openai_price_estimate(
            self.model.name, dataset=dataset, prompt_tokens=total_prompt_tokens
        )
        price_actual_per_example = price_actual / len(dataset)
        print(
            f"Estimated price of prompts (total): ${math.ceil(price_estimate * 100) / 100.0 :.2f}"
        )
        print(f"Estimated price per prompt: ${price_estimate_per_example}")
        print(f"Actual price of prompts (total): ${math.ceil(price_actual * 100) / 100.0 :.2f}")
        print(f"Actual price per prompt: ${price_actual_per_example}")

        # this is kinda janky... but to include the price info I'm appending it to the metadata. Any time
        # we need to use the metadata, there should be one item for prediction, so there will be one metadata
        # item which does not have an associated prediction, and it will be the price.
        metadata.append({"price_per_prompt": price_actual_per_example, "price_total": price_actual})

        return Predictions(predictions, metadata)
