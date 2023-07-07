import math
import os
import random
import time
from typing import Any, Optional, Union

import anthropic
import openai
import pytorch_lightning as pl
import torch
from decontext_exp.data import DatasetDict
from decontext_exp.data_utils import OpenAIChatMessage
from decontext_exp.metrics import Rouge, load_metrics
from decontext_exp.utils import OPENAI_CHAT_MODEL_NAMES, Cache  # type: ignore
from omegaconf import DictConfig

# from peft import LoraConfig, TaskType, get_peft_model
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)


class LocalModel(pl.LightningModule):
    """Pytorch-lightning module for running Huggingface models on the cluster GPUs.

    Attributes:
        args: Experiment configuration arguments.
        model: Huggingface model to run.
        tokenizer: Huggingface tokenizer.
        num_training_batches: The number of batches in the training set.
        val_rouge_metric: Metric for calculating ROUGE during validation.
        val_metrics: All metrics that are calculated during validation.
    """

    def __init__(
        self,
        args: DictConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_training_batches: Optional[int] = None,
    ):
        """Initialize the Module.

        Args:
            args: Experiment configuration arguments.
            model: Huggingface model to run.
            tokenizer: Huggingface tokenizer.
            num_training_batches: The number of batches in the training data loader (used for lr sechuler).
        """

        super().__init__()
        self.args = args
        self.model = model
        # Tülu models have to be run in half precision to fit.
        if "7B" in args.model.name or "13B" in args.model.name:
            self.model = self.model.half()
        self.tokenizer = tokenizer
        self.num_training_batches = num_training_batches

        if "rouge" in self.args.generation.metrics:
            self.val_rouge_metric = Rouge()

        # filter out bert_score because it involves putting a model on the gpu
        # which takes up space that we need for finetuning the largest models]
        # (There are ways around this using multiple gpus, but I'm not going to deal
        # with it now.)
        val_metrics = [
            metric
            for metric in args.generation.metrics
            if metric != "bert_score"
        ]
        self.val_metrics = dict(zip(val_metrics, load_metrics(val_metrics)))

    def training_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        """Run training on the given batch."""

        output = self.model(**batch)
        self.log("train_loss", output.loss.cpu(), on_step=True)
        return output.loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        """Run validation on one batch and calculate validation meetrics."""

        output = self.model(**batch)

        # Calculate val_loss and rouge to log
        self.log("val_loss", output.loss, on_epoch=True, on_step=False)

        # For rouge, we actually need to generate tokens using the predict_step method:
        if "rouge" in self.args.generation.metrics:
            predictions_batch: torch.tensor = self.predict_step(
                batch, batch_idx
            )

            predictions = self.tokenizer.batch_decode(
                predictions_batch,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            targets = self.tokenizer.batch_decode(
                torch.where(
                    batch["labels"] != -100,
                    batch["labels"],
                    self.tokenizer.pad_token_id,
                ),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for pred, target in zip(predictions, targets):
                self.val_rouge_metric.add(pred, target)

        if "prf1" in self.val_metrics:
            # TODO fix this so that we somehow have access to label2id here...
            # so we can calculate average precision
            for pred, target in zip(output.logits, batch["labels"]):
                pred_label = torch.argmax(pred)
                self.val_metrics["prf1"].add(pred_label.item(), target.item())

        return output.loss

    def validation_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, list[EPOCH_OUTPUT]]
    ) -> None:
        """Combine validation metrics when the validation epoch is over and send them to wandb."""

        if "rouge" in self.args.generation.metrics:
            val_rouge = self.val_rouge_metric.process_scores()
            self.log("val/rouge1", val_rouge["rouge1"]["fmeasure_mean"])
            self.log("val/rouge2", val_rouge["rouge2"]["fmeasure_mean"])
            self.log("val/rougeL", val_rouge["rougeL"]["fmeasure_mean"])
            self.log("val/avg_rouge12L", val_rouge["rouge_avg_fmeasure_mean"])

            # reset the val scores:
            self.val_rouge_metric.reset()

        if "prf1" in self.val_metrics:
            val_prf1 = self.val_metrics["prf1"].process_scores()
            for name, score in val_prf1.items():
                if name.startswith("macro") and score is not None:
                    self.log(f"val/{name}", score)
            self.val_metrics["prf1"].reset()

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        """Run testing on one batch."""

        output = self.model(**batch)
        self.log("test_loss", output.loss)
        return output.loss

    def predict_step(self, batch, batch_id) -> torch.tensor:
        """Predict one batch by leveraging huggingface's generate method."""

        with torch.no_grad():
            output = self.model.generate(
                inputs=batch.input_ids,
                attention_mask=batch.attention_mask,
                num_beams=self.args.generation.num_beams,
                do_sample=False,
                max_length=self.args.generation.max_gen_length,
            )
        return output.cpu()

    def configure_optimizers(self) -> Any:
        """Create the optimizers for training using the experiment args and number of training batches."""

        # TODO log if self.num_training_batches is None
        # TODO is this the best lr scheduler?
        accum_grad_batches = self.args.model.accumulate_grad_batches
        num_training_steps = self.args.model.max_epochs * math.ceil(
            self.num_training_batches / accum_grad_batches
        )
        num_warmup_steps = int(
            num_training_steps * self.args.model.warmup_ratio
        )
        optimizer = AdamW(self.parameters(), lr=self.args.model.lr)
        optimizer_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_scheduler(
                    name="linear",
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                ),
                "interval": "step",
            },
        }
        return optimizer_dict


class LocalPEFTModel(LocalModel):
    """A LocalModel that uses parameter-efficient finetuning (eg LORA) with other weights frozen.

    These are implemented using the peft library from trasnformers."""

    def __init__(
        self,
        args: DictConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_training_batches: Optional[int] = None,
    ):
        """
        num_training_batches: number of batches in the training data loader (used for lr sechuler)
        """
        super().__init__(args, model, tokenizer, num_training_batches)

        # Commented out because PEFT was not always installed.
        # peft_config = LoraConfig(
        #     task_type=TaskType.SEQ_2_SEQ_LM,
        #     inference_mode=False,
        #     r=4,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        # )

        # self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()


class GalacticaModel(LocalModel):
    """LocalModel that is a decoder-only model."""

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        """Generate predictions for one batch.

        Filter out the inputs from the targets when predicting and implement left-padding because
        this is a decoder-only model.
        """
        # restructure the data
        # 1 is a special token (the padding token)
        inpts = torch.where(batch["labels"] == -100, batch["input_ids"], 1)
        targets = torch.where(batch["labels"] != -100, batch["input_ids"], 1)

        inpts = self.tokenizer.batch_decode(inpts, skip_special_tokens=True)
        targets = self.tokenizer.batch_decode(
            targets, skip_special_tokens=True
        )

        pad_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inpts = self.tokenizer(inpts, return_tensors="pt").to(self.args.device)
        self.tokenizer.padding_side = pad_side

        output = self.model.generate(
            inputs=inpts.input_ids,
            attention_mask=inpts.attention_mask,
            num_beams=self.args.generation.num_beams,
            do_sample=False,
            max_length=self.args.generation.max_gen_length,
        )

        # zero-out the input from the prediction
        input_token_mask = torch.ones_like(
            output, device=self.args.device, dtype=bool
        )
        if output.shape[1] < batch["labels"].shape[1]:
            input_token_mask[:, : output.shape[1]] = (
                batch["labels"][:, : output.shape[1]] != -100
            )
        else:
            input_token_mask[:, : batch["labels"].shape[1]] = (
                batch["labels"] != -100
            )
        output = torch.where(input_token_mask, output, 1)  # 1 is padding token
        return output


class LocalClassificationModel(LocalModel):
    """Local model for classification."""

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        output = self.model(**batch)
        pred_labels = torch.argmax(output.logits, dim=1)
        pred_probs = torch.softmax(output.logits, dim=1)
        return {"labels": pred_labels, "probs": pred_probs}


class ApiModel(PreTrainedModel):
    """Class for models that are behind APIs (e.g. GTP3)"""

    def __init__(self, args: DictConfig) -> None:
        self.name = args.model.name

    def resize_token_embedding(size: int):
        raise ValueError("Cannot resize the token embeddings of an api model")


class GPT3Model(ApiModel):
    """Training and predicting with GPT3.

    Attributes:
        _name: the name of the model.
        cache: A local cache to avoid sending the same request twice.
        params: The default generation parameters sent to the model with the prompt.
        is_chat_model: True if the model requires hitting the OpenAI Chat endpoint.
        is_anthropic_model: True if the model is Claude.
    """

    def __init__(self, args: DictConfig) -> None:
        """Initialize GPT3 model.

        This involves setting up the cache, API key, and default parameters.

        Args:
            args (DictConfig): Experiment configuration arguments.
        """
        self._name = args.model.name
        self.cache = Cache.load()
        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.params = {
            "model": self.name,
            "logprobs": 5,
            "user": "[ANON]",
            "temperature": args.generation.get("temperature", 0.7),
            "top_p": args.generation.get("top_p", 1.0),
            "max_tokens": args.generation.get("max_tokens", 150),
            "stop": ["END"],
        }

        # chatgpt_models = ["gpt-3.5-turbo-0301", "gpt-3.5-turbo", "gpt-4"]
        self.is_chat_model = self.name in OPENAI_CHAT_MODEL_NAMES
        self.is_anthropic_model = False

    @property
    def name(self):
        """Getter for the name attribugt"""

        return self._name

    @name.setter
    def name(self, value):
        """Write the value to _name and the params list."""

        self._name = value
        self.params["model"] = value

    def prompt_with_cache(self, params):
        """Send a request to the API with the given params if they haven't been used yet.

        This is done by creating a unique key based on the params dict and having the cache handle running
        the function to prompt the model if the key is not in the cache.

        Args:
            params (dict): The parameters used to prompt the model with.
        """
        key = "-".join(
            [
                f"{param_k}_{param_v}"
                for param_k, param_v in params.items()
                if param_k not in {"user", "prompt"}
            ]
        )
        if self.is_chat_model:
            for message in params["messages"]:
                key += f"-msg_r_{message['role']}_c_{message['content']}"
        else:
            key += f"-prompt_{params['prompt']}"  # [:100]  # that should be enough, right?

        def prompt():
            # GPT4 has a lower rate-limit.
            if "gpt-4" in self.name:
                time.sleep(0.25)
            else:
                time.sleep(0.1)
            try:
                if self.is_chat_model:
                    response = openai.ChatCompletion.create(**params)
                else:
                    response = openai.Completion.create(**params)
            except openai.error.InvalidRequestError:
                print(
                    "Stopping to investigate why there was an invalid request to the API..."
                )
                breakpoint()
            return response.to_dict_recursive()

        return self.cache.query(key, prompt)

    def __call__(self, text_prompt: str) -> tuple[dict, dict]:
        """Perform inference on the model with the given prompt.

        Overwrite the params with the given prompt. For Chat models, use a simple system message and put the
        prompt in the user message."""
        params = {k: v for k, v in self.params.items()}
        if self.is_chat_model:
            params["messages"] = [
                {
                    "role": "system",
                    "content": "You are a helpful scientific assistant, helping to provide additional"
                    " context for text extracted from scientific papers.",
                },
                {"role": "user", "content": text_prompt},
            ]
            params.pop("logprobs")  # chat endpoint doesn't accept logprobs
        else:
            params["prompt"] = text_prompt
        return params, self.prompt_with_cache(params)


class GPT3ChatModel(GPT3Model):
    """Run inference with the Chat endpoint and arbitrary messages."""

    def __call__(self, messages_prompt: list[OpenAIChatMessage]) -> tuple[dict, dict]:  # type: ignore[override]
        params = {k: v for k, v in self.params.items()}
        params.pop("logprobs")
        params["messages"] = [message.dict() for message in messages_prompt]
        return params, self.prompt_with_cache(params)


class ClaudeModel(GPT3Model):
    """Call the Anthropic Claude API."""

    def __init__(self, args: DictConfig) -> None:
        """Initailize the model.

        Create an anthropic client and use a smaller set of parameters compared to OpenAI.
        """

        self._name = args.model.name
        self.cache = Cache.load()
        self.client = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
        self.params = {
            "stop_sequences": [anthropic.HUMAN_PROMPT],
            "model": self.name,
            "max_tokens_to_sample": args.generation.get("max_tokens", 150),
        }

        # Technically it is a chat model, but we're querying it like it's a completion model.
        self.is_chat_model = False
        self.is_anthropic_model = True

    def prompt_with_cache(self, params):
        """Prompt with the anthropic library instead of the OpenAI one."""
        key = "-".join(
            [f"{param_k}_{param_v}" for param_k, param_v in params.items()]
        )

        key += f"-prompt_{params['prompt']}"  # [:100]  # that should be enough, right?

        def prompt():
            time.sleep(0.1)
            try:
                response = self.client.completion(**params)
            except anthropic.ApiException:
                print(
                    "Stopping so you can determine why there was an API exception."
                )
                breakpoint()
            return response

        return self.cache.query(key, prompt)


class FewShotModel:
    """Converts models and datasets into a few-shot format.

    This is different from the dataset.TemplateDataset because the fewshot examples are assembled using a number
    of arguments like:
        `fewshot.x_prefix`: which comes before the input
        `fewshot.x_suffix`: which is added to the input
        `fewshot.y_prefix`: which comes after x_suffix but before the gold output.
    This class also supports including the task instructions (`fewshot.instructions`) before or after the few-shot
    examples.
    """

    @classmethod
    def create_prompt(
        cls,
        fewshot: DictConfig,
        in_context_examples: list[dict[str, str]],
        dataset: DatasetDict,
    ) -> list[str]:
        """Create the fewshot example prompt using the passed in context examples

        Args:
            fewshot (DictConfig): A config containing the parameters for creating the few-shot in-context prompt.
            in_context_examples (list[dict[str, str]]): List of few shot examples. Each example has an `x_label`
                and a `y_label`
            dataset (DatasetDict): the dataset the in-context-examples come from. Used for accessing the `x_label`
                and `y_label` values.

        Returns:
            A list of str where each str is an element (usually a line) of the few-shot prompt. These will be
            stitched together later.
        """

        instruction_position = fewshot.get(
            "instructions_position", "before_examples"
        )
        prompt = []
        if instruction_position == "before_examples":
            prompt = [fewshot.instructions]
        for ex in in_context_examples:
            x = fewshot.x_prefix + ex[dataset.train_dataset.x_label]
            x += fewshot.get("x_suffix", "")
            y = fewshot.y_prefix + ex[dataset.train_dataset.y_label]
            prompt.append(x + "\n" + y)

        if instruction_position == "after_examples":
            prompt.append(fewshot.instructions)
        return prompt

    @classmethod
    def create_messages_dataset(
        cls,
        args: DictConfig,
        model: Union[ApiModel, pl.LightningModule],
        dataset: DatasetDict,
    ):
        """
        An attempt to store message templates in few-shot examples. This should not be used and
        dataset.TemplateDataset should be used instead.
        """

        import omegaconf

        fewshot = args.model.few_shot
        x_label = dataset.val_dataset.x_label
        new_dataset = []
        for example in dataset.val_dataset:
            messages = []
            for message in fewshot.messages:
                try:
                    message["content"] = message["content"].substitute(
                        "{{example}}", example[x_label]
                    )
                    message["content"] = message["content"].substitute(
                        r"{{instructions}}", fewshot.instructions
                    )
                except omegaconf.errors.InterpolationResolutionError:
                    breakpoint()
                messages.append(dict(message))

            ex_copy = {k: v for k, v in example.items()}
            ex_copy[x_label] = messages
            new_dataset.append(ex_copy)
        dataset.val_dataset.data = new_dataset
        dataset.val_dataset.create_data_loader()
        return model, dataset.val_dataset, None

    @classmethod
    def fill_prompt_example(
        cls,
        fewshot: DictConfig,
        prompt: list[str],
        example: dict[str, str],
        x_label: str,
    ) -> dict:
        """Add a particular validation example to a prompt.

        Args:
            fewshot (DictConfig): Parameters for the few-shot in-context learning prompts.
            prompt (list[str]): The in-context learning prompt with few shot examples. Each element is a line.
            example (dict[str, str]): The example with the x_label and y_label.
            x_label (str)): The key to the input ('x') label in the examples.

        Returns:
            dict containing the example with the new x value (and all other keys and values the same.
        """

        x = fewshot.x_prefix + example[x_label]
        x += fewshot.get("x_suffix", "")
        y = fewshot.y_prefix.strip()
        ex_copy = {k: v for k, v in example.items()}
        ex_copy[x_label] = "\n\n".join(prompt + [x + "\n" + y])
        return ex_copy

    @classmethod
    def convert_to_few_shot(
        cls,
        args: DictConfig,
        model: Union[ApiModel, pl.LightningModule],
        dataset: DatasetDict,
    ):
        """Convert the validation dataset of a DatasetDict to an in-context learnin (ICL) few-shot dataset.

        This is done by replacing each example in the dataset with a version where the in-context learning
        examples are pre-pended and the example is slotted into the ICL format. There are a few different ways
        to select in-context training examples (eg randomly, first, all, or a given set of indices).

        To do this, first the in-context examples are chosen. Then they are used to create a prompt. Next, the
        prompt is filled with validation examples. Finally, the dataset is updated and returned.

        Args:
            args (DictConfig): Experiment configuration arguments. Includes an `args.model.few_shot`config with
                the parameters needed to create the ICL prompts.
            model (Union[ApiModel, pl.LightningModule]): The model used for ICL.
                (TODO: remove this because it's not used)
            dataset (DatasetDict): The dataset that contains the val dataset to be overwritten.

        Returns:
            A tuple containing the model, new validation dataset, and the in-context examples used.
        """

        # Currently only supports few-shot in-context learning
        fewshot = args.model.few_shot

        data = []
        if fewshot.ic_examples == "leave-one-out":
            # Used for exploring prompt-robustness / different prompts. Creates prompts for each training example
            # using the other training examples as in-context examples.
            assert len(dataset.train_dataset.data) <= 5
            in_context_training = []
            val_examples = []
            for i in range(len(dataset.train_dataset.data)):
                in_context_training.append(
                    dataset.train_dataset.data[:i] + dataset.train_dataset.data[i + 1 :]  # type: ignore
                )
                val_examples.append(dataset.train_dataset.data[i])

                prompt = cls.create_prompt(
                    fewshot,
                    dataset.train_dataset.data[:i] + dataset.train_dataset.data[i + 1 :],  # type: ignore
                    dataset,
                )

                filled_example = cls.fill_prompt_example(
                    fewshot,
                    prompt,
                    dataset.train_dataset.data[i],
                    dataset.train_dataset.x_label,
                )
                data.append(filled_example)
        else:
            if fewshot.ic_examples is None:
                # select n random examples from training to use as in-context learning examples if none are given
                selection_strategy = fewshot.get(
                    "selection_strategy"
                )  # , "random")
                if selection_strategy == "random":
                    rng = random.Random(args.model.seed)
                    in_context_training = rng.sample(
                        dataset.train_dataset.data, k=fewshot.num_shots
                    )
                elif selection_strategy == "first":
                    # if not, just take the first num_shot examples (in order)
                    in_context_training = list(
                        dataset.train_dataset.data[: fewshot.num_shots]
                    )
                else:
                    raise ValueError(
                        f"Unknown selection_strategy for choosing few shot examples: {selection_strategy}"
                    )
            elif fewshot.ic_examples == "all":
                # Use the whole training set as in-context examples
                in_context_training = [
                    instance for instance in dataset.train_dataset.data
                ]
            else:
                # Assume that we are passed a list of idxs that should be used as in-context examples.
                # this doesn't maintain order of the examples!!! (which we probably want...)
                in_context_training = [
                    instance
                    for instance in dataset.train_dataset.data
                    if instance["idx"] in fewshot.ic_examples
                ]

            prompt = cls.create_prompt(fewshot, in_context_training, dataset)

            # now create the in-context labels
            for ex in dataset.val_dataset:
                ex = cls.fill_prompt_example(
                    fewshot, prompt, ex, dataset.val_dataset.x_label
                )
                data.append(ex)

        # update the data
        dataset.val_dataset.data = data
        dataset.val_dataset.create_data_loader()
        return model, dataset.val_dataset, in_context_training


# --- Model for baselines for decontextualization
class IdentityModel:
    """Outputs the input.

    This is a useful baseline for decontextualization.
    """

    def __init__(self, args: DictConfig):
        self.args = args

    def __call__(self, sample: dict) -> str:
        return sample["x"]


class RetrievalModel:
    """Given a document (or set of documents) and a question/query, outputs the most relevant sections.

    This can be combined with a model like GPT3 or flan-t5 that will generate
    the answers to the question based on the retrieved passages. For current implementation see `pipeline.py`.
    Perhaps in the future, implementation will be moved here.
    """

    pass


BASELINE_MODELS = {
    # Baselines specifically for decontextualization
    "baseline_identity": IdentityModel,
}

# DECODER_ONLY_MODELS = {
#     "facebook/galactica-125m",
#     "facebook/galactica-1.6b",
# }


def load_ranking_model(model_name: str) -> PreTrainedModel:
    pass


def load_model(
    args: DictConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Return the model and tokenizer for the given model name in `args`."""

    model: PreTrainedModel
    if args.model.interface == "api":
        if args.model.name in OPENAI_CHAT_MODEL_NAMES:
            model = GPT3ChatModel(args)
        elif "claude" in args.model.name:
            model = ClaudeModel(args)
        else:
            model = GPT3Model(args)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    elif args.model.interface == "retrieval":
        model = None
        tokenizer = None
    elif args.model.interface == "pipeline":
        model, tokenizer = None, None
    else:
        if args.model.prediction_type in ("sequence", "sequence-decoder-only"):
            if args.model.name in BASELINE_MODELS:
                model = BASELINE_MODELS[args.model.name](args)
                tokenizer = None
            elif args.model.prediction_type == "sequence-decoder-only":
                if "30B" in args.model.name:
                    print("Using 8bit!")
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model.name, device_map="auto", load_in_8bit=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model.name
                    )
                tokenizer = AutoTokenizer.from_pretrained(args.model.name)
                # breakpoint()
                if "galactica" in args.model.name:
                    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(1)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model.name)
                tokenizer = AutoTokenizer.from_pretrained(args.model.name)
                # if 'pad_token' not in tokenizer.special_tokens_map:
                #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # if 'eos_token' not in tokenizer.special_tokens_map:
                #     tokenizer.add_special_tokens({'eos_token': '[EOS]'})
                model.resize_token_embeddings(len(tokenizer))
        elif args.model.prediction_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model.name, num_labels=args.model.num_labels
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model.name)
            model.resize_token_embeddings(len(tokenizer))
        elif args.model.prediction_type == "ranking":
            model = load_ranking_model(args.model)
    return (model, tokenizer)
