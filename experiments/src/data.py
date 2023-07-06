import json
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, TypeVar, Union

import torch
import yaml
from jinja2 import DebugUndefined, Template
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader
from transformers import BatchEncoding, PreTrainedTokenizer

from contrastive_tldrs.data_utils import (
    OpenAIChatMessage,
    QasperFullText,
    QasperSection,
)


class Dataset:
    """Super class that contains logic for datasets.

    Attributes:
        args: Contains configuration material for the run.
        split: Which split this data belongs to (one of "train", "val" (validation), or "test").
        tokenizer: Tokenizer (usually from HuggingFace). Used for both API models and local models.
        data: List of dictionaries of samples that make up the dataset.
        x_label: The key of the dictionary of each that represents the input.
        y_label: The key of the dictionary of each sample that contains the gold output.
    """

    def __init__(self, args: DictConfig, tokenizer: PreTrainedTokenizer, split: str) -> None:
        """Initialize a dataset by reading data and creating the dataloader.
        
        Args:
            args: The configuration for the experiment.
            tokenizer: Tokenizer (usually from HuggingFace) for tokenizing inputs.
            split: Which split this data belongs to (one of "train", "val" (validation), or "test").
        """

        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.data = self.read_data(args.data.get(split).path)

        # default keys for in the data list dicts
        self.x_label = "x"
        self.y_label = "y"

        self.create_data_loader() 

    def create_data_loader(self):
        """Bundle `self.data` into a pytorch DataLoader.
        
        Only shuffles the training set.
        """

        self.dataloader = DataLoader(
            self.data,
            batch_size=self.args.model.batch_size,
            collate_fn=self.collate,
            shuffle=self.split == "train",
        )

    def read_data(self, data_path_or_name: str) -> Sequence[Any]:
        """Read data from path and returns it as a Sequence that can be loaded into a torch dataloader.

        Subclasses will override this method.

        Args:
            data_path_or_name: The path to the dataset (or name if using a huggingface dataset).

        Returns:
            Sequence (usually a list) containing the data.
        """

        raise NotImplementedError

    def collate(self, batch: Sequence[Any]) -> BatchEncoding:
        """Tokenize the batch and convert it to tensors.

        Collate examples from the batch for input to the model in the torch DataLoader.

        Args:
            batch: The samples that constitue a single batch.
        
        Returns:
            The collated batch for input into a Huggingface model.
        """

        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        """Return the example at the given index from the dataset."""
        return self.data[idx]

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator for iterating over the dataset."""
        return iter(self.data)


class RewriteDataset(Dataset):
    """The default dataset for HuggingFace models.
    
    Data is read from a `jsonl` file, tokenized and batched using a Huggingface transformers tokenizer."""
    def read_data(self, data_path_or_name: str) -> Sequence[Any]:
        """Read data from `jsonl` file.

        Args:
            data_path_or_name: The path to the json lines file containing the dataset.

        Returns:
            list[dict] containing the data.
        """

        data = []
        with open(data_path_or_name) as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append(sample)
        return data

    def collate(self, batch: Sequence[Any]) -> BatchEncoding:
        """Tokenize each batch using the huggingface tokenizer.
        
        See overridden function in Dataset for information on args and return type.
        """

        inputs, targets = zip(*[(sample[self.x_label], sample[self.y_label]) for sample in batch])

        inputs = self.tokenizer(list(inputs), padding=True, truncation=True)
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(list(targets), padding=True, truncation=True)

        # Replace all "pad token ids" with -100 because we want to ignore them when calculating the loss
        targets["input_ids"] = [
            [(token_id if token_id != self.tokenizer.pad_token_id else -100) for token_id in label]
            for label in targets["input_ids"]
        ]

        inputs["labels"] = targets["input_ids"]
        return inputs.convert_to_tensors(tensor_type="pt")


class RewriteDatasetDecoderOnly(Dataset):
    """Default dataset for decoder-only Huggingface models (e.g. GPT2 or Galactica)."""
    def __init__(self, args: DictConfig, tokenizer: PreTrainedTokenizer, split: str) -> None:
        """Initialize the dataset with a different x_label and y_label.
        
        See overridden function in Dataset for more information
        """

        super().__init__(args, tokenizer, split)
        self.x_label = "prompt"
        self.y_label = "completion"

    def read_data(self, data_path_or_name: str) -> Sequence[Any]:
        """Read data from `jsonl` file.

        Args:
            data_path_or_name: The path to the `jsonl` file containing the dataset.

        Returns:
            list[dict] containing the data.
        """

        data = []
        with open(data_path_or_name) as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append(sample)
        return data

    def collate(self, batch: Sequence[Any]) -> BatchEncoding:
        """Use the huggingface tokenizer to tokenize each batch.

        Place the inputs and targets side by side rather than tokenizing them separately because the
        model takes them both at the same point.
        """

        inputs, targets = zip(*[(sample[self.x_label], sample[self.y_label]) for sample in batch])

        inputs = self.tokenizer(
            inputs, targets, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        )
        inputs["labels"] = torch.clone(inputs["input_ids"])
        token_type_ids = inputs.pop("token_type_ids")
        inputs["labels"][
            token_type_ids == 0
        ] = -100  # we don't want to caculate loss on the input tokens

        return inputs


class ChatTemplateDataset(RewriteDataset):
    """A dataset for templating prompts for API endpoints.

    This dataset was designed with the OpenAI chat and completion endpoints in mind, but also
    can work with other API providers.
    This dataset probably should not be 
    Could also be used for completition endpoint if so desired.
    Probably shouldn't be used for fewshot learning. Look at `models.FewShotModel`
    for the code for doing few shot learning.

    Updates self.data directly because we send text to GPT3.

    Attributes:
        template: The template to format the data in.
    """

    def __init__(self, args: DictConfig, tokenizer: PreTrainedTokenizer, split: str) -> None:
        """Initialize the dataset.
        
        In addition to loading the data, also load the template. Also, save the template and a sample example
        in the results directory for debugging the template. Finally, try to create the dataloader, but it
        is not necessary because we are not doing batching. The `train.jsonl` might not exist, so skip this
        step if it doesn't.
        """

        # set up the dataset
        self.args = args
        self.split = split
        self.tokenizer = tokenizer

        self.template = self.load_template(args.data.template)
        self.data = self.read_data(args.data.get(split).path, self.template)

        # it's good to have an example data sample for debugging and reproducibility
        # so save one in the results dir:
        if self.split == "val":
            save_dir = Path(args.results_path) / (args.data.val._id + "-" + args.generation._id)
            save_dir.mkdir(parents=True, exist_ok=True)

            with open(save_dir / "sample_val_data.json", "w") as f:
                json.dump(self.data[0], f, default=dict, ensure_ascii=False)

            # additionally, save the template:
            with open(Path(args.results_path) / "template.json", "w") as f:
                if isinstance(self.template, str):
                    f.write(self.template)
                else:
                    template_dict = [t.dict() for t in self.template]
                    json.dump(template_dict, f)

        # default keys for in the data list dicts
        self.x_label = "x"
        self.y_label = "y"

        try:
            self.create_data_loader()
        except ValueError:
            if split != "train":
                raise ValueError
            else:
                print(f"No data found! File {args.data.get(split).path} is empty.")

    def load_template(self, template: Union[str, list]) -> Union[list[OpenAIChatMessage], str]:
        """Load the template from the config.

        The passed template takes one of three forms:
        1. a list[dict[str, str]] (for the OpenAI Chat Endpoint). The keys are the role ("user", "system")
            and the values are the template for that message. The dict[str, str] is converted into an 
            `OpenAIChatMessage`.
        2. a string containing the template (for the OpenAI Completion or Claude endpoints)
        3. a string with a yaml filepath to either of the two above template types.
        The template strings are jinja templates.

        Args:
            template (Union[str, list]): The template or a path to a yaml file with template.
        
        Returns:
            list[OpenAIChatMessage] for the OpenAI Chat API case and a str for the Completion or Claude
            cases with the template. The template is not filled at this point.
        """

        # there are a few choices for template:
        if isinstance(template, str) and template.startswith("https://"):
            # assume that the template is a public google sheet and read it into pandas as a csv
            raise NotImplementedError()
        elif isinstance(template, str) and len(template) < 256 and Path(template).is_file():
            # read the template from the file path
            with open(template) as f:
                if template.endswith("yaml"):
                    template = yaml.safe_load(f)["template"]
                    if isinstance(template, list):
                        template = [OpenAIChatMessage(**item) for item in template]
                else:
                    template = f.read()
        elif isinstance(template, str):
            # assume the template is for a non-chat model
            template = template
        elif isinstance(template, list) or isinstance(template, ListConfig):
            # assume that the passsed thing is the template dict itself
            template = [OpenAIChatMessage(**item) for item in template]
        else:
            raise ValueError("Template must be either a list, url or path to a valid file")

        return template

    def read_data(  # type: ignore
        self, data_path_or_name: str, template: Union[str, list[OpenAIChatMessage]]  # type: ignore
    ) -> Sequence[Any]:  # type: ignore
        """Read the data by filling in the template.
        
        Args:
            data_path_or_name (str): path to the `jsonl` file containing the data.
            template (Union[str, list[OpenAIChatMessage]]): the jinja template that will be filled in.
        
        Returns:
            A list of samples, where the value at `self.data[i][self.x_label]` is the template filled in with
            the data for the `i`th sample.
        """

        data = []
        with open(data_path_or_name) as f:
            for line in f:
                sample = json.loads(line.strip())
                # overwrite full_text if it's present:
                if "full_text" in sample:
                    sample["full_text"] = QasperFullText(
                        title=sample["title"],
                        abstract=sample["abstract"],
                        full_text=[
                            QasperSection(**section) for section in sample["full_text"]["full_text"]
                        ],
                    )

                    while (
                        len(self.tokenizer.tokenize(str(sample["full_text"]))) > 8080 - 100
                    ):  # 755:  # len(prompt)
                        if not sample["full_text"].full_text[-1].paragraphs:
                            sample["full_text"].full_text.pop(-1)

                        sample["full_text"].full_text[-1].paragraphs.pop(-1)

                # substitute any variables into the template
                if isinstance(template, str):
                    new_messages = Template(template, undefined=DebugUndefined).render(sample)
                else:
                    new_messages = []
                    for chat_message in template:
                        new_message_content = Template(
                            chat_message.content, undefined=DebugUndefined
                        ).render(
                            sample
                        )  # any extra elements will be ignored
                        new_messages.append(
                            OpenAIChatMessage(
                                role=chat_message.role,
                                content=new_message_content,
                            )
                        )
                data.append(
                    {
                        "idx": sample["idx"],
                        "x": new_messages,
                        "y": sample["y"],
                    }
                )
        if self.split == "train":
            # save a few training examples with the prompts for debugging
            with open(Path(self.args.results_path) / "filled_train_templates.jsonl", "w") as f:
                for sample in data:
                    try:
                        f.write(json.dumps(sample) + "\n")
                    except TypeError:
                        f.write(
                            json.dumps(
                                {
                                    "idx": sample["idx"],
                                    "x": [json.loads(message.json()) for message in sample["x"]],
                                    "y": sample["y"],
                                }
                            )
                        )

            print(
                "Saved train prompts to:",
                Path(self.args.results_path) / "filled_train_templates.jsonl",
            )
        return data

    def collate(self, batch: Sequence[Any]) -> BatchEncoding:
        """Use the huggingface tokenizer to tokenize each batch"""
        inputs, targets = zip(*[(sample[self.x_label], sample[self.y_label]) for sample in batch])

        inputs = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=1023, return_tensors="pt"
        )
        inputs["labels"] = torch.clone(inputs["input_ids"])
        token_type_ids = inputs.pop("token_type_ids")
        inputs["labels"][
            token_type_ids == 0
        ] = -100  # we don't want to caculate loss on the input tokens

        return inputs


class PipelineDataset(Dataset):
    """Dataset for the pipeline experiments.

    The self.x_label and self.y_label are different from other datasets. Additionally, loading the dataset
    is outsourced to a different module (the `pipeline` module) because the data that should be loaded
    depends on which steps of the pipeline are run. For example, if the current run is an experiment with gold
    questions, those have to be loaded in at this stage.
    """

    def __init__(self, args: DictConfig, tokenizer: PreTrainedTokenizer, split: str) -> None:
        """Initialize the dataset.
        
        No dataloader is created because it is not needed.
        """

        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.data = self.read_data(args.data.get(split).path)

        # default keys for in the data list dicts
        self.x_label = "snippet"
        self.y_label = "y_gold"

        # Don't create a dataloader because it is not needed.

    def read_data(self, data_path_or_name: str) -> Sequence[Any]:
        """Offload reading data to the `pipeline` module.

        Offload reading data to the `pipeline` module because different runs have different data
        that has to be loaded in.
        """
        import contrastive_tldrs.pipeline as pipeline

        run_steps = pipeline.load_steps_to_run(self.args)
        return pipeline.load_data(data_path_or_name, run_steps)


# NOTE: I don't know if the below actually works....
# Creates a DatasetCls type, that has to be a subclass of
# Dataset (which is what the `bound=Dataset` means)
# and Generic[DS] is a subtype of Generic[DatasetCls] if
# DS is a subtype of Dataset (which is what `covariant` means)
# https://mypy.readthedocs.io/en/latest/generics.html#variance-of-generic-types
DatasetCls = TypeVar("DatasetCls", bound=Dataset, covariant=True)


class DatasetDict:
    """Store train, val, and test datasets in one object that can be passed around.
    
    Attributes:
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        test_dataset: The test dataset.
    """

    def __init__(
        self, args: DictConfig, tokenizer: PreTrainedTokenizer, DatasetCls: type[DatasetCls]
    ) -> None:
        """Initialize the DatasetDict.
        
        Loads in the train, val, and test (if uncommented).
        
        Args:
            args (DictConfig): The experiment configuration arguments.
            tokenizer (PreTrainedTokenizer): The Huggingface tokenizer to be used to tokenize the data.
            DatasetCls (type[DatasetCls]): The Dataset class used to load in the datasets.
        """

        try:
            self.train_dataset: Dataset = DatasetCls(args, tokenizer, split="train")
        except FileNotFoundError:
            print(
                "Unable to load training data. "
                "Check to make sure `train` file exists if you are trying to train this."
            )
            self.train_dataset = None  # type: ignore
        self.val_dataset: Dataset = DatasetCls(args, tokenizer, split="val")
        # self.test_dataset = DatasetCls(args, tokenizer, split="test")


def load_dataset(args: DictConfig, tokenizer: PreTrainedTokenizer) -> DatasetDict:
    """Instantiates a DatasetDict with the correct Dataset class based on the args.

    Args:
        args (DictConfig): The experiment configuration arguments.
        tokenizer (PreTrainedTokenizer): The Huggingface tokenizer to be used to tokenize the data.

    Returns:
        DatasetDict containing the train, val (and sometimes test) data for an experiment.
    """
    if args.data.name == "rewrite":
        return DatasetDict(args, tokenizer, RewriteDataset)
    elif args.data.name == "decoder_only":
        return DatasetDict(args, tokenizer, RewriteDatasetDecoderOnly)
    elif args.data.name == "template":
        return DatasetDict(args, tokenizer, ChatTemplateDataset)
    elif args.data.name == "pipeline":
        return DatasetDict(args, tokenizer, PipelineDataset)
    raise ValueError(f"Unable to find dataset with name {args.data.name}")
