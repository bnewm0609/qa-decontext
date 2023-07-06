"""Defines utility functions for manipulating configs"""
import base64
import hashlib
import json
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union, cast

import anthropic
from filelock import FileLock
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from contrastive_tldrs.data import Dataset


OPENAI_CHAT_MODEL_NAMES = {"gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4", "gpt-4-32k"}


def convert_paths_to_absolute(config: DictConfig):
    """
    Turn any path in the passed config to an absolute path by recursing down the tree defined by
    the config yaml files by modifying the passed config in-place.
    """
    to_resolve = []
    for key in config:
        if isinstance(config[key], DictConfig):
            to_resolve.append(config[key])  # check if there's anything to resolve in the sub-config
        elif "path" in key:
            config[key] = to_absolute_path(config[key])
    for sub_config in to_resolve:
        convert_paths_to_absolute(sub_config)


def get_last_checkpoint_path(
    config: DictConfig, key: str = "val_loss", best: str = "min"
) -> Optional[Path]:
    """Load the last checkpoint at the path specified by the config's results_path attribute.

    Assume that the checkpoints are saved with pytorch_lightning, so they have the format:
    `ckpt-{key1}={value1}-{key2}={value2}.ckpt`

    1. Convert these to lists of key values
        e.g. ["ckpt", "{key1}={value1}", "{key2}={value2}"]
    2. Filter out any that don't have "="
        e.g. ["{key1}={value1}", "{key2}={value2}"]
    3. Convert this to a dictionary
        e.g. [{key1: value1, key2: value2}]

    The strategy for determining the "last" checkpoint is using the key passed.
    If the key is not available, use the val_loss.
    If val_loss is unavailable, should use step, and if step is unavailable, it should use epoch.

    Return the last checkpoint path if there is one otherwise None
    """
    checkpoint_dir = Path(config.results_path) / "checkpoints"
    checkpoint_files: Sequence[Path] = list(checkpoint_dir.glob("*.ckpt"))

    if not checkpoint_files:
        # Change to logging warning
        print(
            "WARNING: NO TRAINED CHECKPOINT FOUND. PREDICTIONS WILL BE MADE WITH THE PRE-TRAINED MODEL."
        )
        return None

    ckpt_kv_pairs: list[dict[str, Union[str, int]]] = []
    for idx, ckpt_filename in enumerate(checkpoint_files):
        kv_pairs = ckpt_filename.stem.split("-")
        kv_pairs = filter(lambda kv_pair: "=" in kv_pair, kv_pairs)
        kv_pairs = map(lambda kv_pair: kv_pair.split("="), kv_pairs)
        kv_pairs = dict(kv_pairs) | {"idx": idx}
        ckpt_kv_pairs.append(kv_pairs)

    # first try sorting by key
    best_ckpt_idx: Union[int, str]
    if all([key in kv_pair for kv_pair in ckpt_kv_pairs]):
        # extract val
        if best == "min":
            sorted_kv_pairs = sorted(ckpt_kv_pairs, key=lambda kv_pair: float(kv_pair[key]))
        elif best == "max":
            sorted_kv_pairs = sorted(ckpt_kv_pairs, key=lambda kv_pair: -float(kv_pair[key]))
        else:
            raise ValueError("Argument `best` must be one of 'min' or 'max'")
        best_ckpt_idx = sorted_kv_pairs[0]["idx"]
    elif all(["val_loss" in kv_pair for kv_pair in ckpt_kv_pairs]):
        # extract val
        sorted_kv_pairs = sorted(ckpt_kv_pairs, key=lambda kv_pair: float(kv_pair["val_loss"]))
        best_ckpt_idx = sorted_kv_pairs[0]["idx"]
    elif all(["step" in kv_pair for kv_pair in ckpt_kv_pairs]):
        sorted_kv_pairs = sorted(
            ckpt_kv_pairs, key=lambda kv_pair: int(kv_pair["step"]), reverse=True
        )
        best_ckpt_idx = sorted_kv_pairs[0]["idx"]
    elif all(["epoch" in kv_pair for kv_pair in ckpt_kv_pairs]):
        sorted_kv_pairs = sorted(
            ckpt_kv_pairs, key=lambda kv_pair: int(kv_pair["epoch"]), reverse=True
        )
        best_ckpt_idx = int(sorted_kv_pairs[0]["idx"])
    else:
        # Give up and return the file that has been modified least recently
        modification_times = [
            {"mod_time": f.stat().st_mtime, "idx": i} for i, f in enumerate(checkpoint_files)
        ]
        sorted_kv_pairs = sorted(
            modification_times, key=lambda kv_pair: kv_pair["mod_time"], reverse=True  # type: ignore
        )
        best_ckpt_idx = sorted_kv_pairs[0]["idx"]

    best_ckpt_idx = cast(int, best_ckpt_idx)
    return checkpoint_files[best_ckpt_idx]


def get_prediction_save_dir(args: DictConfig, split: str):
    """Given a prediction split ("val" or "test"), creates the directory for the predictions.

    Args:
        args: Experiment configuration args.
        split: The split of the data "val" or "test".
    Returns:
        The save directory based on the split.
    """

    assert split in ["val", "test"], "`split` must be 'val' or 'test'"
    save_dir = Path(args.results_path)
    if split == "val":
        save_dir = save_dir / (args.data.val._id + "-" + args.generation._id)
    else:
        save_dir = save_dir / (args.data.test._id + "-" + args.generation._id)

    try:
        save_dir.mkdir()
    except FileExistsError:
        print(f"WARNING: OVERWRITING PREDICTIONS IN {save_dir}")
        save_dir.mkdir(exist_ok=True)

    return save_dir


# From https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def run_subprocess(cmd: list[str]) -> Iterator[str]:
    """Run cmd and yields output from stdout one line a time."""

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    if popen.stdout is None:
        # We can't read anything if stdout isn't defined. This check also makes mypy happy.
        return
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def hash_strs(*strs, lim=10) -> str:
    """Hash a list of strs.
    
    Args:
        strs (list[str]): the strings to hash.
        lim (int): how many characters the output string should have.

    Returns:
        A hash `lim` characters long.
    """
    if isinstance(strs[0], int):
        lim, *str_tup = strs
        strs = tuple(str_tup)
    inpt = "".join([f"s{i}:{str(x)}" for i, x in enumerate(strs)])
    digest = hashlib.md5(inpt.encode("UTF-8")).digest()
    return base64.b64encode(digest).decode("UTF-8")[:lim]


def get_openai_price_estimate(
    model_name: str,
    dataset: Dataset = None,
    # tokens_used: int = None,
    prompt_tokens: int = None,
    completion_tokens: int = None,
    max_gen_len: int = 0,
) -> float:
    """Use the latest prices to estimate the price of calling the given model on the passed dataset.

    Note: Depending on the parameters passed, it might only calculate the cost of the prompt, not the generation.

    Args:
        model_name: which model - determines the price.
        dataset: dataset - determines the number of examples and price per example.
        prompt_tokens: Number of tokens in the prompt. If specified, it's used to calculate the price instead
            of the dataset. It can be passed along with other arguments.
        completion_tokens: Number of tokens in the completion. This is separate from `prompt_tokens` because
            some OpenAI models charge differently for tokens in the prompt vs. those in the completion.
        max_gen_len: if specified, is added to the number of tokens estimated using the
                    dataset. Is not used if only prompt_tokens/completion_tokens are specified but not dataset.

    Returns:
        A float with the price of calling the model on the dataset.
    """
    price_per_1k_token_map = {
        "ada": 0.0004,
        "babbage": 0.0005,
        "curie": 0.002,
        "davinci": 0.02,
        "turbo": 0.002,
        "gpt-4-32k": 0.12,  # order is important here
        "gpt-4": 0.06,
        "claude": 0.03268,
    }

    price_per_1k_prompt_token_map = price_per_1k_token_map | {
        "gpt-4-32k": 0.06,
        "gpt-4": 0.03,
        "claude": 0.01102,
    }

    tokens_are_provided = prompt_tokens is not None and completion_tokens is not None

    price_per_1k_sample = -1.0
    price_per_1k_prompt = -1.0
    for name, candidate_price_per_1k in price_per_1k_token_map.items():
        if name in model_name:
            price_per_1k_sample = candidate_price_per_1k
            price_per_1k_prompt = price_per_1k_prompt_token_map[name]

    if dataset is not None:
        total_tokens = 0  # if not tokens_are_provided else tokens_used
        for example in dataset:
            if model_name in OPENAI_CHAT_MODEL_NAMES:
                total_tokens += sum(
                    [
                        len(dataset.tokenizer(message.content))
                        for message in example[dataset.x_label]
                    ]
                )
            elif "claude" in model_name:
                total_tokens += anthropic.count_tokens(example[dataset.x_label])
            else:
                total_tokens += len(dataset.tokenizer(example[dataset.x_label]).input_ids)

            # if tokens_used isn't provided, then estimate the number of tokens
            # used by assuming that max_gen_len tokens are always geneated.
            # Note: this is an overestimate
            if not tokens_are_provided:
                total_tokens += max_gen_len
        print("Total tokens:", total_tokens)
        print("Total tokens (-generation):", total_tokens - (max_gen_len * len(dataset)))

        total_price_per_1k = (total_tokens - (max_gen_len * len(dataset))) * price_per_1k_sample
        total_price_per_1k += (max_gen_len * len(dataset)) * price_per_1k_prompt
        return max(-1, total_price_per_1k / 1_000)
    elif dataset is None and tokens_are_provided:
        # If dataset isn't provided then we can't differentiate between the number of prompt
        # tokens and not prompt tokens. This will OVERESTIMATE GPT-4 cost because we use the
        # sampling cost which is more expensive for GPT-4
        prompt_price_per_1k = price_per_1k_prompt * (  # type: ignore
            prompt_tokens if prompt_tokens is not None else 0  # type: ignore
        )  # type: ignore
        completion_price_per_1k = price_per_1k_sample * (  # type: ignore
            prompt_tokens if completion_tokens is not None else 0  # type: ignore
        )  # type: ignore
        return max(-1, (prompt_price_per_1k + completion_price_per_1k) / 1_000)
    else:
        raise ValueError("One of `dataset` or `tokens_used` must be provided")


@dataclass
class Predictions:
    """Class for storing predictions returned from models."""
    predictions: Sequence[str]
    metadata: Optional[list[dict[str, Any]]] = None


class RunMode(str, Enum):
    """Enum for tracking the mode that the experiment was called with.

    Inherits from str mixin so we can do RunMode.TRAIN == "train".
    """

    TRAIN = "train"
    PREDICT = "predict"
    EVALUATE = "evaluate"


OPENAI_CACHE_DIR = "/home/benjaminn/nfs/.cache/openai/"


class Cache:
    """Cache for storing results of calls to models bethind APIs.

    This is a singleton object and should be initialized by calling `Cache.load`.
    
    Attributes:
        _cache: A dict representing the actual cache.
        cache_dir: the directory where the cache is saved.
        enforce_cached: If True, an error is thrown if the item queried is not in the Cache.
        lock: A FileLock to prevent concurrent edits to the cache file.
    """

    def __init__(self, cache: dict, cache_dir: str, enforce_cached: bool = False) -> None:
        """Initialize the Cache.
        
        Cache should be loaded in using Cache.load rather than the constructor.
        
        Args:
            cache (dict): the key-value store that makes up the cache.
            cache_dir (str): the directory where the cache is saved.
            enforce_cached (bool): If True, an error is thrown if the item queried is not in the Cache.
        """

        self._cache = cache
        self.cache_dir = cache_dir
        self.enforce_cached = enforce_cached  # True
        cache_filelock_path = Path(cache_dir) / "cache.json.lock"
        self.lock = FileLock(cache_filelock_path)

    @classmethod
    def load(cls, cache_dir=OPENAI_CACHE_DIR, enforce_cached: bool = False) -> "Cache":
        """Return an instance of a cache at the location pointed to by cache_dir.
        
        If `enforce_cache` is True, an error is thrown if the queried result is not in the cache.

        Args:
            cache_dir (str): the directory where the cache is saved.
            enforce_cached (bool): If True, an error is thrown if the item queried is not in the Cache.

        Returns:
            The cache object.
        """

        cache_path = Path(cache_dir) / "cache.json"

        if cache_path.exists():
            # Add filelock to avoid multiple edits to the cache at once.
            cache_filelock_path = Path(cache_dir) / "cache.json.lock"
            lock = FileLock(cache_filelock_path)
            with lock:
                with open(cache_path) as f:
                    return cls(json.load(f), cache_dir)
        return cls({}, cache_dir, enforce_cached=enforce_cached)

    def save(self) -> None:
        """Save the cache to the cache path.
        
        Do not allow for interruptions because these corrupt the cache, making it impossible to load
        in subsequent runs.
        """
        # TODO: change this so we only add rather than rewrite the entire cache every time
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / "cache.json"
        try:
            with self.lock:
                with open(cache_path, "w") as f:
                    json.dump(self._cache, f)
        except KeyboardInterrupt:
            print(
                "\n\n\n-------------------\n"
                "KEYBOARD INTERURPT DETECTED. CLEANING UP......"
                "(DO NOT PRESS KEYBOARD INTERRUPT AGAIN)"
            )
            with open(cache_path, "w") as f:
                json.dump(self._cache, f)
            import sys

            sys.exit(1)

    def query(self, key, fn):
        """Query the cache and call the function upon a cache miss.

        If the key is not in the Cache, call the function and store the result of the function call in the cache
        at the current key.

        Args:
            key (str): The key to the cache.
            fn (Callable): The function to call upon a cache miss.
        
        Returns:
            The value stored at the key or the result of calling the function.
        """
        if key in self._cache:
            print("Found example in cache")
            return self._cache[key]
        else:
            if not self.enforce_cached:
                self._cache[key] = fn()
                self.save()
                return self._cache[key]
            else:
                raise ValueError(
                    f"Cache.enforce_cache is True, but the following key was not found in the cache! Key: `${key}`"
                )


class RangeList:
    """Data structure for keeping track of ranges in a sorted list that prevents overlaps.

    Used in `scripts/propagate_data_changes.py`.
    """

    def __init__(self):
        self.ranges = []

    def get_merged_bounds(self, range1, range2):
        range1, range2 = sorted([range1, range2])
        range1_start, range1_end = range1
        range2_start, range2_end = range2

        if range1_end < range2_start:
            return [range1, range2]

        return [(min(range1_start, range2_start), max(range1_end, range2_end))]

    def add(self, start, end):
        if end < start:
            raise ValueError("`end` must be greater than or equal to `start`")

        new_ranges = []
        for i in range(len(self.ranges)):
            bounds = self.get_merged_bounds((start, end), self.ranges[i])
            if len(bounds) == 2:
                # the two ranges are disjoint and the first range is lower
                new_ranges.append(bounds[0])
                start, end = bounds[1]
            else:  # here len(bounds) == 1
                # the two ranges overlap
                start, end = bounds[0]
        new_ranges.append((start, end))
        self.ranges = new_ranges

    def __repr__(self):
        return f"RangeList({self.ranges})"
