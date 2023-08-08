import json
import os
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional

from diskcache import Index
from filelock import FileLock

# Use an IntEnum because it defines __eq__ that compares values rather than reference ids
# This is important because if the cache_module is reloaded, the reference id will change,
# cache_state is CacheState.XXXX will be false, even if cache_state.value == CacheState.XXXX
# By using the comparison operator == and and an IntEnum, this issue is avoided.
CacheState = IntEnum("CacheState", ["NO_CACHE", "INVALIDATE", "NORMAL", "ENFORCE_CACHE"])


class Cache:
    def __init__(self, cache: dict, cache_dir: str, enforce_cached: bool = False) -> None:
        # CACHE_DIR = os.environ.get("DECONTEXT_CACHE_DIR", f"{Path.home()}/.cache/decontext")
        # OPENAI_CACHE_DIR = f"{CACHE_DIR}/jsoncache/"
        # OPENAI_DISKCACHE_DIR = f"{CACHE_DIR}/diskcache/"
        raise NotImplementedError()

    @classmethod
    def get_default_cache_dir(cls) -> str:
        return os.environ.get("DECONTEXT_CACHE_DIR", f"{Path.home()}/.cache/decontext")

    def query(self, key: str, fn: Callable[[], Any], invalidate: bool) -> Any:
        """Query the cache for a key, and if it doesn't exist, run the function to get the value.

        Args:
            key (str): the key to query the cache for.
            fn (Callable[[], Any]): the function to run to get the value if the key is not in the cache.

        Returns:
            The value of the key in the cache.
        """
        raise NotImplementedError()

    def remove(self, key: str):
        raise NotImplementedError()

    def remove_all_unsafe_no_confirm(self):
        raise NotImplementedError()


class DiskCache(Cache):
    # def __init__(self, cache: dict, cache_dir: str, enforce_cached: bool = False) -> None:
    def __init__(self, cache: dict, cache_dir: str) -> None:
        """Initialize the Cache.

        Cache should be loaded in using Cache.load rather than the constructor.

        Args:
            cache (dict): the key-value store that makes up the cache.
            cache_dir (str): the directory where the cache is saved.
            enforce_cached (bool): If True, an error is thrown if the item queried is not in the Cache.
        """

        self._cache = cache
        self.cache_dir = cache_dir
        self.default_cache_state = CacheState.NORMAL
        # self.enforce_cached = enforce_cached  # True

    @classmethod
    def load(cls, cache_dir: Optional[str] = None) -> "DiskCache":
        # def load(cls, cache_dir=OPENAI_DISKCACHE_DIR, enforce_cached: bool = False) -> "DiskCache":
        """Return an instance of a cache at the location pointed to by cache_dir.

        If `enforce_cache` is True, an error is thrown if the queried result is not in the cache.

        Args:
            cache_dir (str): the directory where the cache is saved.
            enforce_cached (bool): If True, an error is thrown if the item queried is not in the Cache.

        Returns:
            The cache object.
        """
        if cache_dir is None:
            cache_dir = os.path.join(cls.get_default_cache_dir(), "diskcache")

        cache = Index(cache_dir)
        return cls(cache, cache_dir)

    def query(self, key: str, fn: Callable[[], Any], cache_state: Optional[CacheState] = None) -> Any:
        """Query the cache and call the function upon a cache miss.

        If the key is not in the Cache, call the function and store the result of the function call in the cache
        at the current key.

        Args:
            key (str): The key to the cache.
            fn (Callable): The function to call upon a cache miss.

        Returns:
            The value stored at the key or the result of calling the function.
        """

        if cache_state is None:
            cache_state = self.default_cache_state

        # If not using a cache, just call the function
        if cache_state == CacheState.NO_CACHE:
            return fn()
        # Otherwise, return what's in the cache unless we're invalidating the key
        elif key in self._cache and cache_state != CacheState.INVALIDATE:
            print("Found example in cache")
            return self._cache[key]
        elif cache_state == CacheState.ENFORCE_CACHE:
            raise ValueError(
                f"Cache.enforce_cache is True, but the following key was not found in the cache! Key: `${key}`"
            )
        else:
            self._cache[key] = fn()
            return self._cache[key]

    def remove(self, key: str) -> Any:
        del self._cache[key]

    def remove_all_unsafe_no_confirm(self):
        self._cache.clear()


class JSONCache(Cache):
    """Cache for storing results of calls to models bethind APIs.

    This is a singleton object and should be initialized by calling `Cache.load`.

    Attributes:
        _cache: A dict representing the actual cache.
        cache_dir: the directory where the cache is saved.
        default_cache_state: The default cache state to use when querying the cache.
        lock: A FileLock to prevent concurrent edits to the cache file.
    """

    def __init__(self, cache: dict, cache_dir: str) -> None:
        """Initialize the Cache.

        Cache should be loaded in using Cache.load rather than the constructor.

        Args:
            cache (dict): the key-value store that makes up the cache.
            cache_dir (str): the directory where the cache is saved.
        """

        self._cache = cache
        self.cache_dir = cache_dir
        cache_filelock_path = Path(cache_dir) / "cache.json.lock"
        self.lock = FileLock(cache_filelock_path)
        self.default_cache_state = CacheState.NORMAL

    @classmethod
    def load(cls, cache_dir: Optional[str] = None) -> "Cache":
        """Return an instance of a cache at the location pointed to by cache_dir.

        If `enforce_cache` is True, an error is thrown if the queried result is not in the cache.

        Args:
            cache_dir (str): the directory where the cache is saved.

        Returns:
            The cache object.
        """

        if cache_dir is None:
            cache_dir = os.path.join(cls.get_default_cache_dir(), "jsoncache")

        cache_path = Path(cache_dir) / "cache.json"

        if cache_path.exists():
            # Add filelock to avoid multiple edits to the cache at once.
            cache_filelock_path = Path(cache_dir) / "cache.json.lock"
            lock = FileLock(cache_filelock_path)
            with lock:
                with open(cache_path) as f:
                    return cls(json.load(f), cache_dir)
        return cls({}, cache_dir)

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

    def query(self, key: str, fn: Callable[[], Any], cache_state: Optional[CacheState] = None) -> Any:
        """Query the cache and call the function upon a cache miss.

        If the key is not in the Cache, call the function and store the result of the function call in the cache
        at the current key.

        Args:
            key (str): The key to the cache.
            fn (Callable): The function to call upon a cache miss.
            cache_state: The cache state to use when querying the cache.
                NO_CACHE: Don't use the cache at all, just call the function.
                INVALIDATE: Call fn and cache the result, overwriting what's in the cache.
                NORMAL: Return what's in the cache if it exists, otherwise call fn and cache the result.
                ENFORCE_CACHE: Throw an error if the key is not in the cache.

        Returns:
            The value stored at the key or the result of calling the function.

        Raises:
            ValueError if cache_state is ENFORCE_CACHE and the key is not in the cache or if cache_state is not
            provided and the default_cache_state is ENFORCE_CACHE and the key is not in the cache.
        """
        if cache_state is None:
            cache_state = self.default_cache_state

        if cache_state == CacheState.NO_CACHE:
            return fn()
        elif key in self._cache and cache_state != CacheState.INVALIDATE:
            print("Found example in cache")
            return self._cache[key]
        elif cache_state == CacheState.ENFORCE_CACHE:
            raise ValueError(
                f"Cache.enforce_cache is True, but the following key was not found in the cache! Key: `${key}`"
            )
        else:
            self._cache[key] = fn()
            self.save()
            return self._cache[key]

    def remove_all_unsafe_no_confirm(self):
        self._cache.clear()
        self.save()
