import json
from pathlib import Path

from diskcache import Index
from filelock import FileLock


OPENAI_CACHE_DIR = f"{Path.home()}/nfs/.cache/openai/"
OPENAI_DISKCACHE_DIR = f"{Path.home()}/nfs/.cache/openai_diskcache/"


class DiskCache:
    def __init__(
        self, cache: dict, cache_dir: str, enforce_cached: bool = False
    ) -> None:
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

    @classmethod
    def load(
        cls, cache_dir=OPENAI_DISKCACHE_DIR, enforce_cached: bool = False
    ) -> "DiskCache":
        """Return an instance of a cache at the location pointed to by cache_dir.

        If `enforce_cache` is True, an error is thrown if the queried result is not in the cache.

        Args:
            cache_dir (str): the directory where the cache is saved.
            enforce_cached (bool): If True, an error is thrown if the item queried is not in the Cache.

        Returns:
            The cache object.
        """

        cache = Index(cache_dir)
        return cls(cache, cache_dir, enforce_cached)

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
                # self.save()
                return self._cache[key]
            else:
                raise ValueError(
                    f"Cache.enforce_cache is True, but the following key was not found in the cache! Key: `${key}`"
                )


class Cache:
    """Cache for storing results of calls to models bethind APIs.

    This is a singleton object and should be initialized by calling `Cache.load`.

    Attributes:
        _cache: A dict representing the actual cache.
        cache_dir: the directory where the cache is saved.
        enforce_cached: If True, an error is thrown if the item queried is not in the Cache.
        lock: A FileLock to prevent concurrent edits to the cache file.
    """

    def __init__(
        self, cache: dict, cache_dir: str, enforce_cached: bool = False
    ) -> None:
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
    def load(
        cls, cache_dir=OPENAI_CACHE_DIR, enforce_cached: bool = False
    ) -> "Cache":
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
