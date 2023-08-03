import os
import importlib
import tempfile
from pathlib import Path


def test_cache_dir_from_enviro():
    old_val_for_decontext_cache_dir = os.environ.get("DECONTEXT_CACHE_DIR")

    with tempfile.TemporaryDirectory() as tempdirname:
        os.environ["DECONTEXT_CACHE_DIR"] = tempdirname
        # import here so the new environment variable is used
        import decontext.cache

        importlib.reload(decontext.cache)
        from decontext.cache import Cache

        cache = Cache.load()
        cache.query("test-key", lambda: "test-val")
        assert (Path(tempdirname) / "jsoncache/cache.json").exists()
        assert (Path(tempdirname) / "jsoncache/cache.json.lock").exists()

    if old_val_for_decontext_cache_dir is None:
        del os.environ["DECONTEXT_CACHE_DIR"]
    else:
        os.environ["DECONTEXT_CACHE_DIR"] = old_val_for_decontext_cache_dir


def test_diskcache_dir_from_enviro():
    old_val_for_decontext_cache_dir = os.environ.get("DECONTEXT_CACHE_DIR")

    with tempfile.TemporaryDirectory() as tempdirname_new:
        os.environ["DECONTEXT_CACHE_DIR"] = tempdirname_new
        # import/reimport here so the new environment variable is used
        import decontext.cache

        importlib.reload(decontext.cache)
        from decontext.cache import DiskCache

        cache = DiskCache.load()
        cache.query("test-key", lambda: "test-val")
        assert (Path(tempdirname_new) / "diskcache/cache.db").exists()

    if old_val_for_decontext_cache_dir is None:
        del os.environ["DECONTEXT_CACHE_DIR"]
    else:
        os.environ["DECONTEXT_CACHE_DIR"] = old_val_for_decontext_cache_dir
