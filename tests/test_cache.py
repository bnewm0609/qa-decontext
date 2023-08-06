import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

class TestCache(unittest.TestCase):
    def setUp(self):
        self.using_github_actions = (
            "USING_GITHUB_ACTIONS" in os.environ and os.environ["USING_GITHUB_ACTIONS"] == "true"
        )
        if self.using_github_actions:
            self.skipTest(
                "Skipping test_retrieval because it requires an openai key."
            )

    def test_jsoncache_dir_from_enviro(self):
        with tempfile.TemporaryDirectory() as tempdirname:
            with mock.patch.dict(os.environ, {"DECONTEXT_CACHE_DIR": tempdirname}, clear=True):
                # import here so the new environment variable is used
                import decontext.cache

                importlib.reload(decontext.cache)
                from decontext.cache import JSONCache

                cache = JSONCache.load()
                cache.query("test-key", lambda: "test-val")
                assert (Path(tempdirname) / "jsoncache/cache.json").exists()
                assert (Path(tempdirname) / "jsoncache/cache.json.lock").exists()

    def test_jsoncache_clear(self):
        with tempfile.TemporaryDirectory() as tempdirname_new:
            with mock.patch.dict(os.environ, {"DECONTEXT_CACHE_DIR": tempdirname_new}, clear=True):
                # import/reimport here so the new environment variable is used
                import decontext.cache

                importlib.reload(decontext.cache)
                from decontext.cache import JSONCache

                cache = JSONCache.load()
                cache.query("test-key", lambda: "test-val")
                
                assert len(cache._cache) == 1
                cache.remove_all_unsafe_no_confirm()
                assert len(cache._cache) == 0

    def test_diskcache_dir_from_enviro(self):
        with tempfile.TemporaryDirectory() as tempdirname_new:
            with mock.patch.dict(os.environ, {"DECONTEXT_CACHE_DIR": tempdirname_new}, clear=True):
                # import/reimport here so the new environment variable is used
                import decontext.cache

                importlib.reload(decontext.cache)
                from decontext.cache import DiskCache

                cache = DiskCache.load()
                cache.query("test-key", lambda: "test-val")
                assert (Path(tempdirname_new) / "diskcache/cache.db").exists()

    def test_diskcache_clear(self):
        with tempfile.TemporaryDirectory() as tempdirname_new:
            with mock.patch.dict(os.environ, {"DECONTEXT_CACHE_DIR": tempdirname_new}, clear=True):
                # import/reimport here so the new environment variable is used
                import decontext.cache

                importlib.reload(decontext.cache)
                from decontext.cache import DiskCache

                cache = DiskCache.load()
                cache.query("test-key", lambda: "test-val")
                assert (Path(tempdirname_new) / "diskcache/cache.db").exists()
                
                assert len(cache._cache) == 1
                cache.remove_all_unsafe_no_confirm()
                assert len(cache._cache) == 0

    def test_diskcache_invalidate(self):
        with tempfile.TemporaryDirectory() as tempdirname_new:
            with mock.patch.dict(os.environ, {"DECONTEXT_CACHE_DIR": tempdirname_new}, clear=True):
                # import/reimport here so the new environment variable is used
                import decontext.cache

                importlib.reload(decontext.cache)
                from decontext.cache import DiskCache

                cache = DiskCache.load()
                test_val_1 = cache.query("test-key", lambda: "test-val")
                assert (Path(tempdirname_new) / "diskcache/cache.db").exists()
                
                test_val_2 = cache.query("test-key", lambda: "test-val-2", invalidate=True)
                
                assert test_val_1 != test_val_2
                assert cache.query("test-key", lambda: "test-val-3") == test_val_2