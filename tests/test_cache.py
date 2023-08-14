import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from decontext.cache import DiskCache, JSONCache, CacheState


class TestCache(unittest.TestCase):
    def setUp(self):
        self.using_github_actions = (
            "USING_GITHUB_ACTIONS" in os.environ and os.environ["USING_GITHUB_ACTIONS"] == "true"
        )
        if self.using_github_actions:
            self.skipTest("Skipping test_retrieval because it requires an openai key.")

    def cache_invalidate_helper(self, cache, tempdirname_new, CacheState):
        # test default
        test_val_1 = cache.query("test-key", lambda: "test-val")
        assert "test-key" in cache._cache

        # test invalidating the cache
        test_val_2 = cache.query("test-key", lambda: "test-val-2", cache_state=CacheState.INVALIDATE)

        assert test_val_1 != test_val_2
        assert cache.query("test-key", lambda: "test-val-3") == test_val_2

        # test not storing in the cache
        test_val_not_cached = cache.query(
            "test-key-no-cache", lambda: "test-val-4", cache_state=CacheState.NO_CACHE
        )
        assert "test-key-no-cache" not in cache._cache

        # test enforce cache
        with self.assertRaises(ValueError):
            test_val_enforce_cache = cache.query(
                "test-key-not-in-cache", lambda: "test-val-5", cache_state=CacheState.ENFORCE_CACHE
            )

        test_val_enforce_cache = cache.query(
            "test-key", lambda: "test-val-6", cache_state=CacheState.ENFORCE_CACHE
        )
        assert test_val_enforce_cache == test_val_2

        # test changing default cache_state
        cache.default_cache_state = CacheState.NO_CACHE
        test_val_new_default = cache.query("test-key", lambda: "test-val-7")
        assert test_val_new_default == "test-val-7"

    def test_jsoncache_dir_from_enviro(self):
        with tempfile.TemporaryDirectory() as tempdirname:
            with mock.patch.dict(os.environ, {"DECONTEXT_CACHE_DIR": tempdirname}, clear=True):
                cache = JSONCache.load()
                cache.query("test-key", lambda: "test-val")
                assert (Path(tempdirname) / "jsoncache/cache.json").exists()
                assert (Path(tempdirname) / "jsoncache/cache.json.lock").exists()

    def test_jsoncache_clear(self):
        with tempfile.TemporaryDirectory() as tempdirname_new:
            with mock.patch.dict(os.environ, {"DECONTEXT_CACHE_DIR": tempdirname_new}, clear=True):
                cache = JSONCache.load()
                cache.query("test-key", lambda: "test-val")

                assert len(cache._cache) == 1
                cache.remove_all_unsafe_no_confirm()
                assert len(cache._cache) == 0

    def test_jsoncache_invalidate(self):
        with tempfile.TemporaryDirectory() as tempdirname_new:
            with mock.patch.dict(os.environ, {"DECONTEXT_CACHE_DIR": tempdirname_new}, clear=True):
                cache = JSONCache.load()
                self.cache_invalidate_helper(cache, tempdirname_new, CacheState)

    def test_diskcache_dir_from_enviro(self):
        with tempfile.TemporaryDirectory() as tempdirname_new:
            with mock.patch.dict(os.environ, {"DECONTEXT_CACHE_DIR": tempdirname_new}, clear=True):
                cache = DiskCache.load()
                cache.query("test-key", lambda: "test-val")
                assert (Path(tempdirname_new) / "diskcache/cache.db").exists()

    def test_diskcache_clear(self):
        with tempfile.TemporaryDirectory() as tempdirname_new:
            with mock.patch.dict(os.environ, {"DECONTEXT_CACHE_DIR": tempdirname_new}, clear=True):
                cache = DiskCache.load()
                cache.query("test-key", lambda: "test-val")
                assert (Path(tempdirname_new) / "diskcache/cache.db").exists()

                assert len(cache._cache) == 1
                cache.remove_all_unsafe_no_confirm()
                assert len(cache._cache) == 0

    def test_diskcache_invalidate(self):
        with tempfile.TemporaryDirectory() as tempdirname_new:
            with mock.patch.dict(os.environ, {"DECONTEXT_CACHE_DIR": tempdirname_new}, clear=True):
                cache = DiskCache.load()

                self.cache_invalidate_helper(cache, tempdirname_new, CacheState)
