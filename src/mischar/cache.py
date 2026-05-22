"""
Content-addressed caching layer backed by diskcache.

Wraps ``diskcache.Cache`` (SQLite-backed) with per-stage namespacing and
content-addressed key generation. Every cached value is keyed by a hash of its
inputs so that changes to prompts, models, or config automatically invalidate
stale entries.

Cache failures are non-fatal: corruption or disk-full conditions are logged
and the pipeline proceeds without caching for that lookup.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache

from mischar.logging import get_logger

log = get_logger("cache")


class Cache:
    """
    Persistent, content-addressed cache backed by SQLite.

    Keys are namespaced by pipeline stage (``resolve``, ``attribute``, etc.)
    so stages can be cleared independently.  Values are pickled by diskcache.

    Args:
        path: Directory for the diskcache database.
        enabled: If False, all reads return None and writes are no-ops.
            Supports the ``--no-cache`` CLI flag.
    
    '*' forces everything after it to be passed as a keyword argument, so 
    Cache("/path/to/cache", enabled=False) works but Cache("/path/to/cache", False)
    does not. This guards against positional argument mistakes.
    """
    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._path = Path(path)

        if self._enabled:
            # Create the cache directory if it doesn't already exist.
            # diskcache stores its SQLite database and blob files here.
            self._path.mkdir(parents=True, exist_ok=True)
            self._disk_cache = diskcache.Cache(str(self._path))
            log.info("cache_initialized", path=str(self._path))
        else:
            # When disabled (--no-cache flag), we skip creating the database
            # entirely. All reads will return None, all writes will be no-ops.
            self._disk_cache = None
            log.info("cache_disabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, stage: str, key: str) -> Any | None:
        """
        Retrieve a cached value, or None on miss / error.

        Args:
            stage: Pipeline stage name (e.g. ``"resolve"``).
            key: Content-addressed key from :meth:`make_key`.

        Returns:
            The cached value, or None if not found or cache is disabled.
        """
        if not self._enabled:
            return None

        # Prefix the key with the stage name so that different stages
        # can use identical content keys without colliding.
        full_key = self._full_key(stage, key)
        try:
            value = self._disk_cache.get(full_key, default=None)

            if value is not None:
                # Log a truncated key (first 16 hex chars) for readability
                log.debug("cache_hit", stage=stage, key=key[:16])

            return value
        except Exception:
            # Cache corruption, SQLite lock contention, etc. — treat as a
            # miss rather than crashing the pipeline.
            log.warning("cache_read_error", stage=stage, key=key[:16], exc_info=True)

            return None

    def set(self, stage: str, key: str, value: Any) -> None:
        """
        Store a value in the cache.

        Args:
            stage: Pipeline stage name.
            key: Content-addressed key from :meth:`make_key`.
            value: Any picklable Python object.
        """
        if not self._enabled:
            return

        full_key = self._full_key(stage, key)

        try:
            self._disk_cache.set(full_key, value)
            log.debug("cache_set", stage=stage, key=key[:16])
        except Exception:
            # Disk full, permissions error, etc. — log and move on.
            # The pipeline will just redo this work next time.
            log.warning("cache_write_error", stage=stage, key=key[:16], exc_info=True)

    def clear(self, stage: str | None = None) -> None:
        """
        Clear cached entries.

        Args:
            stage: If provided, only entries for that stage are removed.
                If None, the entire cache is cleared.
        """
        if not self._enabled:
            return

        try:
            if stage is None:
                # Wipe everything — useful for a clean-slate rerun.
                self._disk_cache.clear()
                log.info("cache_cleared_all")
            else:
                # Clear only one stage's entries. We iterate all keys and
                # delete those matching the stage prefix. This lets you
                # e.g. clear stale CourtListener results without losing
                # your embedding cache.
                prefix = f"{stage}:"
                keys_to_delete = [
                    k for k in self._disk_cache if isinstance(k, str) and k.startswith(prefix)
                ]

                for k in keys_to_delete:
                    del self._disk_cache[k]

                log.info("cache_cleared_stage", stage=stage, count=len(keys_to_delete))
        except Exception:
            log.warning("cache_clear_error", stage=stage, exc_info=True)

    @staticmethod
    def make_key(*parts: Any) -> str:
        """
        Build a content-addressed cache key by hashing input parts.

        Each part is serialized to a stable string representation, then the
        concatenation is SHA-256 hashed.  For LLM calls, parts should include
        the model name, prompt template version, and input text so that any
        change to these automatically invalidates the cache.

        Args:
            *parts: Hashable components (strings, ints, floats, dicts, lists).
                    '*' means all positional arguments are collected into a tuple, so
                    any number of arguments can be passed in.

        Returns:
            A 64-character hex digest string.

        Example::

            key = Cache.make_key(
                "gemma27b-prompted",     # model name
                "v1.0",                  # prompt version
                passage_text,            # input
                citation.raw_text,       # citation
            )
        """
        hasher = hashlib.sha256()

        for part in parts:
            # Convert each part to a stable string before feeding it into the
            # hash. This ensures that e.g. dict key order doesn't change the
            # resulting hash — see _stable_serialize below.
            serialized = _stable_serialize(part)
            hasher.update(serialized.encode("utf-8"))

        return hasher.hexdigest()

    def close(self) -> None:
        """Close the underlying diskcache database."""
        if self._disk_cache is not None:
            self._disk_cache.close()
            log.info("cache_closed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _full_key(stage: str, key: str) -> str:
        """
        Namespace a key by stage.

        "resolve" + "abc123" becomes "resolve:abc123", ensuring that
        two stages using the same content key don't collide.
        """
        return f"{stage}:{key}"


def _stable_serialize(value: Any) -> str:
    """
    Convert a value to a stable string representation for hashing.

    The key requirement is *determinism*: the same logical value must always
    produce the same string, regardless of insertion order (for dicts) or
    Python internals. Each type is handled explicitly:

    - str: passed through as-is
    - int/float/bool: converted via str()
    - dict/list: JSON-serialized with sorted keys so {"a":1,"b":2} and
      {"b":2,"a":1} hash identically
    - bytes: hashed first (avoids enormous strings for binary data like
      embeddings)
    - anything else: falls back to str() as a best-effort default
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (dict, list)):
        # sort_keys=True is the critical bit — without it, dict ordering
        # could produce different hashes for the same logical content.
        # default=str handles any non-JSON-native types nested inside.
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, bytes):
        # For binary data (e.g. raw embedding vectors), hash the bytes
        # rather than converting to a potentially huge string.
        return hashlib.sha256(value).hexdigest()
    
    return str(value)
