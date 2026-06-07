"""Shared constants for the data layer.

These constants are the single source of truth for conventions shared
across pipeline-core and engine repos.  Engine repos should import
from here rather than hardcoding their own variants.
"""

CACHE_SOURCES = frozenset({"viewser", "datafactory", "synthetic"})

CACHE_FILENAME_TEMPLATE = "{partition}_{source}_df{ext}"
