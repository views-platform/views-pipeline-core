"""`UpdateViewser` — RETIRED 2026-09-16. The name survives for one window; the class does not.

What it was: ADR-037's emergency fallback for the February 2025 ingester outage. When
UCDP/ACLED months came back from viewser as zeros, this replayed a queryset's transformation
chain over hand-supplied GED/ACLED files so a cached VIEWSER frame could be patched without a
full re-fetch. It was the only reason this package depended on the transformation library.

Why it is gone: it was live for six weeks (2025-10-09 to 2025-11-24), switched off with no
recorded reason, never configured on any machine, and its own ADR listed "letting the
fallback system become permanent" as a risk. If the ingester fails again the fallback has to
be rebuilt on the frames-native path — the retired file's own banner said so.

Why this stub exists: `views_pipeline_core.modules.dataloaders.UpdateViewser` was a declared
public name, and engine repos pin `views-pipeline-core <4.0.0`, so a minor release that
deletes it reaches every consumer unchecked — `tests/test_public_surface_requires_a_major_bump.py`
refuses exactly that. So for one window the name resolves and REFUSES on construction, the
same way `--update_viewser` now refuses at the data-fetch step. Both go at 4.0.

Nothing here imports the transformation library. That absence is pinned by
`tests/test_modules/test_update_viewser_is_retired.py`.
"""
from __future__ import annotations

from typing import Any


class UpdateViewser:
    """Retired. Constructing it raises; see the module docstring."""

    # The four parameters the snapshot recorded at 3.0.0 — kept so the surface guard sees
    # the same signature, not a narrowed one. None of them is read.
    def __init__(
        self,
        queryset: Any = None,
        viewser_df: Any = None,
        data_path: Any = None,
        months_to_update: Any = None,
    ) -> None:
        raise RuntimeError(
            "UpdateViewser was retired on 2026-09-16. It was the ADR-037 emergency fallback "
            "for the 2025 ingester outage, dead at runtime since 2025-11-24 and never "
            "configured. Its dependency, the transformation library, is no longer "
            "installed with this package. If the ingester fails again the fallback must be "
            "rebuilt on the frames-native path — see "
            "documentation/ADRs/037_ingester_emergency_solution.md. This name is removed at 4.0."
        )
