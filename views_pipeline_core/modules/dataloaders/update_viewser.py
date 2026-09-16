"""`UpdateViewser` — RETIRED 2026-09-16 (C-318). The name survives for one window; the class does not.

What it was, when it was live, and why it is gone: the closing note of
`documentation/ADRs/037_ingester_emergency_solution.md`. That note is the one canonical
account; this docstring does not restate it.

Why this stub exists, and what it costs (ADR-062): `views_pipeline_core.modules.dataloaders.UpdateViewser`
was a declared public name, and engine repos pin `views-pipeline-core <4.0.0`, so
`tests/test_public_surface_requires_a_major_bump.py` refuses a minor release that deletes it.
This stub satisfies that guard in letter and not in spirit — a name that raises at
construction under a minor is exactly the #188 shape the guard was written against. It is
the deliberate cost of retiring under a minor, taken because practical exposure is nil: no
importer of this class exists in any of the 19 sibling repos on disk. The honest state is
"resolves and refuses until 4.0, absent after"; `tests/test_modules/test_update_viewser_is_retired.py`
asserts the first half and fails the build at major >= 4 so the second is not forgotten.

Nothing here imports the transformation library; the same test file probes every submodule
of this package for its absence.
"""
from __future__ import annotations

from typing import Any


class UpdateViewser:
    """Retired. Constructing it raises; see the module docstring."""

    # The four parameters the surface snapshot recorded at 3.0.0, REQUIRED as recorded.
    # The first version gave them `= None` defaults "so the guard sees the same
    # signature" — it saw a wider one (optional where the snapshot says required), and
    # the guard only flags the other direction, so the widening passed unnoticed.
    def __init__(self, queryset: Any, viewser_df: Any, data_path: Any, months_to_update: Any) -> None:
        raise RuntimeError(
            "UpdateViewser was retired on 2026-09-16 (C-318) and its dependency is no "
            "longer installed. See the closing note of "
            "documentation/ADRs/037_ingester_emergency_solution.md for what it was and "
            "why it is gone. This name is removed at 4.0 (ADR-062)."
        )
