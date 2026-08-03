# Class Intent Contract: CoreFrameSniffer

**Module:** `views_pipeline_core/modules/validation/core_frame_sniffer.py`
**Pattern:** ADR-041 (Sniffer) — state-bearing audit class, single `sniff_*` entry point
**Introduced:** epic #285 S3 (#288, PR #295); dispatch live since S5 (#290)

## 1. Purpose

Audit-only validator for `views_frames.FeatureFrame`s loaded as model input —
the frames counterpart of `CoreDataSniffer`. Runs after a frame is fetched from
datafactory or read from the directory cache, before any model sees it.

## 2. Non-Goals (Explicit Exclusions)

- No domain validation (value ranges, feature completeness, temporal
  continuity of *values*) — engine-sniffer territory per the #142 boundary.
- No NaN policy — the engine boundary owns it (views-baseline fails loud by
  design); datafactory zero-fills pre-coverage months per its ADR-047 contract.
- No sample-count enforcement — the datafactory contract declares S=1 for
  observed data, but that is a producer-side property this repo does not own
  (considered and deferred at #288; revisit if the contract hardens).
- No mutation, ever: read-only throughout.

## 3. Responsibilities and Guarantees

1. **Non-emptiness** — rows and features both present (shared definition:
   `data/frame_invariants.assert_frame_nonempty`, the same predicate the frame
   cache enforces on write and read).
2. **Spatial level** — the frame's `index.level` equals the model's declared
   level (value equality; `views_frames.SpatialLevel` is the single level
   vocabulary across all sniffers).
3. **Partition compatibility** — COMPLETE month coverage: every month in the
   expected range present, none outside. Expected bounds come from
   `data/partitions.resolve_month_range` — the same pure rule the fetch path
   uses (C-209: one implementation, producer and auditors can never drift).

Invalid partition or level fail loud at construction, not at sniff time.

## 4. Inputs and Assumptions

- `partition_dict: Dict` — `{"train": (first, last), "test": (first, last)}`.
- `partition: str` — `calibration` | `validation` | `forecasting` (anything
  else: `ValueError` at construction).
- `level: str` — `pgm` | `cm`; required, no permissive mode
  (unknown: `NotImplementedError` at construction).
- `override_month: Optional[int]` — forecasting only; deliberately ignored for
  other partitions (legacy contract — managers pass it unconditionally).
- The audited object is a `views_frames.FeatureFrame` (index exposes
  `time`/`unit`/`level`).

## 5. Outputs and Side Effects

- Returns `None`; on a clean pass emits
  `logger.info("CoreFrameSniffer: Loaded frame audited (partition=..., level=...)")`.
- No other side effects: construction is silent (the shared month-range rule is
  pure; the operational override warning belongs to the fetch layer).

## 6. Failure Modes and Loudness

- `ValueError` (construction) — unknown partition string.
- `NotImplementedError` (construction) — unsupported level.
- `ValueError` (sniff) — empty frame; level mismatch; incomplete month
  coverage (message names up to five missing months and the total count).
- Fail Loud and Proud: no bool returns, no warnings-instead-of-errors.

## 7. Boundaries and Interactions

- Invoked by `feature_frame_path.fetch_feature_frame` on BOTH fresh fetches
  (before caching — an audit-failing frame is never persisted) and cache hits
  (a cache hit is not a validation bypass).
- Layered defense with `frame_cache` on emptiness: same shared predicate, two
  boundaries (cache write/read, model hand-off).
- Import-light (no pandas/viewser/ingester3) — pinned by a subprocess probe
  test; safe on the pandas-free path by construction.

## 8. Examples of Correct Usage

```python
sniffer = CoreFrameSniffer(
    partition_dict={"train": (121, 444), "test": (445, 492)},
    partition="calibration",
    level="pgm",
)
sniffer.sniff_loaded_frame(frame)  # raises on violation; logs on success
```

## 9. Examples of Incorrect Usage

```python
CoreFrameSniffer(pd_dict, "production", level="pgm")   # unknown partition: raises
CoreFrameSniffer(pd_dict, "calibration", level="county")  # unknown level: raises
sniffer.sniff_loaded_frame(df)  # pandas DataFrame: wrong type — use CoreDataSniffer
```

## 10. Test Alignment

`tests/test_modules/test_core_frame_sniffer.py`: pass cases (all three
partitions, override, the vendored #162 contract fixture), boundary violations
(parametrized), interior-hole rejection, empty frame, wrong/unsupported level,
invalid partition, level-vocabulary parity across sniffers, read-only proof,
import-weight probe.

## 11. Evolution Notes

- Extend the month-range rule in `data/partitions.py` — never here (C-209).
- The level vocabulary is `views_frames.SpatialLevel`; a new level added
  upstream is accepted here automatically and must be added to
  `EXPECTED_INDEX_NAMES` (CoreDataSniffer) in the same change — the parity
  test fails otherwise, by design.
- The emptiness predicate lives in `data/frame_invariants.py`; change it there.

## 12. Known Deviations

- **Endpoint semantics for `override_month` mirror the legacy contract**
  (silently ignored outside forecasting) rather than failing loud — documented
  choice, revisit if managers ever stop passing it unconditionally.
- **No schema/dtype validation of feature values** — the datafactory contract
  (float32) is enforced upstream by the #162 conformance fixture, not per-load.

## End of Contract
