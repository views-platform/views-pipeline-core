# LEGACY DataFrame tier — pandas by design; retires with roadmap G5–G7 (#313/#307). See C-226.
"""
CorePredictionSniffer: Audit-only validator for prediction DataFrames.
Called after model inference, before evaluation or storage.
Fail Loud and Proud. Follows the hydranet DataSniffer pattern.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import FrozenSet, Optional, Union

import pandas as pd

from views_pipeline_core.modules.validation.core_data_sniffer import (
    EXPECTED_INDEX_NAMES,
    _canonical_index_name,
    _check_multiindex,
)

logger = logging.getLogger(__name__)

#: How many offending entity ids an entity-coverage refusal lists before "and N more".
#: Enough to recognise a pattern (the 22 dissolved states of #509 fit), few enough to read.
ENTITY_COVERAGE_MAX_LISTED = 25

#: The time level shared by every layout in EXPECTED_INDEX_NAMES.
_TIME_LEVEL = "month_id"


def _entity_level(level: str) -> str:
    """The entity index name for a level — 'country_id' or 'priogrid_id' (canonical)."""
    names = EXPECTED_INDEX_NAMES[level]
    return next(n for n in names if n != _TIME_LEVEL)


def _entity_values(df: pd.DataFrame, level: str) -> pd.Index:
    """The entity level of a MultiIndex, found by canonical name so the transitional
    `priogrid_gid` spelling (see `_GRID_ID_ALIASES`) resolves like `priogrid_id`."""
    wanted = _entity_level(level)
    for name in df.index.names:
        if _canonical_index_name(name) == wanted:
            return df.index.get_level_values(name)
    raise ValueError(
        f"CorePredictionSniffer: no '{wanted}' level in index {list(df.index.names)}."
    )


def reference_entities_from_raw(path: Path, level: str, at_month: int) -> FrozenSet[int]:
    """The entity ids present in a raw data cache at one month — the reference set for
    `sniff_predictions(reference_entities=...)` (ADR-064).

    Reads only the two index columns of the parquet, so it costs a filter, not a load.
    Raises if the month is absent from the cache: a reference taken at a month the data
    does not cover would make every prediction look like a phantom.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = _entity_level(level)
    schema = pq.read_schema(path)
    entity_col = next(
        (n for n in schema.names if _canonical_index_name(n) == wanted), None
    )
    if entity_col is None or _TIME_LEVEL not in schema.names:
        raise ValueError(
            f"CorePredictionSniffer: raw cache {path} has no '{_TIME_LEVEL}'/'{wanted}' "
            f"columns to take a reference entity set from (found {schema.names})."
        )
    table = pq.read_table(path, columns=[_TIME_LEVEL, entity_col])
    at = table.filter(pc.equal(table[_TIME_LEVEL], at_month))
    if at.num_rows == 0:
        raise ValueError(
            f"CorePredictionSniffer: raw cache {path} has no rows for month {at_month}; "
            f"cannot take the reference entity set there (months "
            f"{pc.min(table[_TIME_LEVEL]).as_py()}–{pc.max(table[_TIME_LEVEL]).as_py()})."
        )
    return frozenset(int(v) for v in at[entity_col].to_pylist())


class CorePredictionSniffer:
    """
    Audits prediction DataFrame output before evaluation or storage. Read-only throughout.
    Fail Loud and Proud. Follows the hydranet DataSniffer pattern.

    PredictionFrame is self-validating at construction and requires no external sniffer.
    """

    def __init__(self, level: str) -> None:
        self._level = level

    def sniff_predictions(
        self,
        df: pd.DataFrame,
        targets: Union[str, list],
        reference_entities: Optional[FrozenSet[int]] = None,
    ) -> None:
        """Audit suite for prediction DataFrame output before evaluation or storage.

        Args:
            df: the predictions, indexed (entity, month) per EXPECTED_INDEX_NAMES.
            targets: the target name(s) whose `pred_*` columns must be present.
            reference_entities: the entity ids present in the model's input at its last
                observed month (ADR-064; `reference_entities_from_raw`). When given, a
                prediction for any entity outside it is refused — an engine that forecasts
                a dissolved state is forecasting nothing, and nothing downstream can tell
                (#509). When None the check is skipped and says so; a caller that cannot
                supply the reference is loud about it rather than silently unguarded.
        """
        self._check_not_empty(df)
        self._check_targets_type(targets)
        self._check_prediction_columns(df, targets)
        self._check_multiindex_structure(df)
        if reference_entities is None:
            logger.info(
                "CorePredictionSniffer: entity coverage NOT checked — no reference entity "
                "set was supplied (ADR-064)."
            )
        else:
            self._check_entity_coverage(df, reference_entities)
        logger.info("CorePredictionSniffer: prediction DataFrame audited.")

    def _check_entity_coverage(
        self, df: pd.DataFrame, reference_entities: FrozenSet[int]
    ) -> None:
        """Every predicted entity must exist in the reference set (ADR-064).

        Structural, not semantic (ADR-040): two index sets are compared. The reference is
        the input's entity set at the last observed month; an entity absent from it has
        ceased to exist and has no forecast to make. The refusal names the entities and the
        months they were forecast for, because the count alone — the only thing the
        ensemble guard reported in #509 — sent the operator to reconstruct the cause by hand.
        """
        entities = _entity_values(df, self._level)
        phantom = sorted(set(int(v) for v in entities.unique()) - set(reference_entities))
        if not phantom:
            return
        mask = entities.isin(phantom)
        months = df.index.get_level_values(_TIME_LEVEL)[mask]
        lo, hi = int(months.min()), int(months.max())
        n_rows = int(mask.sum())
        # The same "N value(s) [a, b, …, and M more]" shape as the two pool refusals
        # (aggregator._describe_rows, prediction_frame_ensemble._describe_values). Three
        # sites, three containers (pandas Index, polars frame, numpy array); extracted on the
        # FOURTH site, not before — WET before DRY, with the trigger named.
        listed = ", ".join(str(e) for e in phantom[:ENTITY_COVERAGE_MAX_LISTED])
        more = len(phantom) - ENTITY_COVERAGE_MAX_LISTED
        raise ValueError(
            f"CorePredictionSniffer: predictions cover {len(phantom)} "
            f"{_entity_level(self._level)} value(s) absent from the input at its last "
            f"observed month, so treated as having no forecast to make (ADR-064; #509): "
            f"[{listed}{f', and {more} more' if more > 0 else ''}], forecast for "
            f"{f'month {lo}' if lo == hi else f'months {lo}–{hi}'} "
            f"({n_rows} row{'' if n_rows == 1 else 's'}). The reference set has "
            f"{len(reference_entities)} entities. An engine that reindexes its input to "
            f"every entity ever seen produces exactly this; forecast the entities present "
            f"in the last observed month instead."
        )

    def _check_not_empty(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("CorePredictionSniffer: Prediction DataFrame is empty.")

    def _check_targets_type(self, targets: object) -> None:
        if not isinstance(targets, (str, list)):
            raise ValueError(
                f"CorePredictionSniffer: Invalid targets type: {type(targets)}. "
                f"Expected str or list."
            )

    def _check_prediction_columns(
        self, df: pd.DataFrame, targets: Union[str, list]
    ) -> None:
        required = {
            f"pred_{t}" for t in ([targets] if isinstance(targets, str) else targets)
        }
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CorePredictionSniffer: Missing prediction columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

    def _check_multiindex_structure(self, df: pd.DataFrame) -> None:
        _check_multiindex(df, self._level, self.__class__.__name__)
