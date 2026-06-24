"""`SpatialLevel` is now the published leaf type (views-frames); pipeline-core
re-exports it from `views_pipeline_core.domain` (issue #187, retiring the local
copy). These tests only assert the re-export wiring — the leaf owns its own
behaviour tests (time-first `index_names`, consistent `priogrid_id`, etc.).
"""
import views_frames

from views_pipeline_core.domain import SpatialLevel


def test_domain_reexports_the_leaf():
    # the public `views_pipeline_core.domain.SpatialLevel` IS the leaf's enum,
    # not a local duplicate — so there is one source of truth platform-wide.
    assert SpatialLevel is views_frames.SpatialLevel


def test_members_and_string_construction():
    assert {m.value for m in SpatialLevel} == {"cm", "pgm"}
    # config["level"] strings construct via the standard Enum(value) constructor
    assert SpatialLevel("cm") is SpatialLevel.CM
    assert SpatialLevel("pgm") is SpatialLevel.PGM


def test_pgm_entity_column_is_consistent():
    # the local copy's latent gid/id inconsistency (register C-18/C-65) is gone:
    # the leaf uses priogrid_id for the PGM entity column and index name alike.
    assert SpatialLevel.PGM.entity_column == "priogrid_id"
    assert SpatialLevel.CM.entity_column == "country_id"
