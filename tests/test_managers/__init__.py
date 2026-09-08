"""Marks `tests/test_managers` as a package, matching `tests/test_domain`.

Without it, `tests/test_managers/*` is imported by pytest under its bare basename while
`tests/test_modules/test_migrated_source_completes_a_run.py` imports one of them as
`tests.test_managers....` — two module objects for one file. Harmless today; the kind of
thing that stops being harmless the first time module-level state matters.
"""
