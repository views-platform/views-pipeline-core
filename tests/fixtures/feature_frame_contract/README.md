# Vendored: datafactory consumer-contract fixture (ADR-050 upstream)

Vendored verbatim from **views-datafactory**
`tests/fixtures/feature_frame_contract/` (their ADR-050 / epic #342;
adopted here for pipeline-core #162).

- `contract.json` — the language-neutral consumer contract: valid
  `output_format` vocabulary, identifier semantics, dtype, tensor
  shape, layout file list, and the pinned `fixture_digest`.
- `frame/` — a real `views_frames.FeatureFrame.save()` output,
  committed generator output, **never hand-edited**.

`tests/test_modules/test_datafactory_contract_conformance.py` is the
consumer-side conformance suite: fixture integrity (digest), layout
round-trip through our installed views-frames, `_LOA_TO_OUTPUT_FORMAT`
vocabulary containment, and a freshness alarm against the installed
datafactory package (skips where datafactory is not installed, e.g. CI).

## Re-vendoring

Only when upstream bumps `contract_version` (their stability promise:
member meanings never change; rename/removal = MAJOR, addition = MINOR):

```
cp -r <views-datafactory>/tests/fixtures/feature_frame_contract/{contract.json,frame} \
      tests/fixtures/feature_frame_contract/
git add -f tests/fixtures/feature_frame_contract/frame/values.npy   # *.npy is gitignored
```

Review the upstream diff first — a changed fixture is the drift alarm
working, not a nuisance. Never re-vendor to silence a failing test
without reading upstream's changelog/ADR-050 record.
