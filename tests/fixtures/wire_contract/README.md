# ADR-013 §10 Golden Fixture — the wire contract's executable spec

**Contract:** ADR-013 "The Sampled-Forecast Wire Contract" v1.5 (incl. Erratum E1).
**Canonical source:** THIS directory in views-postprocessing. Other repos **vendor a copy**
and carry a pinned-hash equality test (§10.1) — **the hash, not the bytes, is the
cross-repo contract**; on mismatch, re-vendor from here.

**The pinned root hash (SHA-256 of `SHA256SUMS`):**

```
9658a6484cc9d975412e52624d52f328985f14cf58e3fc9fbdf3e64ab5a0564b
```

## Contents (1 run × 1 target × 1 month × 6 cells × S=4)

| File | Contract § | What it pins |
|---|---|---|
| `fixture_run_0__lr_ged_sb__m000543.tap.zip` | §3.1 | Hop-A Track-A shard: `y_pred.npy` (6,4) float32 + `identifiers.npz` (time/unit) + `metadata.json` (the §2 header, byte-pinned) |
| `fixture_run_0__lr_ged_sb__manifest.json` | §3.2 | Hop-A (run,target) manifest — shard hash, expected months/cells, `sidecar_sha256: null` (**Erratum E1**) |
| `fixture_run_0__lr_ged_sb__m000543.arrow.parquet` | §4.1 | Hop-B shard via `views_frames.io.arrow` — §2 header in the `views_frames` KV metadata; `sample` column = `tile(arange(4), 6)` (the §4.5(b) ordering oracle) |
| `fixture_run_0__sidecar.parquet` | §5.1 | The 9 pinned GAUL columns, gid-keyed; **last gid carries NaN codes** (the NaN-preserved rule, pinned) |
| `fixture_run_0__manifest.json` | §4.2 | Hop-B run manifest — one per run, spans targets, shard hashes + **the sidecar hash** (E1: it lives here, only here) |
| `SHA256SUMS` | §10.1 | Per-file hashes; its own SHA-256 is the root hash above |

Deliberate data properties: values row 0 is draw-degenerate (pins §6's
per-row-zero-variance-is-legal); target vocabulary is `lr_ged_sb` (§7a); provenance is
injected fixed literals (`run_id="fixture_run_0"`, `generated_at="2026-07-15T00:00:00Z"`, §10.2).

## Regeneration

`PYTHONPATH=. python3 scripts/build_wire_fixture.py` — byte-reproducible **with the pinned
tool versions** (numpy per lockfile, `pyarrow 16.1.0`). The committed bytes + `SHA256SUMS`
are canonical regardless. **A change to this fixture is a change to the contract (§10)** —
do not regenerate casually.

**`pyarrow` is the version-sensitive one; `views_frames` is not.** Parquet bytes vary
across pyarrow versions — that is why the pin is `>=16.1.0,<17.0.0` and why a local
pyarrow 23.x fails byte-parity while CI passes (register **C-72**). `views_frames` sits
above it: the fixture was generated under `1.0.0`, CI has reproduced it continuously under
`1.6.0`, and `1.10.2` was verified to regenerate **all five artifacts byte-identically**
before the pin was raised (2026-08-02). Raising it again does not require regenerating the
fixture — but it does require proving that, the same way.

**Confirmed across a `views_frames` MAJOR (2026-08-21).** The shard emitted through
`views_frames.io.arrow` under **1.10.2** and under **2.0.0** hashes identically, and both
equal the committed fixture (`203650fd…12c54`) — measured in an isolated environment at the
pinned toolchain (pyarrow 16.1.0, numpy 1.26.4), where all 61 frames-dependent tests also
pass at 2.0.0. So the views-frames 2.0.0 adoption (#286) is **not** a fixture re-vendor. It
is blocked only by views-pipeline-core, every published release of which (through 3.1.1)
pins `views-frames <2.0.0`. Recorded so the byte question is not re-opened when that
constraint widens.