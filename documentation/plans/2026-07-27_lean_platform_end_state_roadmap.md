# Roadmap: The Lean Platform End-State — what remains, in what order, and when it's over

**Date:** 2026-07-27
**Author:** pipeline-core agent (owner: Simon)
**Branch:** development
**Relationship to other documents:** This roadmap is the **owner of the convergence** that
views-models **vm-017 §1 "Direction of travel"** (their `017_source_composition_delivery.md`) describes and explicitly does not decide
("owned by the views-frames migration + vpp ADR-013"). It formalizes the retirement promises
already made by **ADR-042** (DF path is temporary, named removal targets) under **ADR-004**'s
rule ("No silent legacy retention"), and continues the **"Pure Math Engine" vision**
(ADR-039, the 2026-02-27 Phase-3 purge roadmap). It supersedes no existing plan; the
2026-06-01 PFE production roadmap remains authoritative for its own scope (shipping PFE).

> **Numbering note:** "vm-017" throughout = **views-models'** source/composition/delivery ADR (their repo, doc 017) (a different repo's ADR — pipeline-core has no ADR 017; spelled vm-017 so the local cross-ADR validator doesn't resolve it here, mirroring vm-017's own "vpp ADR-013" convention).

**Vocabulary (adopted verbatim from vm-017 so the two documents read as one system):**
the **store** = the legacy `views-forecasts` central store (VPN-only Postgres at PRIO,
pandas-only door, feeds the public API). The **shelf** = the Appwrite `production_forecasts`
bucket. Tags: **[LEGACY]** alive-until-retired · **[CURRENT]** the working present ·
**[TARGET]** the destination.

---

## 1. Executive summary

The platform *feels* messy because three producer generations, two stores, two shelf
dialects, and two delivery legs coexist — but the mess is **transitional, fully mapped**
(pinned executable: `tests/test_managers/test_delivery_characterization.py`), and it
converges on one shape:

> **[TARGET]** FeatureFrame in → frames-native compute → PredictionFrame out → the shelf's
> contract dialect → contract-reading consumers → one delivery declaration per consumer
> (vm-017) → pandas nowhere on any data path, and eventually not installed at all.

What was missing was not direction but **ownership**: three demolitions had no issue
anywhere, and no document stated the dependency order. This roadmap fixes that. It uses
**gates, not dates** — every "when" in this platform is an event, per the publish-last
doctrine.

## 2. The end-state, concretely ("when is it over?")

The platform is *lean* when ALL of the following hold:

- [ ] Every production model/ensemble is frames-native (FeatureFrame in where datafactory
      feeds it; PredictionFrame out) — no gen-1 machinery has a production user.
- [ ] The **store is retired**: `api.viewsforecasting.org` serves from a modern source; the
      gen-1 io machinery (`PredictionIOManager._upload_to_prediction_store`,
      `ViewsForecastsSaver`, the `ViewsMetadata` run registry, the legacy ensemble
      `read_store` transport) is deleted; the `views_forecasts` package leaves the platform.
- [ ] The **shelf speaks one dialect** (`type="sampled_forecast_*"`); the legacy
      `type=model/ensemble` documents and the legacy FAO leg are gone (vpp ADR-013 §11.4
      completes).
- [ ] **Delivery is declared** per vm-017 (maturity / composition / delivery axes;
      `is_in_production` derived; liveness observing every surface).
- [ ] The **pandas ecosystem pins are lifted** (register C-112: 8 packages pin
      `pandas <2.0`; empirically pandas 3.0.3 passes 1385/1386 pipeline-core tests).
- [ ] `get_data` (the pandas input path) is deprecated per the declared end-state
      (epic #285 condition 5; executed at Epic C close-out #267).

## 3. The gates, in dependency order

```
G0 [CURRENT]  S1 proof run (rusty_bucket at scale)         ← waits on: disk + ZINB (owner)
G1            Epic B release train (#261/#264)              ← waits on: G0 + owner sign-off
G2 [CURRENT]  Pandas-free pilot completes (#300/#303)       ← waits on: baseline#64 → models#277 (owner)
G3            Engine activation (#266 + views-hydranet#157) ← after G0 (input-path stability argument)
G4            Engine PF-output for stepshifter + r2darts2   ← the C-140 verification gate
              (2026-06-01 roadmap §7: only hydranet verified for raw-target-space undo)
G5            Monthly ensembles → PFE (4 ensembles, 38 constituents, all on the G4 engines)
              ⚠ COUPLED TO G6 — see §4. Kills: gen-1 writes, the store's constituent-
              transport role, the legacy shelf dialect's production feeders.
G6            Public-API re-point (prio-data/views_api, external) — the store loses its
              last consumer. MUST land before or with the END of G5 (see §4).
G7            Store retirement (delete gen-1 io machinery + ViewsForecastsSaver +
              ViewsMetadata + the VPN Postgres estate) + C-112 coordinated pandas unpin
              + vm-017 Phases 2–4 land (rename, write-gate, guard re-home).
```

G2 and G3/G4 are parallel tracks to G0/G1; G5 needs G4; G7 needs G5+G6. vm-017's Phase 2
(the maturity rename) explicitly cannot ride pipeline-core 3.0 (G1) — it lands post-3.0.

## 4. The load-bearing coupling (the trap this document exists to prevent)

**The PFE never writes to the store.** It emits `save_pf` (npy/npz) + wire shards only.
Meanwhile the store has THREE roles today (touchpoint inventory, 2026-07-27 investigation):

1. **Public-API backend** — `api.viewsforecasting.org` serves store runs [LEGACY];
2. **The legacy ensembles' internal transport** — `EnsembleManager` READS its constituents'
   forecasts from the store (`ensemble.py:603,637,802` `forecasts.read_store`) [LEGACY];
3. **Run-metadata registry** — `ViewsMetadata` run bookkeeping (`model.py:299`) [LEGACY].

Therefore: completing G5 (last monthly ensemble → PFE) **starves the store**, and with it
the public API. Roles 2 and 3 die *with* G5 harmlessly — role 1 does not. **G6 must land
before, or in the same phase as, the end of G5** — or a temporary bridge egress
(a ViewsForecastsSaver at the PFE level) must be consciously added and later deleted.
Nobody should discover this mid-migration; it is now written down.

## 5. What already exists vs what this roadmap adds

**Existing owners (unchanged):** Epic B #261/#264 (G0–G1) · Epic C #265/#268 (input side,
G3) · Epic #300/#303 (G2) · vpp ADR-013 + faoapi#184 (FAO contract-leg activation) ·
views-models vm-017 (delivery governance, Phases 1–4) · shim policy (#184-closed,
keep-until-qualified).

**Filed WITH this roadmap (previously zero coverage):**

- **views-models#278: "Epic: migrate the four monthly ensembles to PFE"** (G4+G5) — open-form;
  names the engine gate (stepshifter + r2darts2 PF-output, C-140 verification) and the §4
  coupling; engine-repo asks get filed when the epic activates.
- **pipeline-core#307: "Epic: retire the views-forecasts store"** (G6+G7) — blocked/needs-decision;
  carries the full touchpoint inventory and the external coordination need (prio-data/views_api).
- **pipeline-core#308: "C-112: coordinated pandas unpin"** (G7 tail) — the 8-package
  simultaneous-release coordination issue.

**Register entries added with this roadmap:** C-215 (ensemble guard's stale-log gap —
requested by vm-017 §12), C-216 (undeclared `views_forecasts` runtime dependency in
pipeline-core and views-models/liveness; stepshifter carries the only pin and doesn't use it).

## 6. Non-goals of this document

No dates (gates only). No new architecture decisions (vm-017 owns delivery governance;
vpp ADR-013 owns the wire; this document owns *sequence and ownership*). No demolition
starts early: G5 explicitly does NOT begin until G4's verification gate opens, and nothing
here changes what today's monthly production run does.

## 7. Maintenance

This document shrinks: when a gate closes, its line gains the merge/PR receipt; when a
[LEGACY] element dies, vm-017 §1 retires its line and this document strikes the gate.
Review trigger: any time a new epic touches delivery, input, or the stores, it must name
its gate here — an epic with no gate is a sign this map has drifted.
