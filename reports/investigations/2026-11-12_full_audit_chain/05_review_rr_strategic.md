# 5. review-rr strategic — reports/technical_risk_register.md (2026-11-12, after C-362–C-377 and D-42)

Read in full; counts measured by script, not by eye. Nothing below has been applied — this is the proposal. The strategic pass of the same morning (before this chain) curated structure and drafted clusters K, L, M; this pass measures what that one left and what the chain added.

## 1. Structural checks

| Check | Status | Detail |
|---|---|---|
| Header counts | PASS | header 377 concerns (212 resolved) + 42 disagreements; rows: 160 open, 212 resolved, 5 relocated (tombstones, counted in the 377), 42 D; guards `test_falsification_merge_readiness.py` and `test_release_gate_dispositions_what_it_found.py` green |
| Section ordering | PASS (variant) | Format → Causal Clusters → Open Concerns → Relocated → Expert Disagreements → Resolved → Stories → Process. The skill's canonical order puts Disagreements before Resolved; this register does too. The `## Format` section plays the role of Tier Definitions and Register Conventions. |
| ID sequencing | PASS | C-1..C-377 contiguous, D-1..D-42 contiguous, no duplicates, no ID in two sections |
| Formatting | PASS | every open row has six cells (the morning pass fixed C-241 and C-15); backticked paths throughout; Source carries a date on every row registered since 2026-04 |
| Cross-references | PASS | every `C-xxx`/`D-xx` cited in the file resolves |
| Last updated | PASS, with a cost | 2026-11-12 — but the "Last updated" line is now ~6,400 words of changelog on one line. It is accurate and nobody will read it. |

## 2. Duplicates

None exact. Overlaps worth a cross-reference, not a merge:
- **C-362 ↔ C-331** (no evaluate→report on-disk test; the stale-MetricFrame half of C-362 is the test C-331 asks for). Cross-reference both ways.
- **C-368 ↔ C-310** (guards deletable with the suite green): same shape, different files. Cross-referenced already.
- **C-369 ↔ C-57**: same property, two defects (ceiling vs uncached call). Keep separate; cross-referenced.
- **C-371 ↔ C-350 ↔ C-375**: three faces of "timestamp has three meanings" (Hickey). Keep separate — different fixes — but the cluster entry should name them together (see §4).
- **C-367 ↔ C-377**: the CIC half and the ADR half of the same audit. Deliberately two entries (different fix owners: CIC rewrite vs ADR supersession).

## 3. Tier calibration

Open rows by tier: 1 → 2, 2 → 48, 3 → 84, 4 → 26 (live, excluding Accepted/Withdrawn/Mitigated: 1 / 45 / 78 / 23).

| ID | Current | Proposed | Rationale |
|---|---|---|---|
| C-198 | 1 (Mitigated) | keep 1, mark the row Mitigated in the Status cell plainly | it already reads "Mitigated (the mode is now r…" — fine; the tier is right for its history |
| C-300 | 1 | keep | the morning pass settled the Tier 1 wording for published claims; do not reopen |
| C-362 | 2 | keep 2, **with the Tier 1 question recorded** | silent wrong evaluation-of-record — the only thing keeping it off Tier 1 is one ERROR line in the log. If a report built on a stale MetricFrame has ever been published, it is Tier 1 by this register's own definition. Measure before raising. |
| C-364 | 2 | keep 2 pending measurement | same reasoning: whether any production ensemble output carries a filled cell is unmeasured |
| C-293 | 2 | **3** | "the register ranks by severity and nothing connects that ranking to work" is a process concern with a tracking issue (#479); it is not structural fragility in the code. Peer Tier 2s are code defects. |
| C-288 | 2 | keep 2 | views-impact is a live consumer two majors behind with no owner — a realistic break on the next major |
| C-254 | 2 | **3** | "vendor exit cost unmeasured" has no trigger a developer can fire; D-41 already holds the decision (drill first). Tier 2 overstates a known unknown. |
| C-206 | 2 | keep 2 | proofs against wrong worktrees; realistic and recurs |
| C-349 | 3 | keep 3 | decorative `@abstractmethod`: loud at call time; Tier 2 would overstate |
| C-357 | 2 | keep 2 | accepted at the 3.5.0 gate knowingly with a stated order of operations |
| C-343 / C-344 | 2 | keep 2 | they make CI red at the next registration after a ship — which this chain just avoided only by extending the shipped gate block (see §7) |
| C-372 | 4 | keep 4 | one `raise` |
| C-374 | 4 | keep 4; **close it in the same commit this chain ships** | the fix is `git add` or `git rm` |

Consistency check across Tier 2: after the two demotions, every Tier 2 is a code or test defect with a realistic trigger, or a process defect that has already bitten (C-343, C-344, C-346, C-361). Across Tier 3/4 the boundary holds; the 26 Tier 4s are single-site.

## 4. Causal clusters

The cluster table was re-measured against 77 open concerns on 2026-06-19 and against 142 live on 2026-11-12 (morning). Measured now against 160 open rows: **81 belong to no cluster, 21 of them Tier 1 or 2** — the chain added 16 rows and none is in the table yet.

Proposed assignments (no new clusters needed; every new entry fits an existing one):

| Cluster | Add | Why |
|---|---|---|
| **J — "No" is not the same as "I don't know"** | C-362 *(absent actuals read as nothing to evaluate — keystone candidate for the un-versioned MetricFrame path)*, C-365 *(aliased key read as a match)*, C-268, C-285, C-347 | the fourth and fifth sub-shapes: a read of the wrong THING reported as the right one (C-365), and a read that never happened reported as done (C-362) |
| **E — Fail-Loud gaps** | C-363 *(unreachable refusal)*, C-372 *(implicit None)* | both are a refusal that exists and cannot fire |
| **A — Data-Contract Gap** | C-364 *(the `_ViewsDataset` zero-fill is the god class's behaviour, not a separate cause)*, C-373, C-197, C-198, C-278 | all live in the pandas tier the keystone retires |
| **K — the record is the defect** | C-367, C-370, C-371, C-375, C-377, C-252, C-263, C-350 | C-375 is the sharpest K instance this register holds: the canonical template teaches the forbidden form and the ADR points at it |
| **L — a guard that cannot fail** | C-368, C-374, C-376, C-375 *(second membership)* | eight vacuous guards, an untracked guard, three semantics held by text guards |
| **H — external-service fragility** | C-366, C-369 | both in the GitHub tag lookup |
| **M — the release ritual is true when run** | C-206, C-293 | a proof against the wrong worktree is the same shape as a check that expired |
| **B — cross-repo coupling** | C-202, C-287, C-288 | each is a contract held as two independently hard-coded things |
| **D — ensemble inheritance** | C-348 | already named there in prose, not in the member list |
| **I — Appwrite seam** | C-254 | the exit drill is the seam's unmeasured cost |

After assignment: 49 unclustered, none Tier 1/2 except C-198 (Mitigated, Cluster A proposed). That leaves the "genuinely single-cause" remainder the morning pass claimed, now true.

**One keystone changes.** Cluster J's keystone is "a rule plus one executable guard" (no unbounded `list_*`, no bare except in storage, no discarded `OperationResult`). C-362 is not reachable by that guard and is the cluster's most consequential live member. The keystone should gain a second clause: **a stage that produced nothing must not report success** — enforced by the alert-withheld test family the test review asks for (beige, Critical), which also reaches C-227's lineage and the five WandB swallow sites.

**Cluster K's keystone holds and now has its best example.** "Check citations, do not parse claims": C-375 was found by checking what ADR-052's citation pointed at, exactly as C-355 prescribed, and it is a Tier 2 defect in a shipped template.

**Cluster L's honest limit is confirmed from the other side.** The morning entry said derived scopes can derive from the wrong source. C-368 adds the older failure: a scope that was right when written and whose target was then deleted. The remedy is not another guard; it is a rule that a guard's target must exist, which `test_falsification_*` files could assert about themselves in one line (`assert path.exists()`, not `if path.exists()`).

## 5. Trigger quality

| ID | Current | Class | Proposed rewrite |
|---|---|---|---|
| C-39 | "Any modification to rolling-origin evaluation semantics" | perpetual | "When S-07 consolidates step mapping via `ForecastHorizon.build_step_mappings`, or when either of `model.py:1955-1958` / `evaluation/stage.py:466-470` is edited alone — change both" |
| C-340 | "Any change that makes a `prediction_format: dataframe` path save per target…" | perpetual in form, specific in content | "When a DataFrame-tier forecast gains `target_identifier` in its filename (the multi-target path of #459 reaching forecasts) — the cm loader's parquet branch must then select by target" |
| C-312 | "Deciding whether the resolver or the warning is wrong — and before anyone relies on ADR-057…" | compound | "When ADR-057's transition window is closed (views-models reports zero legacy configs) — decide which of the resolver and the warning is wrong before deleting either vocabulary" |
| C-314 | "The first config script that imports another config module by name — or any change making…" | compound | keep the first clause only |
| C-360 | "The next release whose content sits behind any extra, or any edit to `[tool.poetry.extras]`" | compound, acceptable | keep; both halves are specific |
| C-144 | long but specific | actionable | keep |
| C-352 | specific | actionable | keep |

All 16 new triggers name an action and a check. Of the 160, 155 are actionable after the two rewrites.

## 6. Signal-to-noise

No tech-debt backlog exists (`reports/` holds none; C-42's "demoted to tech-debt backlog" demoted it to the Resolved section). Demotion candidates meeting all four criteria (mechanical, single-file, perpetual, Tier 4):

| ID | Title | Reason | Target |
|---|---|---|---|
| C-147 | `_exit_with_error` uses `sys.exit(1)` | one-line change, single file, always true | backlog or fix in passing |
| C-159 | `sys.modules` pollution in two loaders | mechanical, standing | backlog |
| C-181 | "No run available" logged at two severities | cosmetic | fix in passing |
| C-194 | adapter deep-copies then overwrites | single file, standing | backlog |
| C-211 | non-atomic pandas cache write | has a named fix (reuse the frame-cache stager) and a trigger; **keep** | — |
| C-267 | `file_size`/`mime_type` null on every document | symptom of C-352; **merge into C-352** | — |
| C-372 | `find_project_root` implicit None | one `raise`; **fix in passing rather than track** | — |
| C-374 | untracked test | **close this week** | — |

Recommendation: create `reports/tech_debt_backlog.md` as a plain table and move C-147, C-159, C-181, C-194 there; merge C-267 into C-352; fix C-372 and C-374 in the next commit and close them. That removes six rows without losing a risk.

## 7. Blind spots

Categories present and well covered: data integrity (Clusters A, J), boundary contracts (B, I), failure propagation (E, J), testability (G, L), operational/release (M), documentation truth (K). Categories thin or absent:

1. **Reproducibility.** C-184 names it as a blind spot and nothing has moved since June: the only seed in the package is the DF pool's `default_rng(42)`; engine seeds are unobserved; no run manifest ties outputs to inputs (C-347). Still a blind spot.
2. **Training/serving skew.** The evaluation and forecasting paths build actuals and reference entities from different sources (raw cache vs frame cache vs `models[0]`), and nothing checks a forecast was made from the same input vintage the evaluation scored (`data_version` is recorded in the MetricFrame but compared by nothing). Not registered.
3. **Concurrency.** C-169 (thread pool shares mutable state) is the only entry; the ensemble subprocess fan-out shares `data_generated` directories with no locking, and two ensembles sharing a member would race on its cache. Not registered; realistic only when runs are parallelised, which the FAO cadence does not do yet.
4. **The register-guard coupling (C-343).** This chain registered 16 concerns after a shipped release and kept CI green only by appending to the SHIPPED 3.5.0 gate block, which is the workaround C-343 describes. Each audit pass will have to do the same until the guard distinguishes "between releases" from "undispositioned". Not a blind spot — a known one that just cost twenty lines of care.
5. **Security.** C-300/C-301 and the Appwrite permission audit cover the partner store; nothing covers the `shell=True` subprocess in `package.py:136` (repository name interpolated into a shell command — config-controlled today) or the executed config scripts (`importlib` on files under the model directory, by design). Low, but unlisted.

## 8. Disagreement quality

Of 42 D-entries: 29 carry an actionable resolution with evidence; 8 resolved by subsequent code (D-01, D-05, D-06, D-08, D-11, D-21, D-40, D-41) and say so; 4 are "now vs later" prioritisations rather than tensions (D-04, D-13, D-22, D-26) — valid history, weak as disagreements; D-02 (freeze config) is resolved on paper and not implemented, and C-375 is the consequence — it should be reopened or its resolution dated as deferred; D-36 (flip the report default) resolved to "warn now, reject later" with no trigger for "later"; D-42 (new) is genuinely in tension and names Simon as the decider because it amends a constitutional ADR.

## 9. Summary of proposed changes

| ID | Change | Rationale |
|---|---|---|
| Cluster table | add 32 memberships as in §4; no new clusters | 81 → 49 unclustered; every T1/T2 clustered |
| Cluster J keystone | add "a stage that produced nothing must not report success" + the alert-withheld test family | reaches C-362 and the five swallow sites |
| C-293 | 2 → 3 | process concern with a tracking issue |
| C-254 | 2 → 3 | no developer-fireable trigger; D-41 holds the decision |
| C-39, C-340, C-312, C-314 | trigger rewrites as in §5 | perpetual/compound |
| C-362 ↔ C-331 | cross-reference | the test one asks for is the other's remedy |
| C-267 → C-352 | merge | symptom of the two-builder split |
| C-147, C-159, C-181, C-194 | demote to a new `reports/tech_debt_backlog.md` | mechanical, perpetual, Tier 4 |
| C-372, C-374 | fix and close in the next commit | one `raise`; one `git add` |
| D-02 | reopen or date as deferred | its non-implementation produced C-375 |
| D-36 | add the trigger for "reject" | resolution has no "later" |
| Header "Last updated" line | move the changelog into a `## History` section, keep one sentence in the header | 6,400 words on one line |

## 10. Strategic recommendation

**Verdict: NEEDS CURATION** (structure sound; 81 unclustered rows and two mis-tiered Tier 2s; the header line has become unreadable).

1. Immediately: apply §4's cluster memberships and the two tier changes; fix C-372 and C-374 in the same commit; cross-reference C-362/C-331.
2. Before the next planning cycle: the four trigger rewrites; the backlog file; the header-line split.
3. For the roadmap: Cluster J's second keystone clause is the one piece of test infrastructure that reaches the most live Tier 2s at once (C-362, C-227's lineage, C-21's beige gap); Cluster K's remedy is now concrete and small — supersede five ADRs, fix one template, rewrite five CICs — and it closes nine entries.
4. Demote or archive: the four Tier 4 standing items.

Want me to apply these fixes? The cluster memberships, tier changes, trigger rewrites, cross-references and the C-267 merge are mechanical; the backlog file, the header split and the D-02/D-36 reopenings change the register's shape and are yours to call.
