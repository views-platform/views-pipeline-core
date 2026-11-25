# 6. review-rr prioritize — work plan from the register (2026-11-12)

Inputs: the register after C-362–C-377, the cluster map with §4's proposed memberships applied for scoring, the roadmap context in memory (viewser exit via datafactory, 4.0 waits for Simon's go; the FAO delivery cadence; C-293's #479). Tier weights 1→4, 2→3, 3→2, 4→1. Scope: 1 = one targeted change, 2 = multi-file, 3 = architectural. Read-only; nothing in the register changed.

## 1. Cluster rankings

| Rank | Cluster | Live entries | Avg tier weight | Fix scope | Score | Quick win? |
|---|---|---|---|---|---|---|
| 1 | **K — the record is the defect** (C-367, C-370, C-371, C-375, C-377, C-252, C-263, C-273, C-297, C-300, C-301, C-303, C-304, C-312, C-350, C-355, C-195) | 17 | 2.4 | 1 (supersede 5 ADRs, fix 2 template lines, rewrite 5 CICs, fix the index — each a document edit with no code risk) | **40.8** | Yes |
| 2 | **J — "No" ≠ "I don't know"** (C-362, C-365, C-26, C-145, C-152, C-166, C-178, C-183, C-257, C-339, C-268, C-285, C-347) | 13 | 2.5 | 2 (raise on absent actuals + run id in the MetricFrame path + the alert-withheld tests; the rest stay) | 16.3 | No |
| 3 | **L — a guard that cannot fail** (C-368, C-374, C-376, C-375, C-289, C-294, C-310, C-312, C-336, C-338, C-343, C-344, C-349, C-357) | 14 | 2.4 | 2 (delete eight files, commit one, write three output assertions, and the two guard rewrites C-343/C-344) | 16.8 | partly: the deletions alone are scope 1 and close C-368 + C-374 |
| 4 | **A — Data-Contract Gap** (C-36, C-38, C-40, C-48, C-66, C-67, C-135, C-141, C-164, C-165, C-167, C-176, C-182, C-186, C-187, C-191, C-193, C-364, C-373, C-197, C-278) | 21 | 2.5 | 3 (the DataFrame-tier retirement) | 17.5 | No |
| 5 | **M — the ritual is true when run** (C-344, C-345, C-346, C-343, C-359, C-360, C-361, C-101, C-206, C-293) | 10 | 2.5 | 1 (the publishing guide + one repository setting) | 25.0 | Yes — but C-361 is Simon's and C-343/344 are code |
| 6 | **E — Fail-Loud gaps** (C-363, C-372, C-26, C-166, C-176, C-178) | 6 | 2.3 | 1 (make three existing refusals reachable) | 13.8 | Yes |
| 7 | **I — Appwrite seam** (C-44, C-166, C-217, C-220, C-221, C-234, C-235, C-237, C-245, C-254) | 10 | 2.0 | 3 | 6.7 | No |
| 8 | **D — ensemble inheritance** (C-03, C-55, C-65, C-67, C-141, C-158, C-348) | 7 | 2.3 | 3 | 5.4 | No |
| 9 | **B — cross-repo coupling** (C-26, C-30, C-47, C-50, C-144, C-167, C-168, C-202, C-287, C-288) | 10 | 2.1 | 3 | 7.0 | No |
| 10 | **H — external-service fragility** (C-57, C-101, C-112, C-173, C-179, C-366, C-369) | 7 | 2.1 | 2 | 7.4 | No |
| 11 | G, C, F | 6, 4, 2 | 2.0 | 2-3 | ≤ 6 | No |

## 2. Standalone rankings (unclustered after §4 assignments, top 10 of 49)

| Rank | ID | Title | Tier | Trigger proximity | Fix effort | Score |
|---|---|---|---|---|---|---|
| 1 | C-215 | ensemble deprecation check reads stale logs | 3 | imminent (ADR-017 Phase 4 is views-models' next guard move) | medium | 2.0 |
| 2 | C-331 | no evaluate→report on-disk test | 3 | imminent (it is C-362's test) | medium | 2.0 |
| 3 | C-214 | frame-path opt-in degrades silently on two engines | 3 | plausible (#266 activation) | small | 2.0 |
| 4 | C-203 / C-204 | gid→id consolidation residue | 3 | plausible (PR-3, #259) | medium | 1.0 |
| 5 | C-216 | views_forecasts undeclared where used | 3 | plausible (G7) | small | 2.0 |
| 6 | C-322 | pandas-1 pickles untested on pandas 3 | 3 | unlikely until 4.0 | large | 0.3 |
| 7 | C-330 | views-evaluation `[frames]` caps views-frames <2 | 3 | theirs | — | blocked |
| 8 | C-332 | wandb tables keyed per schema not per target | 4 | plausible | small | 1.0 |
| 9 | C-211 / C-212 | two cache-write and two frame-load semantics | 4 | plausible | small | 1.0 |
| 10 | C-05, C-06, C-07, C-08, C-17, C-18, C-29 | Accepted standing items | 3-4 | — | — | accepted |

## 3. Blocked items

| ID | Blocked by | Unblocking action |
|---|---|---|
| C-36, C-135, C-182, C-364 (Cluster A core) | the DataFrame-tier retirement has no ADR and no window; D-36 resolved "warn now, reject later" with no "later" | **write the retirement ADR with an ADR-062-style refusing window** (one document) — it is the prerequisite for half the register |
| C-361 | repository ruleset | Simon: change the ruleset (three approvals → one; add `run_pytest` as a required check) |
| C-330, C-288, C-263 | sibling repos (views-evaluation, views-impact) | issues filed; nothing here |
| C-343, C-344 | the release-gate guard's design | rewrite the guard to read the gate's `Shipped` stamp and to treat concerns registered after it as "next gate" — small, and every audit pass pays for its absence |
| D-42 (and C-377's ADR-008 row) | amends a constitutional ADR | Simon's decision: cite 047 as the carve-out, or restore the raise |
| C-375 (full closure) | D-02 (freeze config) | fixing the template is unblocked now; making the facade perform the copy-back is the D-02 implementation |

Prerequisites (fix first, unblocks others): the retirement ADR (unblocks Cluster A's scope-3 work and lets C-191/C-187/C-193 close as "retired"); the guard rewrite (C-343/C-344, unblocks every future registration); the alert-withheld test family (unblocks C-362's closure and gives Cluster E its mechanism).

## 4. Recommended sprint plan (execution order)

1. **Cluster K quick win — documents only, ~half a day, closes or narrows 9 entries.** Fix the two template lines (C-375 half); strike ADR-052's "already demonstrates" sentence; supersede 014, 020, 022, 029 and 042-M4 with one paragraph each; fix the 16 index statuses; rewrite the five Critical CICs' cited sections (C-367); amend ADR-008's row once D-42 is decided (C-377). Why first: no code risk, and it removes the false record that produced C-375 before the next engine is scaffolded.
2. **Cluster E/J narrow code fixes — one PR, ~a day, closes C-362 (half), C-363, C-372, C-374; narrows C-26.** Evaluation stage raises on absent actuals; `run_id` or the artifact stamp in the MetricFrame directory with `PerModelMetricFrameSource` refusing a frame whose provenance predates the run; `_normalize_weights_new` refuses unmatched names and the sniffer checks `weights`; the `raise` after `find_project_root`'s loop; commit or delete the untracked test. Each fix lands with the red test that fails today.
3. **Cluster L deletions and the guard rewrite — one PR, ~a day, closes C-368, narrows C-376, C-343, C-344.** Delete the eight vacuous guard files; three output assertions on a real two-member fixture; the release-gate guard reads the `Shipped` stamp.
4. **The DataFrame-tier retirement ADR — one document, Simon's go on the window.** Unblocks Cluster A; lets D-36's "later" become a date.
5. **Measure C-364 before touching it.** One script over the last FAO delivery's ensemble outputs: count cells that `_preprocess_dataframe` filled. If zero, the entry narrows to "latent"; if not, it is Tier 1 and goes to the top of the next sprint.

Excluded from the sprint: every scope-3 decomposition (C-01, C-36, C-221, C-353 — behind the retirement), Cluster I (behind the exit drill, D-41), Cluster B (cross-repo, needs þing), C-361 and D-42 (Simon's).

## 5. Deferred items

| Item | Reason |
|---|---|
| Cluster A decomposition (C-36, C-135, C-182) | blocked on the retirement ADR; scope 3 |
| Cluster I (Appwrite seam) | D-41 says drill first; the drill has not run |
| Cluster B contract tests | needs a þing with views-reporting and views-evaluation |
| C-322 pandas-3 pickles | 4.0 train |
| C-215 | ADR-017 Phase 4 belongs to views-models' guard re-home |
| C-254 vendor exit | D-41 |
| C-184/C-185/C-347 reproducibility and lineage | the run-manifest remedy exists for the external store; porting it to the tree is a design, not a fix |

## Verdict: CLEAR PRIORITY

Cluster K scores 40.8 against 25.0 for the next (M, which is mostly Simon's ruleset and the guide), and it is a documents-only quick win that removes the record that is actively teaching the wrong pattern. Start there, then the one-PR code fixes in Cluster E/J, which carry the three Tier 2 silent paths this chain found.
