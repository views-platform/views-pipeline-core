# Why each audit finds defects in the previous fix

**Date:** 2026-08-24 · **Branch:** `fix/292-least-privilege-provisioning` (PR #483)
**Method:** four falsification audits, claim mode. Every number below is measured, not estimated.
**Occasion:** three `/code-review max` rounds and one independent guard-mutation audit produced
54 findings across 17 commits, and the rate did not fall. The question asked was whether that is
a symptom rather than a cause.

---

## The four hypotheses, and the verdicts

| # | claim | verdict |
|---|---|---|
| H1 | The audits are work generators — each fix becomes the next audit's material | **SURVIVED**, with one correction |
| H2 | The code is messy, debt-ridden, whack-a-mole | **FALSIFIED** |
| H3 | The code is sane; old defects are surfacing because the tools improved | **FALSIFIED** for this session |
| H4 | The code is fine; the test suite has grown unruly and we keep patching it | **CONTESTED** — the growth is real; the defect-density half is withdrawn as detection-limited |

---

## The measurements

### 1. What the branch wrote

| | files | lines added |
|---|---:|---:|
| tests | 6 | **1,940** |
| production | 4 | 586 |
| docs + register | 5 | 736 |

**3.3 lines of test per line of production code.**

### 2. Where the 54 findings landed

| target | count | share |
|---|---:|---:|
| production code | 23 | 43% |
| guards / test machinery | 14 | 26% |
| the written record (ADR, CIC, guide, CHANGELOG, register) | 11 | 20% |
| missing tests | 6 | 11% |

**51 of 54 (94%) were in artifacts this branch created.** Three were pre-existing: a sibling
repo's vendored copy, `ensure_bucket`'s absent tests, and the allowlist structure in
`test_read_completeness.py`.

### 3. Defect density by artifact type

| | lines written | findings | per 100 lines |
|---|---:|---:|---:|
| production code | 586 | 23 | **3.9** |
| test code | 1,940 | 20 | 1.0 |
| records / docs | 736 | 11 | 1.5 |

Production code carries **3.8× the defect density** of the tests written to guard it.

### 4. Provenance — which commit introduced each defect

Round 3's findings, blamed to their introducing commit:

| finding | introduced by |
|---|---|
| unparseable grant deletes an open container | round-1 fix commit |
| `or []` reintroduced in `_read_items` | **round-2 fix commit** |
| renderer claims it never read per-item permissions | **round-2 fix commit** |
| refusal fires on `permissions=[]` | **round-2 fix commit** |
| exit code demotes a finding | original implementation |

**Four of five were introduced by an earlier fix in the same session.**

Round 2's findings, blamed the same way, came almost entirely from the **original**
implementation — not from round 1's fixes.

### 5. Line provenance of the two most-churned files

| file | from the original commit | from later fix commits | total |
|---|---:|---:|---:|
| `permissions.py` | 218 (47%) | **249 (53%)** | 467 |
| the AST guard | 316 (34%) | **625 (66%)** | 941 |

Both files are now majority-rework. The guard has been rewritten **three times** and touched in
6 of 17 commits.

### 6. Defect density per fix commit

| commit | lines added | findings later traced to it | per 100 lines |
|---|---:|---:|---:|
| original implementation | 1,217 | 9 | 0.74 |
| round-1 fixes | 607 | 2 | 0.33 |
| round-2 fixes | 498 | 4 | **0.80** |
| mutation-audit fixes | 484 | 2 | 0.41 |

**Fix commits carry the same defect density as the original work.** A fix is not a safer kind
of code.

### 7. Complexity — testing H2 directly

| module | lines | max nesting | longest function |
|---|---:|---:|---:|
| repo median (93 modules) | 127 | 2 | 56 |
| repo 90th percentile | 887 | 4 | 127 |
| **`audit/permissions.py`** — 18 findings, the most | 467 | 4 | 87 |
| `appwrite/file.py` — a registered god class | 2,847 | 6 | 288 |

The file with the most findings sits **at or below the repo's 90th percentile on every
measure**. The genuine god class produced almost no findings this session.

### 8. Test-suite growth — testing H4

| release | production | tests | ratio |
|---|---:|---:|---:|
| 2.3.0 | 26,009 | 19,818 | 0.76 : 1 |
| 3.0.0 | 26,000 | 36,732 | 1.41 : 1 |
| 3.1.0 | 28,411 | 44,895 | 1.58 : 1 |
| HEAD | 29,258 | 47,730 | **1.63 : 1** |

Since 2.3.0: production **+12%**, tests **+141%** — tests grew **11× faster**.

**24% of the suite (10,911 lines across 55 files) is guards *about* the codebase** rather than
tests *of* it. The guard written on this branch, at 941 lines, is now the largest in the repo —
324 lines of it are analysis machinery, not assertions.

### 9. The triggering defect — testing H3

- `Role.any()` introduced **2025-10-22**
- measured as a live exposure **2026-08-14** — ten months later
- **found by an external measurement, not by our tooling**

The guards that would plausibly catch it were mostly written in July–August 2026, *after* the
defect and *because* of neighbouring incidents.

---

## Verdicts

### H1 — SURVIVED, with a correction

The loop is real and measurable: four of five round-3 findings were introduced by an earlier
fix, and 53–66% of the two most-audited files is now rework rather than original work.

**But the audits are not manufacturing false work.** Every finding in all four audits was
verified by execution — a mutation applied, a probe run, a value observed. The most serious
(an open container vanishing from a security report) was reproduced live before it was fixed.

The correct statement is narrower and worse: *fix commits carry the same defect density as
original commits (~0.6–0.8 per 100 lines), and each round writes 350–600 lines. So each round
mechanically produces three to four new defects, and nothing audits a fix except the next
audit.* The audits are not the generator. **Writing 500 lines per round is.**

### H2 — FALSIFIED

The findings cluster in the newest and structurally simplest code. `permissions.py` is below
the repo's 90th percentile on lines, nesting and function length, and holds a third of all
findings. `file.py` — 2,847 lines, nesting 6, a registered god class — produced almost none.
Only 6% of findings were in pre-existing code at all. If accumulated mess were the cause, the
mess is where the findings would be.

### H3 — FALSIFIED for this session

True of the triggering defect: a ten-month-old grant, found late. **But it was found by an
external measurement, not by our tooling** — so it is not evidence that the tools improved.
And 94% of the 54 findings were in code written this week. Better tools are not surfacing old
problems here; they are surfacing new ones, immediately.

### H4 — CONTESTED

The growth is real and extreme: tests up 141% against production's 12%, a quarter of the suite
now guards, this branch at 3.3:1, and the largest guard in the repo is one-third analysis
machinery. Two tests were found asserting the opposite of their own names, and one would have
caused a correct fix to be reverted.

The half about defect density is **withdrawn**. It read: *test code carries 1.0 findings per
100 lines against production's 3.9, so the test suite is not where the defects are.*

**That measured detection, not quality**, and the correction came from a peer session working
across the platform. A guard that cannot fire produces zero findings *because nothing detects
it* — so a low findings-per-line in test code is exactly the signature a suite full of vacuous
guards would produce. Nine of eighteen repositories on this platform have a recorded instance of
a guard that could not fire; this one is among them.

Tested here, by splitting test-code findings by detection method:

| how the finding was made | audits run | test-code findings |
|---|---:|---:|
| reading the code (`/code-review max`) | 3 | 12 |
| executing mutations against it | 1 | **9** |

**One mutation pass found nearly as many defects in the test suite as three reading passes
found in total** — 2.2× the yield per audit. The 1.0-per-100 figure is a floor set by the
method, not a property of the code. It cannot be used to argue the test suite is sound.

What survives of H4 is the growth, which is measured and not in doubt: +141% against
production's +12%, a quarter of the suite now guards, this branch at 3.3:1, and the largest
guard in the repo one-third analysis machinery. Two tests were found asserting the opposite of
their own names, and one would have caused a correct fix to be reverted.

**Known gap, stated rather than closed:** the guard has changed once since the mutation audit
ran, so its current form has never been executed against. That is recorded below as work
deliberately not done.

---

## What the four verdicts amount to

H1 and H4 are one mechanism seen from two angles:

> A finding arrives. It is fixed by writing code. Most of that code is test code (3.3:1).
> All of it carries the normal defect rate. Nothing audits it except the next round.

That is a loop with no damping term. It does not converge because nothing in it gets smaller.

Two things follow, and neither is "audit harder":

**The production surface is the thing to shrink.** 586 lines of new production code drew 23
findings. It is a security instrument written in one week and rewritten four times. The defect
density there — 3.9 per 100 lines — is the number that matters, and it is high because the code
is new, not because it is bad.

**A fix is not a safer kind of code, and the process treats it as one.** Every round shipped its
fixes with a full suite run and no second review of the fix itself. The one time an independent
auditor was pointed at the fixes rather than the original — the guard-mutation audit — it found
nine survivors in an afternoon, including two guards that were pure theatre.

The single highest-value change is therefore not more rounds. It is that **the thing which
reviews a fix must not be the thing that wrote it**, which the guard-mutation audit already
demonstrated and which cost one subagent invocation.


---

## What is deliberately NOT being fixed

Added 2026-08-24, after a peer session pointed out that this is the only intervention on the
platform with a recorded outcome. views-postprocessing declined five surviving mutations in
writing — four routed to the story that owned the scope, one declared infinite regress — and
recorded the result as *"prevented a sixth round."* It is a paragraph, not a system.

**1. The AST guard is not being rewritten a fourth time.** It is 941 lines guarding a 467-line
module, one third of it analysis machinery, 66% rework. views-faoapi's Action 11 reads *"do not
build monitoring larger than the thing it monitors"*, recorded after a 787-line rehearsal
harness produced six of fifteen findings in one review round. The right change is probably to
delete the dataflow resolution entirely and fail on anything that is not a literal — pushing the
proof to a behavioural test at each call site, which is what actually caught things (`A1`, `A2`,
`A4`, `C1`, `C2`, `C3` were all caught by the ~40-line behavioural tests, not by the analyser).
**Trigger:** the next time this guard reports a false positive, or the next time it needs a
change to accommodate a legitimate call shape. Not before — a rewrite now is round four.

**2. The current guard is not being mutation-audited again.** One commit landed on it after the
audit. Re-running is cheap and diagnostic rather than additive, but the result would be a list
of survivors, and fixing survivors is the loop. **Trigger:** before this branch merges, or when
`ship-it` runs guard mode automatically — whichever comes first. Recorded so the gap is known
rather than assumed closed.

**3. `views-crafdapi` still hardcodes `Role.any()`.** Filed as views-crafdapi#123 and
deliberately not patched from here: a second repository, a one-line change with a security
consequence, and `CLAUDE.md` puts a two-repository change in front of the operator.

**4. The probe does not issue an unauthenticated request.** It reads declared configuration with
an API key. `views-models/tools/credentials/close_resource_permissions.py` does probe anonymously
and is what actually measured C-83. The two tools do not reference each other. **Trigger:** any
decision to treat this probe's clean verdict as evidence about reachability rather than about
declared permissions.

**5. `permissions_exit_code` returns one number for two independent facts.** Open-and-incomplete
is reported in the text and collapses to `1` in the status code. A third code would be more
honest and would break every caller that keys on the published 0/1/2 table. **Trigger:** a second
consumer of the exit code.


---

## Addendum, 2026-08-24 — the open question, closed as unanswerable from this sample

The report left one distinction open: *are fixes inherently as defective as any other code,
or is new code defective and all of this happened to be new?* A peer session searched all 16
platform registers — 1,307 entries — for any entry attributing a defect to a specific prior
fix. **It found two**, both in views-frames. Registers here record what a defect is and where
it lives, never that it arrived in the commit that fixed something else. So the question cannot
be answered retrospectively from the corpus at all.

Their proposed discriminator was cheap and correct: split findings per 100 lines by the age of
the **lines touched**, not the age of the file. Run here:

**Exposure — what this branch actually touched:**

| | lines |
|---|---:|
| added | 3,562 |
| pre-existing lines modified | **21** |
| …of which production code | **10** |

**Age of the line each traced finding landed in:**

| finding | line introduced | found | age |
|---|---|---|---:|
| R2 `or []` in `_read_container` | 2026-08-14 | 2026-08-22 | 8 days |
| R2 `users` excluded | 2026-08-14 | 2026-08-22 | 8 days |
| R2 INCOMPLETE hides OPEN | 2026-08-14 | 2026-08-22 | 8 days |
| R2 discards permissions | 2026-07-31 | 2026-08-22 | 22 days |
| R3 unparseable deletes container | 2026-08-22 | 2026-08-24 | 2 days |
| R3 `or []` in `_read_items` | 2026-08-22 | 2026-08-24 | 2 days |
| R3 renderer false claim | 2026-08-22 | 2026-08-24 | 2 days |
| R3 refuses `[]` | 2026-08-22 | 2026-08-24 | 2 days |
| R3 exit code | 2026-08-14 | 2026-08-24 | 10 days |

Range 2–22 days, median 8. **Every finding landed in a line younger than three weeks.**

**The measurement cannot discriminate, and the reason is stronger than confounding.** The
independent variable has no variance: the branch modified ten pre-existing production lines,
so there is essentially no old-code exposure to compute a density against. The three findings
recorded as "pre-existing" are not lines this branch touched and got wrong — they are a sibling
repository's vendored copy, a function that was never tested, and an allowlist structure. None
is a case of an old line being modified and proving defective.

So the correct status of the distinction is **untested, and untestable from this sample** —
not "confounded but leaning". C-303's claim stands as measured (fix commits carry the same
defect density as the original commit, 0.6–0.8 per 100 lines) and does **not** extend to a claim
about old code, because no old code was in the sample.

**The one extra field that would settle it**, on a future round that touches real legacy
surface: `git blame` each changed hunk at the time of the fix and bucket findings by the age of
the lines modified. If fixes are inherently defective, density holds constant across blame-age.
If newness drives it, density falls with age. One column, no new instrument.
