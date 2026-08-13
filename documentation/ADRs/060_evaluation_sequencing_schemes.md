# ADR-060: A config declares how its evaluation is sequenced, and gets that scheme's contract

**Status:** Implemented
**Date:** 2026-08-13
**Implementation Date:** 2026-08-13
**Deciders:** Simon, VIEWS platform team
**Epic:** #458 · **Story:** #460 · **Supersedes the approach in:** #328

---

## Scope of this decision

**One question:** when a model's evaluation is not sequenced the way the platform assumes,
what should the config sniffer do?

Not in scope: what the schemes themselves *are* (the engines own that), and whether
`MAX_SHIFT_COUNT` should change (it should not, and this ADR removes the pressure to
change it for the wrong reason).

## Context

`CoreConfigSniffer._check_evaluation_contract` asserted, for every non-forecasting run:

```python
test_len     = test_end - test_start + 1
expected_len = time_steps + MAX_SHIFT_COUNT   # 36 + 12 = 48
if test_len != expected_len:
    raise NotImplementedError(...)
```

That is a true statement about **rolling-origin** evaluation: origins advance by
`rolling_origin_stride` across a window sized to the forecast horizon plus the shift
allowance. It was applied to every config, because until recently every config sequenced
that way.

views-impact does not. It consumes the test window in blocks:

```python
# views-impact  views_impact/manager/model.py:169-176
horizon = self.configs["output_chunk_length"]
return test_length // horizon + 1
```

Its test partition is not a function of `time_steps` at all, so the check refused a
correct config — correctly, for a scheme nobody had told it about. The check's own error
message already conceded the assumption: *"Update MAX_SHIFT_COUNT in
core_config_sniffer.py when ready."*

**PR #328 responded by commenting the check out.** That disabled it for every model on
the platform to accommodate one, and is the approach this ADR replaces.

## Decision

A config may declare `evaluation_sequencing`. The contract check applies the contract
belonging to the scheme declared.

| scheme | contract |
|---|---|
| `rolling_origin` | `test_len == time_steps + MAX_SHIFT_COUNT`, then the existing sequence accounting |
| `horizon_chunks` | `output_chunk_length` present, a positive integer, and no longer than the test window |

The partition-overlap check belongs to **no** scheme and runs for all of them. Declaring a
scheme must not become a way to evade an unrelated rule.

### The default is the strict scheme

An absent `evaluation_sequencing` means `rolling_origin`.

Two reasons, and the second is the load-bearing one:

1. Every config written before this ADR is rolling-origin. Any other default would change
   the behaviour of every existing model at once.
2. **An unstated scheme should get the tighter contract.** A wrongly-refused config is
   loud and fixable in a minute; a wrongly-accepted one is a run that reports a number
   nobody checked. Same direction as ADR-059's identifying-by-default: when the safe
   answer and the convenient answer differ, default to safe.

### An exempted scheme is not an unchecked scheme

`horizon_chunks` is exempt from the rolling-origin *length* rule and from nothing else.
It has invariants of its own and they are enforced:

- **`output_chunk_length` is required.** Without a block size the scheme is not
  sequenceable, and a config that cannot be sequenced is not a config.
- **It must be a positive integer.** `bool` is excluded explicitly, because `True > 0` and
  `isinstance(True, int)` — the same trap ADR-059 records for `provenance_version`.
- **It must fit inside the test window.** A horizon longer than the window means the model
  predicts further than anything can score, and the consumer's own `test_len // horizon +
  1` yields a single partial block — a number that looks like an evaluation and is not.

**A remainder is allowed**, and logged rather than refused. The consumer's `+ 1`
deliberately covers a partial final block. Refusing it would be this repo inventing a rule
the scheme does not have, which is a different way of being wrong about someone else's
work.

This is the part most likely to erode. A branch added to exempt a scheme, that then
validates nothing, fails exactly the way #328 failed — only quietly, and with a test suite
that stays green. `test_a_horizon_chunked_config_escapes_the_rolling_origin_length_rule`
and the four refusal tests beside it exist to keep the exemption narrow.

## Alternatives rejected

**Comment out or delete the check** (#328's approach). Disables it for every model to
accommodate one, and removes the only thing verifying the partition shape of the
forty-odd models that *are* rolling-origin.

**Raise `MAX_SHIFT_COUNT` until impact's window fits.** The check's error message suggests
this, and it is wrong here: impact's window is not `time_steps + shift` for any shift. It
would make the constant meaningless for the models it governs in order to accommodate a
model it does not.

**Infer the scheme from the config's shape** — e.g. treat the presence of
`output_chunk_length` as meaning chunked. Rejected under ADR-040: no semantic inference. A
model that happens to carry a horizon key for another reason would silently lose its
length contract, and nothing would say so.

**A per-model opt-out flag** (`skip_evaluation_contract: true`). Rejected because it says
what to skip rather than what is true. A declared scheme can be checked; an opt-out can
only be trusted.

## Consequences

- Existing configs are unaffected. Every one of the 84 pre-existing sniffer tests passes
  unchanged, including the two `test_len` tests #328 broke.
- views-impact can declare `horizon_chunks` and be checked rather than exempted.
- A third scheme means a third entry in `SUPPORTED_EVALUATION_SEQUENCING` and a third
  contract method. That is the intended cost: adding a scheme should require stating what
  is true of it.
- `evaluation_sequencing` is a new optional config key. It is not in
  `MANDATORY_KEYS_MODEL` and should not become so — mandatory would force every model to
  restate the default.

## Related

- **ADR-041** (sniffer pattern) — this is a check, and follows its rules: state-bearing,
  read-only, fail loud
- **ADR-040** (no semantic inference) — why the scheme is declared rather than guessed
- **ADR-059** — the identifying-by-default precedent this borrows, and the `bool`/`int`
  trap it records
- **ADR-009** (boundary contracts) — `evaluation_sequencing` is optional and belongs in
  its table if that table is ever made exhaustive
- Issues **#458** (epic), **#460** (this story), **#328** (the approach replaced),
  **views-impact#5**
