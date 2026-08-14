# ADR-061: Containers are provisioned least-privilege; widening is stated by the caller

**Status:** Implemented
**Date:** 2026-08-14
**Implementation Date:** 2026-08-14
**Deciders:** Simon, VIEWS platform team
**Concern:** C-292 · **Supersedes the behaviour carried through:** #331

---

## Scope of this decision

**One question:** what access should a container created by this repo grant by default?

Not in scope: what any *already-provisioned* container should permit (an operator action
against a partner-facing store, and not a code decision), API key scope narrowing, and
whether `document_security` should be `True`.

## Context

`AppwriteProvisioner.ensure_collection` created every metadata collection with:

```python
permissions=[
    Permission.read(Role.any()),
    Permission.create(Role.any()),
    Permission.update(Role.any()),
    Permission.delete(Role.any()),
],
document_security=False,
```

`Role.any()` is **anyone**, including unauthenticated callers holding only the project id
— which is not a secret and appears in client-side configuration by design.
`document_security=False` means those grants govern every document with no per-document
narrowing. So every collection this tool created was readable, writable and **deletable**
by anyone who could reach the endpoint.

Three things made this worse than a bad default.

**`ensure_bucket`, in the same class, has always defaulted to `permissions=[]`** with
`file_security=True`, and the CLI never overrides either. One command produced a locked
bucket and an open collection. Nobody chose that; the two halves simply never met.

**It was not CLI-only.** Before #331 (2026-07-31), `create_metadata_collection_if_not_exists`
was called from `upload_file_with_metadata`, `upload_file_from_bytes_with_metadata` and
`check_file_exists_by_hash` — the ordinary delivery path. The grant dates to `f0351d3`
(2025-10-22). For roughly nine months, **an ordinary delivery to a new partner created an
open collection automatically.**

**Nothing could see it.** The word `permission` appeared in no test in this repo, and
`modules/appwrite/audit/` — the package whose entire job is reading this seam — never read
or reported a permission. We shipped something that created open containers and owned
nothing that could tell us whether any live container was open.

The provenance is #331: the grant was relocated verbatim under that PR's stated rule that
a relocation must not change behaviour. That rule was right and it is why this survived —
a faithful move carries a defect faithfully.

**ADR-046 has argued for least privilege throughout**, four times, about API key scopes
(§ "create scopes, which blocked least privilege", and the §5 supersession note: auto-bucket
creation "blocked least privilege platform-wide"). The ADR and the code have disagreed
since #331, in the same seam, about the same principle.

## Decision

**A container this repo provisions grants nothing by default. Widening is an argument the
caller passes, and a reason the caller records.**

`ensure_collection` gains `permissions: List[str] = None`, applied as
`permissions=[] if permissions is None else list(permissions)` — the same sentinel shape
`ensure_bucket` already uses, so the two methods now have one posture.

`Permission` and `Role` are **not imported** by `provisioning.py`. A caller wanting a wider
grant constructs it at the call site, where the reason for it is visible.

### An empty list is not "no access"

This is the point most likely to be misread later. A **server API key bypasses container
permissions**, and every consumer on this platform authenticates with one — verified
across views-faoapi (`managers/appwrite/auth.py`, `client.set_key`), views-crafdapi,
views-appwrite, views-models and views-postprocessing, swept for `set_session` /
`create_anonymous_session` / `set_jwt` with zero hits. `Role.any()` was load-bearing for
nothing.

The empirical confirmation is in our own system rather than in documentation: **buckets
have always been created with `permissions=[]` and deliveries to them work today.** Same
project, same key, same endpoint.

### `document_security` stays `False`

Deliberate, and not an oversight. With no container-level grants there is nothing for
per-document permissions to narrow, and flipping it would change how every existing
document is evaluated for no benefit anyone can name. Revisit it if per-document access
is ever actually wanted.

## Consequences

- **Nothing already provisioned changes.** Appwrite applies these grants at creation, and
  the existing-collection branch does not re-apply them. Whether live containers are open
  is a separate question, answered by the probe below and remediated — if needed — by an
  operator.
- **New partner onboarding is closed by default**, which matters now: #473 shipped the CLI
  flags that make creating a partner collection easy.
- **We can now see.** `python -m views_pipeline_core.modules.appwrite.audit --permissions`
  reports what a shelf's collection and bucket actually permit. It is read-only and has no
  `--fix`, deliberately: a mutating tool pointed at partner-facing production is what
  `tools/wipe_fao_shelf.py` and C-249/C-250 record going wrong.
- **The probe distinguishes three outcomes**, not two — absent / read / **unreadable**. A
  container whose permissions could not be read exits 2 and renders `INCOMPLETE`. A
  security diagnostic that reported "nothing open" when it was merely not allowed to look
  would be wrong in the reassuring direction.
- **A guard enforces this**, deriving rather than listing: `tests/test_no_container_is_provisioned_open.py`
  AST-walks every `create_*(permissions=...)` call in `views_pipeline_core/` and `tools/`.
  A grant it cannot statically resolve is reported as **unknown**, not passed over.
- Widening now costs a caller an explicit argument. That is the intended cost.

## Alternatives rejected

**Leave it and document it.** What C-292 did yesterday, on the reasonable ground that
partner-facing permissions should not be decided inside a hotfix. The delivery-path history
found the next day changes the calculus: this was not a latent CLI hazard, it ran
automatically for nine months.

**Grant `Role.users()` instead.** Narrower, and still wrong — it grants every authenticated
user of the project, which is not the same as "the pipeline". With key auth there is no
reason to grant a role at all.

**Tighten live containers in the same change.** Rejected on the CLAUDE.md boundary: altering
a partner-facing store touches an external party and belongs to the operator, after looking.

**Delete the parameter and hardcode `[]`.** Simpler, and it removes the ability to widen
deliberately — so the next person needing a wider grant edits the library instead of stating
their intent at the call site. `test_the_collection_default_is_least_privilege` pins the
parameter for this reason.

## Related

- **C-292** — the concern this resolves for new containers; live containers remain open
  until inspected
- **ADR-046** — the Appwrite storage integration, which argued least privilege for the key
  while the code granted `Role.any()` on the container
- **ADR-047** — destination authority; local disk is authoritative, Appwrite is secondary
- **C-232 / C-244 / C-249** — failed read as absence, verdict conflation, a renderer stating
  a conclusion without consulting the record. All three shape the probe's three outcomes
- **C-218** — the belief-mirroring suite at this seam; why the probe's parser is pinned
  against real SDK output rather than a hand-written string
- **#331** (the relocation that carried it), **#473** (the CLI flags that made it reachable)
