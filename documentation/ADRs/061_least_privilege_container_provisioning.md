# ADR-061: Containers are provisioned least-privilege; widening is stated by the caller

**Status:** Implemented
**Date:** 2026-08-14
**Implementation Date:** 2026-08-14
**Deciders:** Simon, VIEWS platform team
**Concern:** C-292 · **Supersedes the behaviour carried through:** #331

---

## In plain terms, before the detail

We send forecast files to partners through a hosted service called Appwrite. Each file
gets an index card describing it, and those cards live in a *collection* — think of a card
drawer next to the folder holding the files.

**Our setup code locked the folders and left the card drawers open to the public.** Anyone
who knew our project's ID could read, change or delete any partner's cards. This changes
new drawers to be locked, and adds a command that tells you whether an existing one is.

It does **not** change any drawer that already exists. Those have to be looked at.

Terms used below: a *container* is a folder or a card drawer. A *call site* is the place
in the code where one function asks another to do something. *AST-walking* means a test
reads the code's structure to find every place a rule could be broken, rather than
searching for particular words.

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

`Role.any()` is **anyone**, including unauthenticated callers holding only the project id.
`document_security=False` means those grants govern every document with no per-document
narrowing. So every collection this tool created was readable, writable and **deletable**
by anyone who could reach the endpoint.

**Two things that sentence depends on, stated rather than assumed.**

*Who can reach the endpoint.* Our default is `https://cloud.appwrite.io/v1`
(`modules/appwrite/file.py:41`) — Appwrite's hosted service, on the public internet, with
no network restriction of ours in front of it. So "anyone who could reach the endpoint"
means anyone, from anywhere. If the platform ever moves to a self-hosted or
network-restricted instance, this sentence needs re-checking rather than re-reading.

*Why the project id is not a credential.* Not because it appears in browser code — **there
is no client-side code anywhere on this platform**; no `package.json` and no JS, TS or HTML
touching Appwrite in any of the 24 repos. It is not a credential because it is carried as
an ordinary environment variable across five repos, CI configuration and operator shells,
and has never been handled as a secret. The API key is the secret; the project id
identifies which project the key is for.

Three things made this worse than a bad default.

**`ensure_bucket`, in the same class, has always defaulted to `permissions=[]`** with
`file_security=True`, and the CLI never overrides either. One command produced a locked
bucket and an open collection. Nobody chose that; the two halves simply never met.

**It was not CLI-only.** Before #331 (2026-07-31), `create_metadata_collection_if_not_exists`
was called from `upload_file_with_metadata`, `upload_file_from_bytes_with_metadata` and
`check_file_exists_by_hash` — the ordinary delivery path. The grant dates to commit
`f0351d3` ("add aw+tests", 2025-10-22), which introduced the Appwrite module. For roughly nine months, **an ordinary delivery to a new partner created an
open collection automatically.**

**Nothing could see it.** The word `permission` appeared in no test in this repo, and
`modules/appwrite/audit/` — the package whose entire job is reading this seam — never read
or reported a permission. We shipped something that created open containers and owned
nothing that could tell us whether any live container was open.

The provenance is #331: the grant was relocated verbatim under that PR's stated rule that
a relocation must not change behaviour. That rule was right and it is why this survived —
a faithful move carries a defect faithfully.

**ADR-046 has argued for least privilege throughout** — the phrase appears four times in
it, every one about API key *scopes* and none about container permissions. Two are quoted
here as representative: "create scopes, which blocked least privilege", and the §5
supersession note that auto-bucket creation "blocked least privilege platform-wide". The
count is reproducible with `grep -ci 'least privilege' documentation/ADRs/046*.md`. The
ADR and the code have disagreed since #331, in the same part of the system, about the same
principle.

## Decision

**A container this repo provisions grants nothing by default. Widening is an argument the
caller passes, and a reason the caller records.**

`ensure_collection` gains `permissions: List[str] = None`, applied as
`permissions=[] if permissions is None else list(permissions)` — the same sentinel shape
`ensure_bucket` already uses, so the two methods now have one posture.

`Permission` and `Role` are **not imported** by `provisioning.py`. A caller wanting a wider
grant constructs it at the call site, where the reason for it is visible.

**That import removal is a convention, not the enforcement — do not mistake one for the
other.** Nothing inspects the import list, and the import can be restored in a single line.
What actually prevents recurrence is
`tests/test_no_container_is_provisioned_open.py`, which AST-walks every
`create_*(permissions=...)` call in `views_pipeline_core/` and `tools/` and fails on any
grant to `any`. If you are looking for the thing that will stop you, that is the thing.
(An earlier revision of this ADR, and of that test's own docstring, credited the missing
import with the protection — a claim the evidence made consistent but did not support.
Corrected 2026-08-21; recorded as an instance of C-273.)

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
- **How you would know this decision was wrong.** It rests on one belief about the
  substrate: *a server API key bypasses container permissions*. If that is false,
  `permissions=[]` removes access that partners depend on. The symptom would be
  **deliveries failing with authorization errors after 3.1.2** — and if you are reading
  this because that is happening, this decision is the first thing to check, and reverting
  is a one-line change to the default plus a re-provision. The belief is argued from our
  own system rather than from documentation (see above), and every consumer was swept for
  anonymous access across all repos and languages with zero hits — but it remains a belief
  about someone else's platform, and it is written down here so the symptom has a path back
  to the cause.
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

**Tighten live containers in the same change.** Rejected on the boundary this repo's
`CLAUDE.md` draws between decisions I make and decisions the operator makes: anything
touching an external party is theirs. Altering a live container changes what a partner can
reach, so it belongs to the operator — and only after looking at what is actually there.

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
- **C-232 / C-244 / C-249** — three past incidents in the risk register, all the same
  shape: a check that could not tell "there is nothing there" from "I was not able to
  look", and reported the reassuring one. They are why the probe has three outcomes rather
  than two
- **C-218** — the belief-mirroring suite at this seam; why the probe's parser is pinned
  against real SDK output rather than a hand-written string
- **#331** (the relocation that carried it), **#473** (the CLI flags that made it reachable)
