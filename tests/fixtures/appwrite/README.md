# Recorded Appwrite response shapes

Fixtures captured from a **live Appwrite instance**, following the precedent of
`tests/fixtures/wire_contract/` and `feature_frame_contract/`.

## Why these exist

Almost every test at the Appwrite seam checks our code against a fake we wrote ourselves,
so it asks *"does the code do what I think Appwrite does?"* and never *"does the code do
what Appwrite actually does?"* When the belief is wrong, the test does not merely miss the
bug — **it certifies it**.

That is register **C-218**, and it proved itself on 2026-08-01: `reconcile.py` shipped
with nine green tests whose fake returned every metadata document in one call, and the
tool reported **436 phantom orphan files against production**.

A fixture written by hand from Appwrite's documentation would not fix this. It would
still be an assertion about the substrate that no substrate confirmed — the same failure,
rebuilt with more steps. **The value is entirely that the shape came from the service.**

## Files

| File | What it pins |
|---|---|
| `list_documents_shape.json` | What `list_documents` returns **with and without** a `Query.limit` — the fact C-241 turned on |

## How to (re)capture

    python tests/fixtures/appwrite/capture_list_documents.py

Read-only: `list_documents` is the only Appwrite method it calls. No writes, no deletes,
no provisioning. It prints what it captured before writing, so the diff can be reviewed
before it is committed.

## What is recorded — and what never is

**Recorded:** how many documents came back, what `total` reported, which field names
exist, and each value's type and size class.

**Never recorded:** any field value. Not filenames, ids, hashes, owners or timestamps —
every value is replaced by a type descriptor before anything is written, and the redactor
is pinned by `test_appwrite_recorded_shape.py::TestTheRedactorCannotLeak`.

**Coordinates are fingerprinted, not published.** PLATFORM-001 §4 forbids baking registry
coordinates into the repo, so provenance identifies the source by a truncated SHA-256 of
the project and collection ids. The same project always yields the same fingerprint, so
captures remain comparable, without an address appearing in git.

## Staleness

This is a snapshot. It catches **our** regressions, not Appwrite's changes — if the
service alters its response shape, this fixture will happily keep asserting the old one.
Catching *their* changes is the job of `tests/test_modules/test_appwrite_sdk_contract.py`,
which drives the real installed SDK. The two tiers are complementary and neither replaces
the other.
