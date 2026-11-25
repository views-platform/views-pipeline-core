# ADR-071: A failed Appwrite database call carries a classified code, not the raw server type

**Status:** Accepted
**Date:** 2026-11-25
**Deciders:** VIEWS platform team (engineering decision under #603)
**Concerns:** C-235 · **Issues:** #603, epic #599

---

## Context

A failed database call returns `OperationResult(success=False, code=...)` in-band (ADR-046 §b),
and under ADR-047 the run exits 0, so the code is often the only signal. Before #603 the code was
the server's `type` string. Three different answers could look alike to a caller: a key the
server refused, a container that does not exist, and a listing with no match. Under the
TablesDB rename (#599) a key minted in a new Appwrite project is refused on every old
`databases.*` call; read as "no duplicate", that uploads a second copy.

## Decision

1. **Every database `except AppwriteException` sets its code with
   `modules/appwrite/file.py::database_error_code`.** HTTP 401/403 → `UNAUTHORIZED`. A 404 whose
   type is `database_not_found`, `collection_not_found` or `table_not_found` →
   `CONTAINER_NOT_FOUND`. Anything else keeps the server's type. Storage calls keep the
   server's type; the classifier is for databases only.
2. **`NOT_FOUND` means one thing:** a listing answered and matched nothing. No exception is ever
   mapped to it.
3. **No branch matches error message text.** Branch on the code.
4. Where the delivery path logs a database failure it uses `database_failure`, which names the
   database, collection, HTTP status and server type, so an expired key and a missing scope
   (both 401) can be told apart in the log.

## Consequences

- A refused key stops the hash lookup instead of reading as absence (pinned by
  `tests/test_modules/test_database_error_classes.py`).
- A new database call site written with `code=e.type` violates this ADR. No structural test
  catches that yet; the TablesDB stories (#604, #605) rewrite these sites and must keep the rule.
- Message-text matches remain at the provisioner's "already exists" branches (HTTP 409) and in
  `get_bucket`; they are outside this rule's first cut and are named on C-235.
- views-faoapi#596/#597 and views-crafdapi#259/#260 propose copying this vocabulary into the
  serving APIs.
