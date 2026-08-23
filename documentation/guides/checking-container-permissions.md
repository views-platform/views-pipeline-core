# Checking who can reach our partner storage

**Who this is for:** anyone with access to the Appwrite console or the platform's
environment variables. No engineering background assumed.

**Why it exists:** until August 2026 our setup code created partner *index-card drawers*
open to the public while locking the *file folders* beside them. **This was not
theoretical.** On 14 August 2026 an anonymous request carrying only our project ID — no
password, no login — returned every row of two production drawers: 111 from the FAO one
and 461 from the internal one. Both were closed the same day. A follow-up on the 15th found
the FAO drawer untouched and eleven altered cards in the internal one, each of which was
then checked against its own record and explained. New drawers are now locked by default (ADR-061).

So: the known ones have been looked at, and what was found is written above. The source entry is careful to say that is **not** the same as "verified clean" — one residual cannot be ruled out even in principle. This guide is for the next one, and
for the next time someone asks.

---

## The one-minute version

Two ways. Either is fine; the console is faster the first time.

**In the browser:** Appwrite console → **Databases** → your metadata database → click a
collection → **Settings** → **Permissions**. You are looking for a row labelled **`Any`**.

**From a terminal**, with the platform environment loaded:

```bash
python -m views_pipeline_core.modules.appwrite.audit --permissions --target forecasts
python -m views_pipeline_core.modules.appwrite.audit --permissions --target unfao
```

For a shelf that is not one of those two — `crafd` is the current example — give **both**
halves of the pair:

```bash
python -m views_pipeline_core.modules.appwrite.audit --permissions \
    --bucket crafd_bucket --collection crafd
```

Both flags or neither. A bucket and its card drawer are one shelf, and giving half the
pair leaves the other half pointing at production. The command refuses rather than
guessing.

*(An earlier version of this guide said `--collection-name`. That flag belongs to the
provisioning command, not this one, and the audit command rejects it — exiting 2, which
this same guide teaches means "could not determine". Corrected 2026-08-22.)*

---

## Reading the result

The command prints each container, what it permits, and a verdict at the end.

### GOOD — what you want to see

```
collection crafd
  documentSecurity: False
  permissions: [] — reachable only with an API key

VERDICT: no container grants anything to an unauthenticated role, and everything
this tool can check was checked.
```

Exit code **0**. `permissions: []` is not "nobody can use it" — our own system key still
works, and that key is what the pipeline and the partner-facing service both already use.
It means *only* the key.

### BAD — this is a problem

```
collection crafd
  documentSecurity: False
  permission: read("any")
  permission: delete("any")
  >> OPEN TO ANYONE: read, delete
  >> anyone with the project id can delete every item here. The project id is not a secret.

VERDICT: OPEN — 1 container(s): crafd
```

Exit code **1**. `any` means anyone at all, including someone who has never logged in.
`read` alone is a disclosure. `create`, `update` or `delete` mean someone could alter or
destroy a partner's records.

### NOT AN ALL-CLEAR — do not read this as fine

```
COULD NOT DETERMINE (1):
  - get_collection(crafd) failed: missing scope (collections.read) — permissions UNKNOWN,
    which is not the same as locked down

VERDICT: INCOMPLETE — 1 thing(s) could not be determined, listed at the top.
This is not an all-clear.
```

Exit code **2**. **Do not read this as "fine".** The tool could not look. Usually the key
lacks a read scope. Get a key that can read collections and run it again.

This distinction is deliberate: a security check that says "nothing open" when it was
merely blocked from looking is worse than no check.

---

## If something is open

**Stop and tell whoever owns the partner relationship first.** This is data belonging to
an external counterparty, and the decision to change it — and whether to say anything to
them — is not a technical one.

The tool will not fix it, on purpose. It has no `--fix` and cannot write. Remediation is
done deliberately, in the console:

1. Console → the collection → **Settings** → **Permissions**
2. Delete the `Any` row
3. Re-run the check and confirm the verdict is now clean
4. Check the partner's delivery still works — it should, because deliveries authenticate
   with the API key rather than a public grant, but confirm rather than assume

Worth recording afterwards: which container, which permissions, how long they were that
way, and whether anything suggests they were used.

---

## Two things that will confuse you

**Buckets look different from collections.** A bucket's flag is `fileSecurity`, a
collection's is `documentSecurity`. Both mean roughly "can individual items carry their own
permissions". **When one of them is ON and the container grants nothing, per-item permissions are the
only thing that can open it** — so the tool reads the individual files and documents too,
and says so. When it is OFF, per-item grants are ignored by Appwrite and the tool does not
read them. When the tool cannot tell which, it says that instead of guessing.

**`users` IS effectively public — this section said the opposite until 2026-08-24.** A row
saying `read("users")`, or `read("users/unverified")`, is flagged as open and exits 1. The
reason is in Appwrite's own documentation: that role grants *"any authenticated **or
anonymous** user"*, and anyone can open an anonymous session with just the project ID.

This guide previously told you the tool shows `users` without flagging it. If you saw a
`users` row flagged and came here to check whether the tool was over-reporting: **it was
not.** Treat it exactly like `Any`.

**What is genuinely narrower:** a row naming a team or a specific user — `team:analysts`,
`user:abc`. Those are real narrowings, and the tool shows them without flagging them.

---

## Related

- **ADR-061** — the decision to lock new containers by default, and what would show it to
  be wrong
- **`AppwriteProvisioner.md`** — the class that creates containers
- **C-292** — the register entry; the "existing containers" half is still open
