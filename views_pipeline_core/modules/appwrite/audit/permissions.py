"""What a container actually permits, read from the live seam. C-292.

Read-only, like the rest of this package. It reads `get_collection` and `get_bucket`,
reports the `$permissions` array verbatim, and says which verbs — if any — are granted
to a role reachable without authenticating (`any` or `guests`).

## Why this exists

`provisioning.py` creates metadata collections with
`Permission.{read,create,update,delete}(Role.any())` and `document_security=False`, while
`ensure_bucket` in the same file creates buckets with `permissions=[]`. One command, two
postures, and until this module **nothing in the package could read a permission at all** —
so the question "is any live container open?" had no answer we could produce.

The grant is not hypothetical. Before #331 (2026-07-31),
`create_metadata_collection_if_not_exists` was called from `upload_file_with_metadata`,
`upload_file_from_bytes_with_metadata` and `check_file_exists_by_hash` — the ordinary
delivery path. The grant has existed since 2025-10-22. Any collection first written to by
a normal delivery in that window was created open, automatically, with nobody choosing it.

## What this module does NOT do

**It does not decide policy, and it cannot change anything.** Whether a partner's metadata
collection *should* permit `any` is a decision with an external blast radius, recorded as
C-292 and owned by the operator. This module reports what is there.

It also has no `--fix`. A mutating tool pointed at partner-facing production is exactly
what `tools/wipe_fao_shelf.py` and C-249/C-250 record going wrong; remediation is a
deliberate, separate act.

## Three outcomes, never two

`absent` / `read` / **`unreadable`**. A container whose permissions could not be read must
never render as "no open permissions found" — that is the failed-read-as-absence defect
this whole package exists to enumerate (C-232, C-244, C-249). `PermissionsReport.is_clean`
is false whenever `indeterminate` is non-empty, and `permissions_exit_code` returns 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

#: Appwrite renders a grant as `verb("role")` — verified against the installed SDK:
#: `Permission.read(Role.any())` is exactly `read("any")`. Whitespace is tolerated
#: because the string comes back off the wire, not from our own formatting.
_GRANT = re.compile(r'^\s*(\w+)\s*\(\s*"([^"]*)"\s*\)\s*$')

#: The roles reachable **without authenticating**. `any` is anyone at all; `guests` is,
#: in the SDK's own words, "any guest user without a session" — the same unauthenticated
#: population this module exists to detect. Treating only `any` as open let a container
#: granted to `guests` render as a full all-clear (found 2026-08-22).
#:
#: `users` is here too, and the reason it was not is worth recording: this module
#: excluded it on the stated grounds that it "means every signed-in user of the project".
#: The installed SDK's own docstring says otherwise — `Role.users()` *"Grants access to
#: any authenticated **or anonymous** user"* — and an anonymous session needs only the
#: project id, which this module says four times is not a secret. Excluding it was the
#: same error as excluding `guests`, one role over, justified by a claim the substrate
#: contradicts (C-218). Corrected 2026-08-22.
ROLES_MEANING_UNAUTHENTICATED = frozenset({"any", "guests", "users"})

#: Verbs that let a caller change or destroy partner-facing data. `read` on a partner's
#: metadata is a disclosure; these are an integrity loss, and the distinction is worth
#: keeping in the output rather than collapsing both into "open".
MUTATING_VERBS = frozenset({"create", "update", "delete", "write"})


def parse_grant(raw: str) -> Optional[tuple]:
    """`'read("any")'` -> `('read', 'any')`. `None` if it is not a grant we can read.

    Returning `None` rather than guessing matters: an unparseable permission string is
    an unknown, and an unknown must reach `indeterminate` rather than be dropped from
    the tally as though it granted nothing.
    """
    if not isinstance(raw, str):
        # A non-string element used to raise TypeError out of `re.match`. Because the
        # unparseable tally sat inside `_read_container`'s try, that raise deleted the
        # WHOLE container from the report — its readable `read("any")` grants included.
        # An open container became invisible, not merely demoted. Found 2026-08-24.
        return None
    match = _GRANT.match(raw)
    return (match.group(1).lower(), match.group(2).lower()) if match else None


def role_is_unauthenticated(role: str) -> bool:
    """Is this role reachable without authenticating?

    Matches on the leading segment. Appwrite's roles take a `role/status` form —
    `Role.users("unverified")` emits `users/unverified`, verified against the installed
    SDK — and an exact-string test read that as an unknown, narrow role: clean, exit 0.
    That is the `guests` miss one spelling over, and it is the spelling a widening caller
    is most likely to write, because the status argument is the reason to call
    `Role.users()` with an argument at all.
    """
    return role.split("/", 1)[0] in ROLES_MEANING_UNAUTHENTICATED


@dataclass
class ContainerPermissions:
    """One container's declared access, as read.

    `security_flag` is Appwrite's `documentSecurity` (collections) or `fileSecurity`
    (buckets). With it `False`, the container-level grants below govern every item and
    nothing can narrow them per document — so `False` alongside an `any` grant is the
    widest possible state, not a detail.
    """

    kind: str  # "collection" | "bucket"
    container_id: str
    grants: List[str] = field(default_factory=list)
    security_flag: Optional[bool] = None
    unparseable: List[str] = field(default_factory=list)
    #: `(item_id, verbs)` for individual documents or files carrying their own grant to an
    #: unauthenticated role. Only populated when `security_flag` is on, because that is
    #: the only state in which per-item grants take effect.
    open_items: List[tuple] = field(default_factory=list)

    @property
    def verbs_open_to_anyone(self) -> List[str]:
        """Verbs granted to the role `any`, in the order Appwrite reported them."""
        found = []
        for raw in self.grants:
            parsed = parse_grant(raw)
            if parsed and role_is_unauthenticated(parsed[1]):
                found.append(parsed[0])
        return found

    @property
    def mutating_verbs_open_to_anyone(self) -> List[str]:
        return [v for v in self.verbs_open_to_anyone if v in MUTATING_VERBS]

    @property
    def is_open_to_anyone(self) -> bool:
        return bool(self.verbs_open_to_anyone or self.open_items)


@dataclass
class PermissionsReport:
    """Every container inspected, plus everything that could not be inspected."""

    containers: List[ContainerPermissions] = field(default_factory=list)
    indeterminate: List[str] = field(default_factory=list)

    @property
    def open_containers(self) -> List[ContainerPermissions]:
        return [c for c in self.containers if c.is_open_to_anyone]

    @property
    def is_clean(self) -> bool:
        """No container open to `any`, AND every container was actually read.

        The second half is the whole point. Without it, a project whose permissions the
        key may not read reports itself locked down — a failed read presented as a
        clean result (C-244).
        """
        return not (self.open_containers or self.indeterminate)

    def render(self) -> str:
        return render_permissions_report(self)


def _read_container(
    fetch, kind: str, container_id: str, security_key: str, report: PermissionsReport
) -> None:
    """Read one container's permissions, recording failure rather than assuming absence."""
    if not container_id:
        report.indeterminate.append(
            f"no {kind} id was configured, so nothing was checked — UNKNOWN, "
            f"which is not the same as locked down"
        )
        return

    # The whole payload handling sits inside the try, not just the fetch. A 200 carrying
    # HTML from a proxy returns bytes, and appwrite >= 14 returns model objects rather
    # than dicts — either makes `raw.get` raise, and an uncaught raise here would abort
    # before the second container is read and exit 1, which this module defines as
    # "a container IS open". Found 2026-08-22.
    try:
        raw = fetch(container_id)
        if not isinstance(raw, dict):
            raise TypeError(
                f"expected a mapping from get_{kind}, got {type(raw).__name__}"
            )
        if "$permissions" not in raw:
            raise KeyError(
                "$permissions absent from the response — an absent key is not an empty "
                "permission list, and the SDK renames fields across majors"
            )
        declared = raw["$permissions"]
        if not isinstance(declared, list):
            # `or []` used to stand here, which turned None, 0, "" and {} into a clean
            # report — precisely the values that look like a stripped or nulled field.
            raise TypeError(
                f"$permissions is {type(declared).__name__}, not a list — a non-list is "
                f"not an empty permission list"
            )
        grants = list(declared)
        security_flag = raw.get(security_key)
    except Exception as e:  # noqa: BLE001 - any failure must be recorded, not swallowed
        report.indeterminate.append(
            f"get_{kind}({container_id}) failed: {e} — permissions UNKNOWN, "
            f"which is not the same as locked down"
        )
        return

    # Deliberately OUTSIDE the try above. `parse_grant` is now total, but the principle
    # is the point: a tally of what could not be read must never be able to discard what
    # WAS read. When this sat inside the try, one malformed element removed the container
    # from the report entirely — grants, verdict and all.
    unparseable = [g for g in grants if parse_grant(g) is None]
    if unparseable:
        report.indeterminate.append(
            f"{kind} {container_id}: {len(unparseable)} permission string(s) could not "
            f"be parsed: {unparseable} — treated as unknown, not empty. The grants that "
            f"WERE readable are still reported below."
        )

    if security_flag is None:
        report.indeterminate.append(
            f"{kind} {container_id}: {security_key} absent from the response — cannot "
            f"tell whether per-item permissions apply, so this container is UNKNOWN"
        )

    container = ContainerPermissions(
        kind=kind,
        container_id=container_id,
        grants=grants,
        security_flag=security_flag,
        unparseable=unparseable,
    )
    report.containers.append(container)


def _read_items(file_manager, container: ContainerPermissions, report: PermissionsReport) -> None:
    """Check the ITEMS inside a container whose per-item security is on.

    Appwrite **unions** container grants with per-item grants when `documentSecurity` /
    `fileSecurity` is on, so a bucket with `permissions=[]` can still hold a file carrying
    `read("any")`. `ensure_bucket` defaults `file_security=True`, which makes that the
    expected state of every live bucket — so without this, the bucket half of every
    all-clear rested on a question the tool never asked.

    The listings already exist: `walk.py` pages every file and document and returns the
    raw records, each carrying its own `$permissions`. Reusing them costs one walk and
    means the verdict covers what it claims to cover, rather than reporting INCOMPLETE
    forever — which is how a tool stops being run (C-244).
    """
    from views_pipeline_core.modules.appwrite.audit.walk import (
        list_all_documents,
        list_all_files,
    )

    from views_pipeline_core.modules.appwrite.audit.walk import unique_by_id

    try:
        if container.kind == "bucket":
            items = list_all_files(file_manager, container.container_id, report)
            noun = "file"
        else:
            items = list_all_documents(file_manager, report)
            noun = "document"
    except Exception as e:  # noqa: BLE001 - a walk fault must be recorded, not raised
        # `_read_container` wraps its payload handling for exactly this reason and says
        # why: an uncaught raise exits 1, which this module defines as "a container IS
        # open". `_read_items` was added without the same wrapper.
        report.indeterminate.append(
            f"{container.kind} {container.container_id}: could not list its items: {e} "
            f"— per-item permissions UNKNOWN"
        )
        return

    # The only caller of these walks that skipped this. `walk.py` documents it as the
    # sole detector of an offset-unstable page: a container shifting under concurrent
    # writes returns `total` rows of which some are repeats and an equal number were
    # never seen, so the count guard agrees with itself and says nothing. A file granted
    # `read("any")` can be one of the never-seen rows.
    items = unique_by_id(items, report, noun)

    for item in items:
        if not isinstance(item, dict):
            report.indeterminate.append(
                f"{noun} record is {type(item).__name__}, not a mapping — cannot read "
                f"its permissions"
            )
            continue
        if "$permissions" not in item:
            # `or []` stood here, which turned an absent key, None, 0, "" and {} into
            # "carries no grant" — the same conflation `_read_container` raises KeyError
            # for 60 lines above, reintroduced on the half that covers production.
            report.indeterminate.append(
                f"{noun} {item.get('$id')}: $permissions absent from the record — an "
                f"absent key is not an empty permission list"
            )
            continue
        grants = item["$permissions"]
        if not isinstance(grants, list):
            report.indeterminate.append(
                f"{noun} {item.get('$id')}: $permissions is "
                f"{type(grants).__name__}, not a list"
            )
            continue
        open_verbs = []
        for raw in grants:
            parsed = parse_grant(raw)
            if parsed is None:
                report.indeterminate.append(
                    f"{noun} {item.get('$id')}: unreadable grant {raw!r}"
                )
            elif role_is_unauthenticated(parsed[1]):
                open_verbs.append(parsed[0])
        if open_verbs:
            container.open_items.append((str(item.get("$id")), sorted(set(open_verbs))))


def read_permissions(file_manager) -> PermissionsReport:
    """Read the declared permissions of this shelf's collection and bucket.

    Coordinates come off the manager's config, the way `walk.py` takes them, so the
    shelf inspected is the shelf `targets.py` resolved — no second source of truth.
    """
    report = PermissionsReport()
    config = file_manager.config

    _read_container(
        lambda cid: file_manager.databases.get_collection(config.database_id, cid),
        "collection",
        config.collection_id,
        "documentSecurity",
        report,
    )
    _read_container(
        file_manager.storage.get_bucket,
        "bucket",
        config.bucket_id,
        "fileSecurity",
        report,
    )

    # Only where per-item grants can actually take effect. With the flag off they are
    # ignored by the substrate, so reading them would be noise.
    for container in report.containers:
        if container.security_flag:
            _read_items(file_manager, container, report)
    return report


def render_permissions_report(report: PermissionsReport) -> str:
    """Print what is there. State the verdict *after* consulting `indeterminate`.

    C-249 was a defect of the renderer alone: the data recorded that the read was
    incomplete and the renderer printed a conclusion above it without looking. The
    unknowns are printed first here for that reason.
    """
    lines: List[str] = ["APPWRITE CONTAINER PERMISSIONS", ""]

    if report.indeterminate:
        lines.append(f"COULD NOT DETERMINE ({len(report.indeterminate)}):")
        lines += [f"  - {note}" for note in report.indeterminate]
        lines.append("")

    for container in report.containers:
        flag_name = "documentSecurity" if container.kind == "collection" else "fileSecurity"
        lines.append(f"{container.kind} {container.container_id}")
        lines.append(f"  {flag_name}: {container.security_flag}")
        if not container.grants:
            if container.security_flag is False:
                lines.append("  permissions: [] — reachable only with an API key")
            elif container.security_flag is True:
                lines.append(
                    f"  permissions: [] at the container, and {flag_name} is on — so "
                    f"per-item grants apply, and {len(container.open_items)} of the "
                    f"items read carry one."
                )
            else:
                lines.append(
                    f"  permissions: [] at the container — but {flag_name} could not be "
                    f"read, so whether per-item grants apply is UNKNOWN and they were "
                    f"not inspected."
                )
        else:
            lines += [f"  permission: {g}" for g in container.grants]
        if container.open_items:
            lines.append(
                f"  >> {len(container.open_items)} individual item(s) carry their own "
                f"grant to an unauthenticated role, and {flag_name} is on so they apply:"
            )
            for item_id, verbs in container.open_items[:10]:
                lines.append(f"       {item_id}: {', '.join(verbs)}")
            if len(container.open_items) > 10:
                lines.append(f"       ... and {len(container.open_items) - 10} more")
        if container.verbs_open_to_anyone:
            verbs = ", ".join(container.verbs_open_to_anyone)
            lines.append(f"  >> OPEN TO ANYONE: {verbs}")
            mutating = container.mutating_verbs_open_to_anyone
            if mutating:
                lines.append(
                    f"  >> anyone with the project id can {', '.join(mutating)} "
                    f"every item here. The project id is not a secret."
                )
            if container.security_flag is False:
                lines.append(
                    f"  >> {flag_name}=False, so these grants govern every item and "
                    f"nothing can narrow them per item."
                )
        lines.append("")

    # OPEN and INCOMPLETE are independent facts and both are stated. Until 2026-08-22 a
    # single malformed grant string suppressed the OPEN verdict entirely: the body printed
    # ">> OPEN TO ANYONE: read, delete" and the run then ended "INCOMPLETE — at least one
    # container could not be read", which the operator guide teaches means "the key lacks
    # a scope, get a better key and re-run". A live open container read as a credentials
    # chore. One condition must never hide another.
    if report.open_containers:
        names = ", ".join(c.container_id for c in report.open_containers)
        lines.append(f"VERDICT: OPEN — {len(report.open_containers)} container(s): {names}")
        lines.append(
            "This tool does not change permissions and has no --fix. Whether a "
            "partner-facing container should permit an unauthenticated role is C-292, "
            "and it is an operator decision."
        )
    if report.indeterminate:
        lines.append(
            f"VERDICT: INCOMPLETE — {len(report.indeterminate)} thing(s) could not be "
            f"determined, listed at the top. This is not an all-clear"
            + (", and it does not reduce the OPEN finding above." if report.open_containers
               else ".")
        )
    if not (report.open_containers or report.indeterminate):
        lines.append(
            "VERDICT: no container grants anything to an unauthenticated role, and "
            "everything this tool can check was checked."
        )
    return "\n".join(lines)


#: A finding outranks an incomplete read. Both are reported in the text, but a status
#: code is one number and it must carry the actionable half.
EXIT_CLEAN, EXIT_OPEN, EXIT_INCOMPLETE = 0, 1, 2


def permissions_exit_code(report: PermissionsReport) -> int:
    """0 nothing open · 1 something open · 2 could not complete.

    **A finding wins over an incomplete read**, which is the opposite of what this
    returned until 2026-08-24. `indeterminate` used to win unconditionally, so a run that
    found an open container AND hit one unparseable grant exited 2 — and the operator
    guide teaches 2 as *"usually the key lacks a read scope; get a key that can read
    collections and run it again"*. A live open container was reported to any cron job or
    CI step as a credentials chore.

    Indeterminate notes are cheap: an absent security flag, one unreadable grant string,
    a record with no `$id`, a walk hitting the page guard. Any of them would have masked
    a finding. The renderer was fixed for exactly this on 2026-08-22 — *"one condition
    must never hide another"* — and the fix was applied to the human-readable half only.

    2 still means *incomplete*, and an incomplete run that found nothing open has NOT
    established that nothing is open. That is the C-232 rule and it is unchanged: only a
    positive finding is allowed to outrank it, because a positive finding is already the
    louder answer.
    """
    if report.open_containers:
        return EXIT_OPEN
    if report.indeterminate:
        return EXIT_INCOMPLETE
    return EXIT_CLEAN
