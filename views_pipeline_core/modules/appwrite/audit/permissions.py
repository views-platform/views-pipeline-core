"""What a container actually permits, read from the live seam. C-292.

Read-only, like the rest of this package. It reads `get_collection` and `get_bucket`,
reports the `$permissions` array verbatim, and says which verbs — if any — are granted
to the role `any`.

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

#: The role that means *anyone at all*, including unauthenticated callers holding only
#: the project id — which is not a secret. `users` means any signed-in user and is a
#: different, lesser exposure; it is reported but not treated as the same thing.
ROLE_ANYONE = "any"

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
    match = _GRANT.match(raw or "")
    return (match.group(1).lower(), match.group(2)) if match else None


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

    @property
    def verbs_open_to_anyone(self) -> List[str]:
        """Verbs granted to the role `any`, in the order Appwrite reported them."""
        found = []
        for raw in self.grants:
            parsed = parse_grant(raw)
            if parsed and parsed[1] == ROLE_ANYONE:
                found.append(parsed[0])
        return found

    @property
    def mutating_verbs_open_to_anyone(self) -> List[str]:
        return [v for v in self.verbs_open_to_anyone if v in MUTATING_VERBS]

    @property
    def is_open_to_anyone(self) -> bool:
        return bool(self.verbs_open_to_anyone)


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
    try:
        raw = fetch(container_id)
    except Exception as e:  # noqa: BLE001 - any failure must be recorded, not swallowed
        report.indeterminate.append(
            f"get_{kind}({container_id}) failed: {e} — permissions UNKNOWN, "
            f"which is not the same as locked down"
        )
        return

    grants = list(raw.get("$permissions") or [])
    container = ContainerPermissions(
        kind=kind,
        container_id=container_id,
        grants=grants,
        security_flag=raw.get(security_key),
        unparseable=[g for g in grants if parse_grant(g) is None],
    )
    if container.unparseable:
        report.indeterminate.append(
            f"{kind} {container_id}: {len(container.unparseable)} permission string(s) "
            f"could not be parsed: {container.unparseable} — treated as unknown, not empty"
        )
    report.containers.append(container)


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
            lines.append("  permissions: [] — reachable only with an API key")
        else:
            lines += [f"  permission: {g}" for g in container.grants]
        if container.is_open_to_anyone:
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

    if report.indeterminate:
        lines.append(
            "VERDICT: INCOMPLETE — at least one container could not be read. "
            "Nothing below should be read as an all-clear."
        )
    elif report.open_containers:
        names = ", ".join(c.container_id for c in report.open_containers)
        lines.append(f"VERDICT: OPEN — {len(report.open_containers)} container(s): {names}")
        lines.append(
            "This tool does not change permissions and has no --fix. Whether a "
            "partner-facing container should permit `any` is C-292, and it is an "
            "operator decision."
        )
    else:
        lines.append(
            "VERDICT: no container grants anything to `any`. Every container read "
            "successfully."
        )
    return "\n".join(lines)


def permissions_exit_code(report: PermissionsReport) -> int:
    """0 nothing open, 1 something open, 2 could not complete.

    Ordering matters: `indeterminate` wins. A run that could not read one container and
    found the other locked down has not established that nothing is open.
    """
    if report.indeterminate:
        return 2
    return 1 if report.open_containers else 0
