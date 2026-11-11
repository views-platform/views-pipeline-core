"""Deliberate provisioning of Appwrite containers. RUNNABLE, NEVER IMPORTABLE.

Creating storage — a bucket, a database, a collection, a collection's attributes — is a
**setup act performed by a person**, not a side effect of publishing a forecast.

Until þing-02 (#331) it was the latter: an ordinary upload created whatever was missing
on its way through, at five call sites, one of them inside a method whose entire job was
to *look something up*. That is why the platform's Appwrite key carries 20 scopes — the
scopes are what the delivery code demands. Least privilege could not be applied while
narrowing the key broke every upload, so this module exists to make the delivery path
read-and-write-only:

    python -m views_pipeline_core.modules.appwrite.provisioning ensure-collection
    python -m views_pipeline_core.modules.appwrite.provisioning ensure-bucket

**The delivery path must never import this module.** That is not a convention — it is
asserted in a subprocess by ``tests/test_import_purity.py`` (#332), which is why the
dependency runs one way only: ``provisioning`` imports ``file``, never the reverse.

Coordinates and credentials come from the process environment via
``PredictionStoreConfig.from_environment()``, which fails loud naming any missing
variable. Nothing is defaulted here.

**C-238, fixed under þing-04 S17.** ``ensure_bucket``'s follow-on database creation
passed its arguments in the wrong order, so it ensured a database whose id was the
*bucket's name*. It had been reproduced verbatim from ``file.py`` because a relocation
must not change behaviour silently, and it was invisible against the shared production
project, where the real database already existed. The first bring-up against an **empty**
project is where it stops being invisible, and that is this one.

**The deferred schema, now declared.** This module used to derive the collection's
schema from whatever payload it was handed, which contradicts ADR-040 (Authority of
Declarations Over Inference). It was deferred behind a named trigger — *a non-production
Appwrite project existing, since nobody has ever run a first bring-up against an empty
one* (þing-02 Á-5 / G2(h)). **That trigger fired** (þing-04 S17): a CRAF'd-only project
is being created, and it is rehearsed against a throwaway one first. The fields every delivery
document carries are therefore declared in ``DELIVERY_METADATA_ATTRIBUTES`` below, and
``ensure_attributes`` creates them whatever payload it is given.

Payload inference is **retained for anything beyond that set**, because callers may still
pass extra metadata and removing their only route to an attribute is a separate decision
with a separate blast radius. Against a brand-new project it no longer has anything to do
for the delivery path.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from appwrite.exception import AppwriteException
from appwrite.query import Query

# `Permission` and `Role` are deliberately NOT imported. This module grants nothing by
# default (ADR-061), so a caller wanting a wider grant constructs it and passes it in,
# where the reason for it is visible.
#
# That is a CONVENTION, not the enforcement. Nothing inspects this import list and the
# import can be restored in one line. What actually prevents a container being created
# open is `tests/test_no_container_is_provisioned_open.py`, which AST-walks every
# `permissions=` argument and fails on any grant to a role reachable without
# authenticating. An earlier version of
# this comment credited the missing import with the protection — a claim the evidence made
# consistent but did not support (C-273).

from views_pipeline_core.modules.appwrite.file import (
    INITIAL_RETRY_DELAY,
    MAX_ATTRIBUTE_CREATION_RETRIES,
    OperationResult,
    exception_message,
)

logger = logging.getLogger(__name__)

# Page size for the provisioning lookups below, stated rather than inherited from the
# server's 25-row default (C-241 is what inheriting it cost).
_PROVISION_PAGE = 100


def _complete_listing(response: Dict[str, Any], key: str) -> Tuple[List[Any], Optional[str]]:
    """Return ``(items, truncation_reason)`` for a listing that drives a CREATE.

    Every listing in this module answers "does it already exist?", and the answer NO
    means "create it". So a short read does not produce a smaller answer — it produces a
    duplicate, or a spurious "already exists" error from the server. The truncation is
    returned rather than raised so each caller can fold it into the ``OperationResult``
    it already returns — deliberately a different shape from the rest of this module,
    which reports failure as an ``OperationResult`` directly. A private helper cannot
    know which code or message its caller wants, and raising would force every caller
    into a try/except for a condition that is not exceptional. Cluster J.
    """
    items = response.get(key, [])
    total = response.get("total")
    if total is not None and total > len(items):
        return items, (
            f"listing of {key} is incomplete: the server reports {total} but only "
            f"{len(items)} were read, so an existing entry could be missed and "
            f"re-created"
        )
    return items, None

# The attributes every metadata document carries regardless of payload. Unlike the
# payload-inferred ones below, these are declared — ADR-040's preferred direction.
FIXED_METADATA_ATTRIBUTES: List[Dict[str, Any]] = [
    {"key": "fileId", "type": "string", "size": 255, "required": True},
    {"key": "bucketId", "type": "string", "size": 255, "required": True},
    {"key": "filename", "type": "string", "size": 500, "required": True},
    {"key": "file_size", "type": "integer", "required": False},
    {"key": "mime_type", "type": "string", "size": 100, "required": False},
    {"key": "uploaded_at", "type": "datetime", "required": False},
    {"key": "file_hash", "type": "string", "size": 64, "required": False},
]


# The fields every *delivery* document carries, from `FileMetadata.to_dict()`
# (`modules/datastore/datastore.py:166-191`). Together with the fixed attributes above
# these are the whole metadata document that `file.py` builds and writes.
#
# Before these were declared, they existed only on the shared production collection,
# created years ago by the write-time inference #331 removed. A collection provisioned
# by this module's CLI into an EMPTY project carried the seven fixed attributes and
# nothing else — so the first delivery uploaded its file and then failed
# `Unknown attribute: loa` on the document, leaving an orphaned file in the bucket.
#
# `required` is False and strings are 255 throughout, which is exactly what the inference
# produced. That is deliberate: a collection provisioned today must have the same shape as
# the production one provisioned by inference, or the two projects disagree about what a
# delivery document is. Widening `description` past 255, or making `loa`/`category`
# required, are both defensible and are both a change to the production schema — not
# something to smuggle in under a bring-up fix.
DELIVERY_METADATA_ATTRIBUTES: List[Dict[str, Any]] = [
    {"key": "loa", "type": "string", "size": 255, "required": False},
    {"key": "name", "type": "string", "size": 255, "required": False},
    {"key": "type", "type": "string", "size": 255, "required": False},
    {"key": "targets", "type": "string", "size": 255, "required": False, "array": True},
    {"key": "category", "type": "string", "size": 255, "required": False},
    {"key": "description", "type": "string", "size": 255, "required": False},
]

#: What `ensure_attributes` creates regardless of payload.
DECLARED_METADATA_ATTRIBUTES: List[Dict[str, Any]] = (
    FIXED_METADATA_ATTRIBUTES + DELIVERY_METADATA_ATTRIBUTES
)

#: The attribute types this module can create. Declared so the declaration can be
#: checked BEFORE anything is written — see `_validate_declarations`.
CREATABLE_ATTRIBUTE_TYPES = frozenset({"string", "integer", "datetime", "boolean"})

# The indexes the WRITE path needs. Attributes alone are not a complete bring-up:
# `check_file_exists_by_hash` issues `Query.equal("file_hash", ...)` (`file.py:1200`) and
# `_store_metadata_document` issues `Query.equal("fileId", ...)` (`file.py:1301`, `:1769`)
# against this collection, and this package's own comment at `file.py:1216` already states
# that an unindexed equality query "is not dependable against this store" — which is why
# the filename match there is filtered in Python instead.
#
# Nothing in this repository, or in any sibling checkout, created an index before now,
# while the platform key has carried `indexes.read`/`indexes.write` all along. On the
# shared production collection the indexes exist, created outside this code. On an EMPTY
# project they did not, so a bring-up that declared only attributes left the first
# delivery to fail one step later than `Unknown attribute: loa`: the hash lookup returns
# an error, which is not NOT_FOUND, so the upload refuses with "Could not determine
# whether a duplicate exists".
#
# The read path is covered too, but only because a consumer MEASURED its own queries
# rather than us guessing from our docstrings. `file.py:1046-1050` turns any
# caller-supplied `filters` dict into equality queries, so "what the read path needs"
# is not derivable from this repository at all. views-crafdapi read its serving code
# (`views-crafdapi@feat/crafd-project`) and reported two shapes:
#
#   {category, type}          — prediction/manager.py ~:290, :343, :357
#   {name, category, type}    — metadata.py:448
#
# and confirmed that the `loa` filter in our own DatastoreModule docstrings is
# documentation only, reached by nothing on their serving path. One compound index serves
# both shapes, ordered (category, type, name) so the two-attribute query matches a
# leftmost prefix; a (name, …) ordering would serve only the three-attribute one.
#
# `"key"` rather than importing `appwrite.enums.index_type.IndexType.KEY`: checked
# against the SDK this package actually pins (`appwrite>=13.4.1,<14.0.0`, 13.6.1 in the
# dev env), where `create_index` assigns `api_params['type'] = type` with no
# normalisation, and `IndexType.KEY.value == "key"`. So the literal and the enum produce
# identical wire output, and the literal keeps this a declaration rather than an import.
#
# The enum's MODULE PATH differs between SDK majors — `appwrite.enums.index_type` on 13.x,
# `appwrite.enums.databases_index_type` on 19.x, which is what other repos on this
# platform pin — so importing it here would couple this constant to a version range wider
# than our own. A test pins the value against whichever enum is installed; this is not a
# claim resting on a comment.
# `lengths` on the compound index is not a tuning knob, it is a ceiling. The three
# attributes are `string(255)`, and InnoDB's index key limit is 3072 bytes: at utf8mb4's
# 4 bytes per character, 3 x 255 x 4 = 3060. That fits with twelve bytes to spare, before
# any column Appwrite adds to its own indexes — so an unbounded compound index on these
# three is one schema change away from a bring-up that fails while creating an index,
# which is the failure this whole module exists to stop. 64 characters is far longer than
# any value these fields hold ("forecast", an artifact type, a model name) and leaves the
# key at 768 bytes.
DECLARED_INDEXES: List[Dict[str, Any]] = [
    {"key": "idx_fileId", "type": "key", "attributes": ["fileId"]},
    {"key": "idx_file_hash", "type": "key", "attributes": ["file_hash"]},
    {
        "key": "idx_category_type_name",
        "type": "key",
        "attributes": ["category", "type", "name"],
        "lengths": [64, 64, 64],
    },
]


def _validate_declarations() -> None:
    """Fail at import if a declaration cannot be created, rather than part-way through.

    An earlier version raised from inside the creation loop. That is fail-loud, but it
    leaves every attribute before the bad one created and every one after it unattempted
    — a half-built schema, which is the exact outcome `ensure_attributes` refuses to
    report success over forty lines below. The declarations are module constants, so the
    check belongs where they are, not where they are used.
    """
    for attr in DECLARED_METADATA_ATTRIBUTES:
        if attr["type"] not in CREATABLE_ATTRIBUTE_TYPES:
            raise ValueError(
                f"declared attribute {attr['key']!r} has type {attr['type']!r}, which "
                f"this module cannot create. Add a creator for it in "
                f"`_create_declared_attribute` and to CREATABLE_ATTRIBUTE_TYPES."
            )
    declared_keys = {attr["key"] for attr in DECLARED_METADATA_ATTRIBUTES}
    for index in DECLARED_INDEXES:
        unknown = [a for a in index["attributes"] if a not in declared_keys]
        if unknown:
            raise ValueError(
                f"declared index {index['key']!r} covers {unknown}, which no declared "
                f"attribute provides. An index on an attribute that is never created "
                f"fails at bring-up time against an empty project."
            )
        lengths = index.get("lengths")
        if lengths is not None and len(lengths) != len(index["attributes"]):
            raise ValueError(
                f"declared index {index['key']!r} has {len(lengths)} length(s) for "
                f"{len(index['attributes'])} attribute(s). Appwrite matches them "
                f"positionally, so a short list silently caps the wrong column."
            )


_validate_declarations()


def infer_attribute_type(value: Any) -> Tuple[str, bool]:
    """Infer an Appwrite attribute type from a sample value.

    Inference, and ADR-040 says declarations should win — see the module docstring.
    Retained as-is by the relocation; superseded when the schema is declared.
    """
    is_array = isinstance(value, list)
    base_value = value[0] if is_array and value else value

    if isinstance(base_value, bool):
        return "boolean", is_array
    if isinstance(base_value, int):
        return "integer", is_array
    if isinstance(base_value, float):
        return "double", is_array
    if isinstance(base_value, str):
        try:
            datetime.fromisoformat(base_value.replace("Z", "+00:00"))
            return "datetime", is_array
        except (ValueError, AttributeError):
            return "string", is_array
    return "string", is_array


class AppwriteProvisioner:
    """Creates the containers the delivery path expects to already exist.

    Constructed around a live ``AppWriteFileModule`` so it reuses that module's
    authenticated clients rather than opening a second connection with a second copy
    of the credential.
    """

    def __init__(self, file_manager):
        self._fm = file_manager
        self.databases = file_manager.databases
        self.storage = file_manager.storage
        self.config = file_manager.config

    # -- databases ---------------------------------------------------------------

    def ensure_database(
        self, database_id: str = None, database_name: str = None
    ) -> OperationResult:
        """Create the metadata database if it does not exist."""
        db_id = database_id or self.config.database_id
        db_name = database_name or self.config.database_name

        if not db_id or not db_name:
            return OperationResult(
                success=False,
                error="Database ID and name must be provided in config or as parameters",
                code="MISSING_CONFIG",
            )

        try:
            listing = self.databases.list(queries=[Query.limit(_PROVISION_PAGE)])
            databases, truncated = _complete_listing(listing, "databases")
            if truncated:
                return OperationResult(
                    success=False, error=truncated, code="LISTING_INCOMPLETE"
                )

            for db in databases:
                if db["name"] == db_name or db["$id"] == db_id:
                    logger.info(f"Database '{db_name}' already exists")
                    return OperationResult(success=True, data=db, code="EXISTS")

            try:
                result = self.databases.create(
                    database_id=db_id, name=db_name, enabled=True
                )
                logger.info(f"Created new database: {db_name}")
                return OperationResult(success=True, data=result)
            except AppwriteException as create_error:
                # Hitting the database limit while the database already exists is benign.
                if "maximum number of databases" in exception_message(create_error).lower():
                    listing = self.databases.list(
                        queries=[Query.limit(_PROVISION_PAGE)]
                    )
                    databases, truncated = _complete_listing(listing, "databases")
                    for db in databases:
                        if db["$id"] == db_id:
                            logger.warning(
                                f"Database limit reached, but database '{db_id}' exists"
                            )
                            return OperationResult(success=True, data=db, code="EXISTS")
                    if truncated:
                        # Not found — but the listing was short, so "not found" here is
                        # "not seen". Re-raising the create error alone would report
                        # "maximum number of databases" when the truthful answer is
                        # "could not confirm whether it already exists".
                        logger.error(
                            f"Database {db_id!r} was not found while recovering from a "
                            f"create failure, but the listing was incomplete: "
                            f"{truncated}"
                        )
                raise create_error

        except AppwriteException as e:
            logger.error(f"Database operation failed: {e.message}")
            return OperationResult(
                success=False,
                error=f"Database operation failed: {e.message}",
                code=e.type,
            )

    # -- collections and attributes ----------------------------------------------

    def ensure_collection(
        self,
        metadata: Dict[str, Any] = None,
        collection_name: str = None,
        collection_id: str = None,
        database_id: str = None,
        permissions: List[str] = None,
    ) -> OperationResult:
        """Create the metadata collection (and its database) if they do not exist.

        **`permissions` defaults to `[]` — least privilege (C-292, ADR-061).** An empty
        list is not "no access": a server API key bypasses container permissions, and
        every consumer on this platform authenticates with one. It means *only* the key.

        Until 2026-08-14 this method hardcoded
        `Permission.{read,create,update,delete}(Role.any())` with `document_security=False`,
        making every collection it created readable, writable and deletable by anyone
        holding the project id — which is not a secret. `ensure_bucket`, in this same
        class, has always defaulted to `permissions=[]`; the two halves of one tool
        disagreed, and nobody chose it. The grant was carried verbatim through #331
        under the rule that a relocation must not change behaviour, and predates it.

        It was not merely reachable by CLI. Before #331 this creation path ran from
        `upload_file_with_metadata`, `upload_file_from_bytes_with_metadata` and
        `check_file_exists_by_hash` — so an ordinary delivery to a new partner created
        an open collection automatically.

        **Widening is now something a caller states**, and stating it should come with a
        reason. This changes nothing already provisioned: Appwrite applies these grants
        at creation, and the existing-collection branch below does not re-apply them.
        Inspect live containers with
        `python -m views_pipeline_core.modules.appwrite.audit --permissions`.
        """
        db_id = database_id or self.config.database_id
        coll_name = collection_name or self.config.collection_name
        coll_id = collection_id or self.config.collection_id

        if not db_id or not coll_name or not coll_id:
            return OperationResult(
                success=False,
                error=(
                    "Database ID, collection name, and collection ID must be provided "
                    "in config or as parameters"
                ),
                code="MISSING_CONFIG",
            )

        db_result = self.ensure_database(db_id, self.config.database_name)
        if not db_result.success:
            return db_result

        try:
            listing = self.databases.list_collections(
                db_id, queries=[Query.limit(_PROVISION_PAGE)]
            )
            collections, truncated = _complete_listing(listing, "collections")
            if truncated:
                return OperationResult(
                    success=False, error=truncated, code="LISTING_INCOMPLETE"
                )

            # Identity is the id. The name must agree with it, and a disagreement is
            # refused rather than resolved.
            #
            # This used to be `$id == coll_id or name == coll_name`, which meant either
            # half of the pair could select a collection on its own: override only
            # APPWRITE_PROD_FORECASTS_COLLECTION_ID and the name still says
            # production_forecasts, so the name disjunct matches it and attributes are
            # written into the wrong shelf. Whichever appeared first in the listing won,
            # nothing compared the winner's other field, and nothing noticed that two
            # different collections had each matched one half. C-250 and C-271 are the
            # shelf-level form of the same conflation.
            for collection in collections:
                if collection["$id"] != coll_id:
                    continue
                if collection["name"] != coll_name:
                    return OperationResult(
                        success=False,
                        error=(
                            f"Collection '{coll_id}' exists but is named "
                            f"'{collection['name']}', not '{coll_name}'. Refusing rather "
                            f"than guessing which you meant — a coordinate pair that "
                            f"disagrees is how one shelf's schema gets written into "
                            f"another's. Correct the id/name pair and re-run."
                        ),
                        code="COORDINATE_MISMATCH",
                    )
                # BEFORE any write, and gated on a non-empty list rather than on
                # `is not None`. Two corrections, both found 2026-08-24:
                #
                # `permissions=[]` is the least-privilege posture ADR-061 tells callers
                # to state, and it is identical to the default. Refusing it made the
                # method non-idempotent — succeed once, hard-fail on every re-run — and
                # produced the message "the 0 permission(s) you passed were NOT applied".
                # `ensure_bucket` treats `None` and `[]` identically and always has.
                #
                # And this sat AFTER `ensure_attributes`, so it claimed "nothing was
                # written" having already created attributes on the live collection.
                # A refusal that has already written is not a refusal.
                if permissions:
                    # A caller stating a non-empty grant is asking for it to be applied.
                    # Appwrite fixes grants at creation, so on this branch nothing can
                    # be. Returning success here is C-291's exact shape — "reported OK
                    # while writing nothing" — on the security-relevant argument, in the
                    # method that was fixed for C-291 in #473.
                    return OperationResult(
                        success=False,
                        error=(
                            f"Collection '{coll_id}' already exists, and permissions are "
                            f"applied only at creation — the {len(permissions)} "
                            f"permission(s) you passed were NOT applied and nothing was "
                            f"written. Change them deliberately in the Appwrite console, "
                            f"or call this without `permissions` if you only meant to "
                            f"ensure the schema. Inspect what is actually there with "
                            f"`python -m views_pipeline_core.modules.appwrite.audit "
                            f"--permissions`."
                        ),
                        code="PERMISSIONS_NOT_APPLICABLE",
                    )

                attr_result = self.ensure_attributes(
                    db_id, collection["$id"], metadata or {}
                )
                if not attr_result.success:
                    return attr_result

                return OperationResult(
                    success=True,
                    data={
                        "collection": collection,
                        "database_id": db_id,
                        "collection_id": collection["$id"],
                    },
                    code="EXISTS",
                )

            result = self.databases.create_collection(
                database_id=db_id,
                collection_id=coll_id,
                name=coll_name,
                # Least privilege by default — see this method's docstring and ADR-061.
                # `document_security=False` is retained deliberately: with no
                # container-level grants there is nothing for per-document permissions
                # to narrow, and flipping it would change how every existing document
                # is evaluated for no benefit we can name.
                permissions=[] if permissions is None else list(permissions),
                document_security=False,
                enabled=True,
            )

            attr_result = self.ensure_attributes(db_id, coll_id, metadata or {})
            if not attr_result.success:
                # The collection exists but its schema is incomplete — say so rather
                # than reporting a successful creation over a half-built container.
                return attr_result

            return OperationResult(
                success=True,
                data={
                    "collection": result,
                    "database_id": db_id,
                    "collection_id": result["$id"],
                },
            )

        except AppwriteException as e:
            logger.error(f"Collection creation failed: {e.message}")
            return OperationResult(
                success=False,
                error=f"Collection creation failed: {e.message}",
                code=e.type,
            )

    def ensure_attributes(
        self,
        database_id: str,
        collection_id: str,
        metadata: Dict[str, Any],
        max_retries: int = MAX_ATTRIBUTE_CREATION_RETRIES,
        initial_delay: float = INITIAL_RETRY_DELAY,
    ) -> OperationResult:
        """Create the declared attributes, plus any inferred from ``metadata``.

        The declared set is created whatever ``metadata`` holds — that is what makes an
        empty dict a complete bring-up. Inference only reaches keys the declaration does
        not already cover.
        """
        delay = initial_delay
        existing_attribute_keys = set()

        for attempt in range(1, max_retries + 1):
            try:
                existing_attributes = self.databases.list_attributes(
                    database_id, collection_id
                )
                existing_attribute_keys = {
                    attr["key"] for attr in existing_attributes["attributes"]
                }
                logger.debug(f"Found {len(existing_attribute_keys)} existing attributes")
                break
            except AppwriteException as e:
                # NOTE: matches the human-readable message rather than e.type — same
                # defect class as C-179, preserved by the relocation (register C-235).
                if "collection_not_found" in exception_message(e) and attempt < max_retries:
                    logger.warning(
                        f"Collection not ready (attempt {attempt}/{max_retries}), "
                        f"retrying in {delay}s"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Failed to list attributes: {e.message}")
                    return OperationResult(
                        success=False,
                        error=f"Failed to list attributes: {e.message}",
                        code=e.type,
                    )

        # Every attribute is attempted before reporting, so one bad field does not
        # hide the others — but a failure is NOT swallowed. This is a setup tool
        # invoked deliberately by a person; ADR-047's graceful degradation governs the
        # delivery path, not this one, and "OK" over a half-built schema is the same
        # class of lie the þing convened about (#322).
        failed: List[str] = []

        for attr in DECLARED_METADATA_ATTRIBUTES:
            if attr["key"] in existing_attribute_keys:
                continue
            try:
                self._create_declared_attribute(database_id, collection_id, attr)
            except AppwriteException as e:
                if "already exists" not in exception_message(e).lower():
                    logger.error(
                        f"Failed to create declared attribute {attr['key']}: {e.message}"
                    )
                    failed.append(f"{attr['key']} ({e.type or e.message})")

        for key, value in metadata.items():
            if key in existing_attribute_keys:
                continue
            try:
                attr_type, is_array = infer_attribute_type(value)
                self._create_attribute_by_type(
                    database_id, collection_id, key, attr_type, is_array
                )
            except AppwriteException as e:
                logger.error(f"Failed to create attribute '{key}': {e.message}")
                failed.append(f"{key} ({e.type or e.message})")

        if failed:
            return OperationResult(
                success=False,
                error=(
                    f"{len(failed)} attribute(s) could not be created in collection "
                    f"'{collection_id}': {', '.join(failed)}. The collection's schema is "
                    f"incomplete; documents using these fields will be rejected."
                ),
                code="ATTRIBUTES_INCOMPLETE",
            )

        pending = self._await_declared_attributes(
            database_id, collection_id, max_retries, initial_delay
        )
        if pending is not None:
            return pending

        return self.ensure_indexes(database_id, collection_id, max_retries, initial_delay)

    def _await_declared_attributes(
        self,
        database_id: str,
        collection_id: str,
        max_retries: int,
        initial_delay: float,
    ) -> Optional[OperationResult]:
        """Block until every declared attribute is usable. Returns a failure, or None.

        Appwrite creates attributes asynchronously: the create call returns while the
        attribute is still ``processing``, and a document referencing one that is not yet
        ``available`` is rejected. Until now this module returned success the moment the
        creates were accepted — harmless for years, because on the production collection
        the attributes had existed since long before any delivery. The whole point of
        this bring-up is that they now exist *seconds* before the first delivery, which
        reintroduces `Unknown attribute: loa` by a different route.

        A missing ``status`` field is treated as available, so this cannot invent a
        failure against a server that does not report one. The field's presence was not
        confirmed against a live instance from here; what is confirmed is that waiting
        for it cannot be worse than not waiting.
        """
        declared = {attr["key"] for attr in DECLARED_METADATA_ATTRIBUTES}
        delay = initial_delay
        unusable: Dict[str, str] = {}

        for attempt in range(1, max_retries + 1):
            try:
                listing = self.databases.list_attributes(
                    database_id, collection_id, queries=[Query.limit(_PROVISION_PAGE)]
                )
            except AppwriteException as e:
                return OperationResult(
                    success=False,
                    error=(
                        f"attributes were created in '{collection_id}' but their "
                        f"readiness could not be confirmed: {e.message}. Do not deliver "
                        f"into this collection until `audit` shows every attribute "
                        f"available."
                    ),
                    code=e.type or "ATTRIBUTES_UNCONFIRMED",
                )

            attributes, truncated = _complete_listing(listing, "attributes")
            seen = {attr["key"] for attr in attributes}

            # Two ways an attribute is not yet usable, and BOTH are waited for rather
            # than failed on. An attribute absent from the listing altogether is the
            # propagation window — treating that as a hard failure would convert a
            # normal delay into a refused bring-up. It is also what a truncated listing
            # looks like, and a short read is not benign on this question: an attribute
            # still `processing` that fell off the page is absent from `unusable` and so
            # would read as ready. "Not seen" reported as "available" is Cluster J on
            # the readiness question itself.
            unseen = declared - seen
            unusable = {
                attr["key"]: attr["status"]
                for attr in attributes
                if attr["key"] in declared
                and attr.get("status", "available") != "available"
            }
            if not unseen and not unusable:
                return None
            if attempt < max_retries:
                logger.info(
                    "waiting for %d attribute(s) to become usable (attempt %d/%d)",
                    len(unseen) + len(unusable),
                    attempt,
                    max_retries,
                )
                time.sleep(delay)
                delay *= 2

        reasons = []
        if unusable:
            reasons.append(f"not available: {unusable}")
        if unseen:
            reasons.append(
                f"created but absent from a listing of {len(seen)} attribute(s): "
                f"{sorted(unseen)}"
                + (f" — {truncated}" if truncated else "")
            )
        return OperationResult(
            success=False,
            error=(
                f"attribute(s) in '{collection_id}' never became usable — "
                + "; ".join(reasons)
                + ". The schema exists but documents using these fields will be "
                "rejected; re-run this command, and if a status is 'failed' or 'stuck' "
                "the attribute must be deleted in the console before it can be "
                "recreated."
            ),
            code="ATTRIBUTES_NOT_AVAILABLE",
        )

    def ensure_indexes(
        self,
        database_id: str,
        collection_id: str,
        max_retries: int = MAX_ATTRIBUTE_CREATION_RETRIES,
        initial_delay: float = INITIAL_RETRY_DELAY,
    ) -> OperationResult:
        """Create the indexes the delivery's own queries need. See DECLARED_INDEXES."""
        try:
            listing = self.databases.list_indexes(
                database_id, collection_id, queries=[Query.limit(_PROVISION_PAGE)]
            )
            indexes, truncated = _complete_listing(listing, "indexes")
            if truncated:
                # Logged rather than returned, and this is the one listing in this module
                # where that is defensible: a missed index is re-created, the server
                # answers "already exists", and the loop below tolerates it. Said out
                # loud because every other listing here treats a short read as a failure.
                logger.warning(
                    "index listing for '%s' is incomplete (%s); an existing index may be "
                    "re-created, which the server refuses harmlessly",
                    collection_id,
                    truncated,
                )
            existing = {index["key"] for index in indexes}
        except AppwriteException as e:
            return OperationResult(
                success=False,
                error=f"Failed to list indexes: {e.message}",
                code=e.type or "INDEXES_UNREADABLE",
            )

        failed: List[str] = []
        for index in DECLARED_INDEXES:
            if index["key"] in existing:
                continue
            try:
                self.databases.create_index(
                    database_id,
                    collection_id,
                    index["key"],
                    index["type"],
                    index["attributes"],
                    lengths=index.get("lengths"),
                )
                logger.debug(f"Created index: {index['key']}")
            except AppwriteException as e:
                if "already exists" in exception_message(e).lower():
                    continue
                logger.error(f"Failed to create index {index['key']}: {e.message}")
                failed.append(f"{index['key']} ({e.type or e.message})")

        if failed:
            return OperationResult(
                success=False,
                error=(
                    f"{len(failed)} index(es) could not be created in collection "
                    f"'{collection_id}': {', '.join(failed)}. The schema is complete but "
                    f"the delivery's hash and fileId lookups are not dependable without "
                    f"them — see DECLARED_INDEXES."
                ),
                code="INDEXES_INCOMPLETE",
            )

        return OperationResult(success=True)

    def _create_declared_attribute(
        self, database_id: str, collection_id: str, attr: Dict[str, Any]
    ):
        """Create one attribute exactly as declared, including its arity.

        ``array`` is threaded through every type rather than only the one that needs it
        today. A declaration that says `array` and a creator that silently drops it is
        how `targets` would become a scalar and reject the list the delivery writes —
        and the guard would have been right about the declaration and wrong about the
        collection.
        """
        array = bool(attr.get("array", False))
        attr_creators = {
            "string": lambda: self.databases.create_string_attribute(
                database_id,
                collection_id,
                attr["key"],
                attr["size"],
                attr["required"],
                array=array,
            ),
            "integer": lambda: self.databases.create_integer_attribute(
                database_id, collection_id, attr["key"], attr["required"], array=array
            ),
            "datetime": lambda: self.databases.create_datetime_attribute(
                database_id, collection_id, attr["key"], attr["required"], array=array
            ),
            "boolean": lambda: self.databases.create_boolean_attribute(
                database_id, collection_id, attr["key"], attr["required"], array=array
            ),
        }
        # Unreachable for the declared set: `_validate_declarations` has already run at
        # import. Kept as the backstop for a hand-built `attr`, and deliberately a raise
        # rather than a silent skip — a skipped declaration is a schema silently short of
        # what the delivery writes. The pre-import check is what keeps a bad declaration
        # from leaving half the attributes created.
        if attr["type"] not in attr_creators:
            raise ValueError(
                f"attribute {attr['key']!r} has type {attr['type']!r}, which this module "
                f"cannot create."
            )
        attr_creators[attr["type"]]()
        logger.debug(f"Created {attr['type']} attribute: {attr['key']}")

    def _create_attribute_by_type(
        self,
        database_id: str,
        collection_id: str,
        key: str,
        attr_type: str,
        is_array: bool,
    ):
        common_args = [database_id, collection_id, key]
        try:
            if attr_type == "string":
                result = self.databases.create_string_attribute(
                    *common_args, size=255, required=False, array=is_array
                )
            elif attr_type == "integer":
                result = self.databases.create_integer_attribute(
                    *common_args, required=False, array=is_array
                )
            elif attr_type == "boolean" and not is_array:
                result = self.databases.create_boolean_attribute(
                    *common_args, required=False
                )
            else:
                result = self.databases.create_string_attribute(
                    *common_args, size=255, required=False, array=is_array
                )
                logger.warning(
                    f"Unsupported type {attr_type} for {key}, defaulted to string"
                )
            logger.debug(f"Created {attr_type} attribute '{key}' (array: {is_array})")
            return result
        except AppwriteException as e:
            if "already exists" in exception_message(e).lower():
                logger.debug(f"Attribute '{key}' already exists")
                return None
            raise e

    # -- buckets -----------------------------------------------------------------

    def ensure_bucket(
        self,
        bucket_id: str,
        name: str = None,
        permissions: List[str] = None,
        file_security: bool = True,
        enabled: bool = True,
        maximum_file_size: int = None,
        allowed_file_extensions: List[str] = None,
        encryption: bool = False,
        compression: str = "none",
        antivirus: bool = True,
        create_metadata_db: bool = True,
    ) -> OperationResult:
        """Create a storage bucket, and optionally its metadata database."""
        try:
            if permissions is None:
                permissions = []
            if allowed_file_extensions is None:
                allowed_file_extensions = []

            result = self.storage.create_bucket(
                bucket_id=bucket_id,
                name=name,
                permissions=permissions,
                file_security=file_security,
                enabled=enabled,
                maximum_file_size=maximum_file_size,
                allowed_file_extensions=allowed_file_extensions,
                encryption=encryption,
                compression=compression,
                antivirus=antivirus,
            )

            if create_metadata_db:
                # C-238, fixed. The signature is (database_id, database_name) and this
                # passed (bucket_name, database_id), so `ensure-bucket` ensured a
                # database whose id was the BUCKET'S NAME. Against the shared production
                # project that was invisible, because the real database already existed.
                # Against an empty one — a new per-audience project — it creates a second,
                # junk database alongside the real one, and whether it lands at all
                # depends on whether the bucket's name happens to be a valid Appwrite id.
                # #331 carried the swap over verbatim under the rule that a relocation
                # must not change behaviour silently; þing-04 S17 is the first bring-up
                # against an empty project, which is where it stops being invisible.
                db_result = self.ensure_database(
                    self.config.database_id, self.config.database_name
                )
                if not db_result.success:
                    # Returned, not discarded. `ensure_database` reports failure by
                    # return value and never raises, so the enclosing
                    # `except AppwriteException` cannot see it: without this the bucket
                    # bring-up printed "OK (CREATED)" and exited 0 with no metadata
                    # database behind it. Tolerable while the swapped arguments above
                    # meant the lost failure was a JUNK database's; now that it is the
                    # real one, discarding it is C-291's shape — "reported OK while
                    # writing nothing" — which `ensure_attributes` refuses by name.
                    return OperationResult(
                        success=False,
                        error=(
                            f"bucket '{bucket_id}' was created, but its metadata "
                            f"database could not be: {db_result.error}. The bucket "
                            f"exists and the database does not, so a delivery into it "
                            f"would upload a file and then fail to record it."
                        ),
                        code=db_result.code or "METADATA_DATABASE_FAILED",
                    )
                result["metadata_database"] = db_result.data

            return OperationResult(success=True, data=result, code="CREATED")

        except AppwriteException as e:
            return OperationResult(success=False, error=e.message, code=e.type)


def build_provisioner(
    bucket_override: Optional[str] = None,
    collection_override: Optional[str] = None,
    collection_name_override: Optional[str] = None,
) -> AppwriteProvisioner:
    """Construct a provisioner from the validated process environment.

    `PredictionStoreConfig._ENV_MAP` names the `production_forecasts` shelf specifically,
    so without the overrides this can only ever provision that one. That is what stopped
    the first CRAF'd delivery: `file.py` tells the operator to run `ensure-collection`,
    and the command could not reach the collection it needed (#473).

    **A collection id and its name are one pair.** Supplying half of it silently targets
    whichever the environment happens to hold for the other — one variable away from
    provisioning `production_forecasts` while meaning a partner's shelf. Refused, in the
    same shape and for the same reason as `audit/targets.py:64-73` (C-250).

    That refusal is duplicated rather than shared: the audit package states it performs no
    writes and no provisioning, and importing it here would make that false. **Named
    trigger for extracting it:** a third caller needing pair-coherent coordinate
    resolution.
    """
    from views_pipeline_core.configs.prediction_store import PredictionStoreConfig
    from views_pipeline_core.exceptions.exceptions import ConfigurationException
    from views_pipeline_core.modules.appwrite.file import (
        AppWriteFileModule,
        AppwriteConfig,
    )

    if bool(collection_override) != bool(collection_name_override):
        supplied, missing = (
            ("collection", "collection-name")
            if collection_override
            else ("collection-name", "collection")
        )
        raise ConfigurationException(
            f"--{supplied} was given without --{missing}. A collection's id and its name "
            f"are one pair; supplying half of it leaves the other resolved from "
            f"APPWRITE_PROD_FORECASTS_*, so a command meaning a partner's shelf can "
            f"provision production_forecasts instead. Supply both, or neither."
        )

    store = PredictionStoreConfig.from_environment()
    config = AppwriteConfig(
        path_manager=None,
        endpoint=store.endpoint,
        project_id=store.project_id,
        credentials=store.api_key,
        auth_method="api_key",
        cache_dir=".appwrite_provisioning_cache",
        bucket_id=bucket_override or store.bucket_id,
        bucket_name=store.bucket_name,
        collection_id=collection_override or store.collection_id,
        collection_name=collection_name_override or store.collection_name,
        database_id=store.database_id,
        database_name=store.database_name,
    )
    return AppwriteProvisioner(AppWriteFileModule(config))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m views_pipeline_core.modules.appwrite.provisioning",
        description=(
            "Create Appwrite containers deliberately. The delivery path no longer "
            "provisions; run this once when setting up a project or a new bucket."
        ),
    )
    parser.add_argument(
        "action",
        choices=["ensure-bucket", "ensure-database", "ensure-collection"],
        help="What to create if it does not already exist.",
    )
    parser.add_argument("--bucket", default=None, help="Bucket id. Defaults to the env var.")
    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "Metadata collection id. Defaults to the env var. Must be given together "
            "with --collection-name."
        ),
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Metadata collection name. Must be given together with --collection.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    provisioner = build_provisioner(
        args.bucket, args.collection, args.collection_name
    )
    bucket_id = args.bucket or provisioner.config.bucket_id

    if args.action == "ensure-bucket":
        result = provisioner.ensure_bucket(
            bucket_id=bucket_id, name=provisioner.config.bucket_name
        )
    elif args.action == "ensure-database":
        result = provisioner.ensure_database()
    else:
        # `metadata={}` rather than the default None. `ensure_attributes` iterates
        # DECLARED_METADATA_ATTRIBUTES unconditionally, so an empty dict is what makes
        # the existing-collection path ensure the declared schema — matching the creation
        # path, which already passes `metadata or {}`. With the default, this command
        # returned OK(EXISTS) having written nothing, so a collection missing `file_hash`
        # stayed missing it while the operator was told it was provisioned (#473). Do NOT
        # pass DECLARED_METADATA_ATTRIBUTES here: it is a list, and the payload pass
        # iterates `metadata.items()`, so it raises `AttributeError` before any write.
        # (An earlier version of this comment said a list would be inferred as
        # `string(255)` junk attributes. The advice was right and the mechanism was
        # wrong — C-273's shape, in a comment written to prevent a mistake.) The empty
        # dict is now the COMPLETE bring-up, not a repair of seven fields
        # (þing-04 S17): the declared set covers every key a delivery document carries.
        result = provisioner.ensure_collection(metadata={})

    if result.success:
        logger.info("%s: OK (%s)", args.action, result.code or "CREATED")
        return 0
    logger.error("%s FAILED: code=%s error=%s", args.action, result.code, result.error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
