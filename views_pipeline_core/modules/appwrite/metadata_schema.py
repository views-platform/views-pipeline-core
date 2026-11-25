"""The declared columns of a delivery metadata row (one table per audience).

The single place the row's shape is written down. The provisioner creates these columns;
the delivery preflight in `file.py` refuses to upload into a table that lacks any of them
(#480, #604), because a missing column lets the file upload and then rejects its row,
leaving an orphan in the bucket. The serving APIs and the postprocessing delivery read
rows by these names, so renaming one is a cross-repository change.
"""

from typing import Any, Dict, List

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

#: The column keys a delivery writes; what the preflight checks a table against.
DECLARED_METADATA_KEYS = frozenset(a["key"] for a in DECLARED_METADATA_ATTRIBUTES)
