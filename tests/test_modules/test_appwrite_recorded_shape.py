"""Drive the paged walker from a RECORDED Appwrite response. Story #348, register C-218.

The fixture is captured from a live instance by
`tests/fixtures/appwrite/capture_list_documents.py`. Until an operator runs that script
these tests **skip**, and say so — a skip that explains itself is honest; a test that
passes against a fixture I invented would be C-218 rebuilt with more steps.

The redaction tests below do NOT skip. They pin the capture tool itself, and they can run
without credentials because they feed it a synthetic response full of things that must
never reach git.
"""

import importlib.util
import json
import pathlib

import pytest

FIXTURE_DIR = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "appwrite"
FIXTURE = FIXTURE_DIR / "list_documents_shape.json"
CAPTURE_SCRIPT = FIXTURE_DIR / "capture_list_documents.py"

_NEEDS_CAPTURE = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=(
        "no recorded Appwrite response yet — run "
        "`python tests/fixtures/appwrite/capture_list_documents.py` with credentials "
        "loaded and commit the result (story #348, part b)"
    ),
)


def _capture_module():
    spec = importlib.util.spec_from_file_location("_capture", CAPTURE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheRedactorCannotLeak:
    """Runs today, without credentials. Pins the tool, not the fixture."""

    _SECRETS = [
        "production_forecasts",
        "lr_ged_sb_m000543.parquet",
        "68f2c1a9b3d4e5f6",
        "simmaa@prio.org",
        "standard_api_key_abc123",
    ]

    def _documents(self):
        return [
            {
                "$id": self._SECRETS[2],
                "filename": self._SECRETS[1],
                "bucketId": self._SECRETS[0],
                "owner": self._SECRETS[3],
                "key": self._SECRETS[4],
                "file_size": 91234,
                "targets": ["ged_sb", "ged_ns"],
                "$createdAt": "2026-07-27T11:04:22.517+00:00",
                "reconciled": True,
                "note": None,
            }
            for _ in range(25)
        ]

    def test_no_field_value_survives_redaction(self):
        capture = _capture_module()
        rendered = json.dumps(capture._shape_of(self._documents()))

        leaked = [s for s in self._SECRETS if s in rendered]
        assert not leaked, f"the capture tool would have written these to git: {leaked}"

    def test_the_shape_is_still_useful_after_redaction(self):
        """Redaction that destroyed the shape would make the fixture worthless."""
        capture = _capture_module()
        shape = capture._shape_of(self._documents())

        assert set(shape) >= {"$id", "filename", "file_size", "targets", "$createdAt"}
        assert shape["file_size"] == ["int"]
        assert shape["targets"] == ["list[str:short]"]
        assert shape["$createdAt"] == ["str:short:iso8601"]
        assert shape["note"] == ["null"]

    def test_a_coordinate_is_fingerprinted_not_published(self):
        capture = _capture_module()
        fingerprint = capture._fingerprint("production_forecasts")

        assert "production_forecasts" not in fingerprint
        assert fingerprint == capture._fingerprint("production_forecasts"), "unstable"
        assert len(fingerprint) == 12


@_NEEDS_CAPTURE
class TestTheRecordedShape:
    """Activates the moment the fixture is committed. No further work needed."""

    @pytest.fixture
    def recorded(self):
        return json.loads(FIXTURE.read_text())

    def test_it_records_what_happens_with_no_limit_supplied(self, recorded):
        """The single fact that would have prevented the 436-orphan incident."""
        probe = recorded["no_limit_supplied"]

        assert "documents_returned" in probe and "total_reported" in probe
        assert isinstance(probe["documents_returned"], int)

    def test_our_page_size_belief_matches_what_the_service_did(self, recorded):
        """If the service truncated, our constant must equal the observed truncation.

        This is the assertion the whole story exists for: `APPWRITE_DEFAULT_PAGE_SIZE`
        is currently 25 because that is what the documentation and a sibling repo's
        comment say. Here it is checked against what the service actually did.
        """
        from views_pipeline_core.modules.appwrite.file import APPWRITE_DEFAULT_PAGE_SIZE

        probe = recorded["no_limit_supplied"]
        returned, total = probe["documents_returned"], probe["total_reported"]

        if total is not None and returned < total:
            assert returned == APPWRITE_DEFAULT_PAGE_SIZE, (
                f"the service returned {returned} of {total} documents with no limit "
                f"supplied, but APPWRITE_DEFAULT_PAGE_SIZE is "
                f"{APPWRITE_DEFAULT_PAGE_SIZE}. Our constant is wrong, and every fake "
                f"built on it is wrong the same way (C-218)."
            )
        else:
            pytest.skip(
                f"the collection holds {total} documents, which fits in one page — this "
                "capture cannot observe truncation. Re-capture against a collection "
                "with more than one page to close C-218 fully."
            )

    def test_an_explicit_limit_was_honoured(self, recorded):
        probe = recorded["explicit_limit"]
        assert probe["documents_returned"] <= probe["limit_requested"], (
            "the service returned MORE rows than the limit requested — every paged walk "
            "in this repo assumes it cannot"
        )

    def test_the_fields_our_code_reads_were_present(self, recorded):
        """`reconcile` and the metadata walkers dereference these by name."""
        shape = recorded["no_limit_supplied"]["document_shape"]
        if not shape:
            pytest.skip("the collection was empty at capture time")

        for field in ("$id", "$createdAt"):
            assert field in shape, f"{field} absent from the recorded documents"

    def test_provenance_is_recorded_without_publishing_a_coordinate(self, recorded):
        provenance = recorded["provenance"]

        assert provenance["captured_at"]
        assert len(provenance["project_fingerprint"]) == 12
        blob = json.dumps(recorded)
        assert "APPWRITE_" not in blob or "APPWRITE_REQUEST" in blob
