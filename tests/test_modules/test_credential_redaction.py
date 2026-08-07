"""Credentials must not be renderable by accident (#325, register C-230).

PLATFORM-001 §5's redaction clause is multi-carrier and binding: a credential in an env
var, a config field, a request header, a netrc entry or a keychain is never logged.
Endpoints may be.

Both config objects on this seam hold a live Appwrite API key in a dataclass field, and a
dataclass generates a ``__repr__`` that renders every field. So the key was one
``logger.debug(f"{config}")``, one W&B run-config capture, or one traceback rendering
locals away from the logs — with nothing in the type to stop it. The audit found no such
call today (recorded in the register), which makes this a latent exposure closed cheaply,
not an incident.

``repr=False`` closes the ACCIDENTAL rendering path only. ``asdict()``/``vars()`` still
return the value — see ``TestWhatRedactionDoesNotCover``, which pins that limit instead of
leaving it to be discovered.
"""

import pytest

from views_pipeline_core.configs.prediction_store import PredictionStoreConfig
from views_pipeline_core.modules.appwrite import AppwriteConfig, AuthMethod

_SECRET = "standard_e7c1a9b4deadbeefcafe0123456789abcdef0123456789abcdef0123456789ab"


@pytest.fixture
def appwrite_config():
    return AppwriteConfig(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="test_project",
        credentials=_SECRET,
        auth_method=AuthMethod.API_KEY,
        bucket_id="test_bucket",
        bucket_name="Test Bucket",
        collection_id="test_collection",
        collection_name="Test Collection",
        database_id="test_database",
        database_name="Test Database",
    )


@pytest.fixture
def store_config():
    return PredictionStoreConfig(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="test_project",
        api_key=_SECRET,
        bucket_id="test_bucket",
        bucket_name="Test Bucket",
        collection_id="test_collection",
        collection_name="Test Collection",
        database_id="test_database",
        database_name="Test Database",
    )


class TestAppwriteConfigRedaction:
    def test_repr_does_not_render_the_key(self, appwrite_config):
        assert _SECRET not in repr(appwrite_config)

    def test_fstring_does_not_render_the_key(self, appwrite_config):
        """The realistic leak: `logger.debug(f"config: {config}")`."""
        assert _SECRET not in f"{appwrite_config}"

    def test_str_does_not_render_the_key(self, appwrite_config):
        assert _SECRET not in str(appwrite_config)

    def test_a_structured_credential_is_redacted_whole(self, appwrite_config):
        """Redaction must not depend on the credential being a string.

        This test used to construct `AuthMethod.SESSION` with an email/password dict.
        Session auth was deleted in #344, so that mode no longer exists — but the
        property it was checking outlives it: `credentials` is typed loosely, and a
        future auth mode carrying a structured secret must not leak through a repr that
        only knew how to hide strings. Kept, re-pointed at the property rather than at
        the retired mode.
        """
        import dataclasses

        structured = dataclasses.replace(
            appwrite_config, credentials={"token": "hunter2", "scope": "files.read"}
        )
        assert "hunter2" not in repr(structured)

    def test_non_secret_fields_are_still_visible(self, appwrite_config):
        """Redaction must not blind the diagnostics it shares a line with.

        PLATFORM-001 §5: credentials are never logged; endpoints may be. A repr that
        hides the coordinates would push people back to printing the raw object.
        """
        rendered = repr(appwrite_config)
        assert "test_bucket" in rendered
        assert "https://cloud.appwrite.io/v1" in rendered

    def test_the_value_is_still_usable(self, appwrite_config):
        """Redaction is about rendering, not about access."""
        assert appwrite_config.credentials == _SECRET


class TestPredictionStoreConfigRedaction:
    """The sibling config for the same seam holds the same key."""

    def test_repr_does_not_render_the_key(self, store_config):
        assert _SECRET not in repr(store_config)

    def test_fstring_does_not_render_the_key(self, store_config):
        assert _SECRET not in f"{store_config}"

    def test_non_secret_fields_are_still_visible(self, store_config):
        rendered = repr(store_config)
        assert "test_bucket" in rendered
        assert "test_project" in rendered

    def test_the_value_is_still_usable(self, store_config):
        assert store_config.api_key == _SECRET


class TestWhatRedactionDoesNotCover:
    """The residual, characterised so it is visible rather than assumed.

    ``repr=False`` closes the *accidental* rendering path. It does not touch the
    *serialization* path: ``dataclasses.asdict()``, ``astuple()``, ``vars()`` and
    ``__dict__`` all still return the credential. These tests assert that limit rather
    than pretending it away — if a future change closes it, they turn red and should be
    inverted, and the register entry updated with them.

    Why this matters on one path in particular: ``modules/wandb/utils.py`` uses
    ``asdict(entry)`` at three sites and ``vars(item)`` at two — the W&B route C-230
    named as a hazard. Those helpers take ``EvaluationMetrics``, not configs, so nothing
    leaks today; the shape is what these tests pin.
    """

    def test_asdict_still_exposes_the_credential(self, appwrite_config):
        import dataclasses

        assert _SECRET in str(dataclasses.asdict(appwrite_config)), (
            "if this fails, asdict has been closed too — invert this test and update C-230"
        )

    def test_vars_still_exposes_the_credential(self, store_config):
        assert _SECRET in str(vars(store_config))

    def test_direct_field_access_is_unaffected(self, store_config):
        """Redaction governs rendering, never access — callers must still get the key."""
        assert store_config.api_key == _SECRET