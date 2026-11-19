"""Which tag counts as a repository's latest release (#612, register C-366, ADR-070).

The answer names the views-forecasts run (`v{XXYYZZ}_{year}_{month}`), so a wrong pick
files every `--prediction_store` run under a wrong name, silently. Before #612 the pick
was a string sort, and views-models' `backup/pre-sync-20260721` tag won it.
"""

import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from views_pipeline_core.managers.package.package import (
    PackageManager,
    latest_release_version,
)

#: The tags on views-models' remote, as `git ls-remote --tags` listed them on 2026-11-19.
VIEWS_MODELS_TAGS = ["0.1.0", "0.2.0", "0.2.1", "backup/pre-sync-20260721", "v0.0.1"]


# --- the rule ----------------------------------------------------------------------


def test_the_views_models_tags_resolve_to_the_newest_release():
    assert latest_release_version(VIEWS_MODELS_TAGS) == "0.2.1"


def test_versions_compare_as_numbers_not_strings():
    assert latest_release_version(["0.9.0", "0.10.0"]) == "0.10.0"
    assert latest_release_version(["v3.9.0", "v3.10.0"]) == "3.10.0"


def test_versions_with_three_digit_parts_compare_numerically():
    assert latest_release_version(["99.0.0", "100.0.0"]) == "100.0.0"
    assert latest_release_version(["1.99.0", "1.100.0"]) == "1.100.0"


def test_the_v_prefix_is_ignored_for_comparison_and_dropped_from_the_answer():
    assert latest_release_version(["v1.0.0", "0.9.9"]) == "1.0.0"


@pytest.mark.parametrize(
    "not_a_release",
    [
        "backup/pre-sync-20260721", "latest", "v1.0", "1.0.0-rc1", "1.0.0^{}", "",
        "vv1.0.0", "V1.0.0", "９.９.９",  # fullwidth digits are not ASCII
    ],
)
def test_a_tag_that_is_not_a_release_version_is_never_chosen(not_a_release):
    assert latest_release_version([not_a_release]) is None
    assert latest_release_version(["0.1.0", not_a_release]) == "0.1.0"


# --- the lookup: git ls-remote -----------------------------------------------------


def _ls_remote(tags):
    return "".join(f"abc123\trefs/tags/{t}\n" for t in tags).encode()


def test_the_lookup_compares_tags_numerically_through_the_method():
    tags = ["0.9.0", "v0.10.0", "backup/x"]
    with patch("subprocess.check_output", return_value=_ls_remote(tags)) as git, \
            patch("requests.get") as api:
        assert PackageManager.get_latest_release_version_from_github("views-models") == "0.10.0"
    api.assert_not_called()
    assert "https://github.com/views-platform/views-models" in git.call_args.args[0]


def test_the_views_models_tags_resolve_through_the_method():
    with patch("subprocess.check_output", return_value=_ls_remote(VIEWS_MODELS_TAGS)), \
            patch("requests.get") as api:
        assert PackageManager.get_latest_release_version_from_github("views-models") == "0.2.1"
    api.assert_not_called()


def test_a_tag_whose_name_contains_refs_tags_is_not_read_as_that_version():
    """`git tag backup/refs/tags/9.9.9` is legal; splitting on the last `refs/tags/`
    used to read it as release 9.9.9."""
    for planted in ("backup/refs/tags/9.9.9", "refs/tags/9.9.9"):
        with patch("subprocess.check_output", return_value=_ls_remote(["0.2.1", planted])), \
                patch("requests.get"):
            assert PackageManager.get_latest_release_version_from_github("views-models") == "0.2.1", planted


def test_tags_with_no_release_return_none_without_asking_the_api(caplog):
    """Every GitHub release has a tag, so the API cannot know a version git did not list."""
    with patch("subprocess.check_output", return_value=_ls_remote(["backup/x", "latest"])), \
            patch("requests.get") as api, caplog.at_level(logging.WARNING):
        assert PackageManager.get_latest_release_version_from_github("views-models") is None
    api.assert_not_called()
    assert "no release tag" in caplog.text


# --- the lookup: GitHub releases API, only when git fails --------------------------


def _git_fails(*_args, **_kwargs):
    raise subprocess.CalledProcessError(128, "git ls-remote")


def _api(payload, status=200):
    response = MagicMock(status_code=status, headers={}, text="body")
    response.json.return_value = payload
    return response


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"tag_name": "v0.3.0", "name": "0.2.0"}, "0.3.0"),  # tag_name wins when it parses...
        ({"tag_name": "v0.2.0", "name": "0.3.0"}, "0.2.0"),  # ...even when name is higher
        ({"tag_name": "nightly", "name": "v0.2.0"}, "0.2.0"),  # name only when tag_name is not a version
        ({"tag_name": "backup/x", "name": "nightly"}, None),  # neither: refused, not returned
    ],
)
def test_when_git_fails_the_api_is_asked_under_the_same_rule(payload, expected):
    with patch("subprocess.check_output", side_effect=_git_fails), \
            patch("requests.get", return_value=_api(payload)) as api:
        assert PackageManager.get_latest_release_version_from_github("views-models") == expected
    api.assert_called_once()
    assert api.call_args.args[0] == (
        "https://api.github.com/repos/views-platform/views-models/releases/latest"
    )


def test_a_failed_api_call_is_logged_not_lost(caplog):
    """The non-200 branch passed two positional strings to `logging.error`; the logging
    call itself failed and the message never reached a handler."""
    with patch("subprocess.check_output", side_effect=_git_fails), \
            patch("requests.get", return_value=_api({}, status=404)), \
            caplog.at_level(logging.ERROR):
        assert PackageManager.get_latest_release_version_from_github("views-models") is None
    assert "Failed to get latest version from GitHub: 404" in caplog.text
