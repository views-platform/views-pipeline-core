"""Version-lookup helper for Poetry packages.

Extracted from ``PackageManager.get_latest_release_version_from_github``
per the M-4 audit decision. The helper tries ``git ls-remote`` first
and falls back to the GitHub API. ``requests`` is imported lazily
inside the function so that importing this module doesn't eagerly pull
the HTTP library.
"""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


def get_latest_release_version_from_github(
    repository_name: str,
    organization_name: str = "views-platform",
) -> Optional[str]:
    """Fetch the latest release version of a repository from GitHub.

    Tries ``git ls-remote --tags`` first (no rate limit). If that fails,
    falls back to the GitHub API ``/releases/latest`` endpoint.

    Args:
        repository_name: The name of the repository (e.g. ``"views-pipeline-core"``).
        organization_name: The GitHub organization name. Defaults to
            ``"views-platform"``.

    Returns:
        The latest release tag (without the leading ``v``), or ``None``
        if no releases are found or the rate limit is hit.
    """
    repo_url = f"https://github.com/{organization_name}/{repository_name}"

    # Step 1: Try git ls-remote (no rate limit, works for public repos)
    try:
        cmd = f"git ls-remote --tags {repo_url}"
        output = subprocess.check_output(cmd, shell=True).decode()
        tags = [
            line.split("refs/tags/")[-1]
            for line in output.split("\n")
            if "refs/tags/" in line
        ]
        if tags:
            latest_tag = sorted(tags, key=lambda v: v.lstrip("v"))[-1]
            return latest_tag.lstrip("v")
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"Failed to get latest version using `git ls-remote`: {e}. "
            f"Falling back to GitHub API."
        )

    # Step 2: Fall back to GitHub API
    github_url = (
        f"https://api.github.com/repos/{organization_name}/"
        f"{repository_name}/releases/latest"
    )
    try:
        # Lazy import so this module doesn't eagerly pull `requests`.
        import requests
    except ImportError as e:
        logger.error(
            f"Cannot fetch latest release version: `requests` is not installed. {e}"
        )
        return None

    try:
        response = requests.get(github_url)

        if response.status_code == 200:
            data = response.json()
            if "tag_name" in data and data["tag_name"] != "":
                return data["tag_name"].lstrip("v")
            elif "name" in data and data["name"] != "":
                return data["name"].lstrip("v")
            else:
                logger.error("No releases found for this repository.")
                return None

        elif response.status_code == 403 and "X-RateLimit-Reset" in response.headers:
            reset_time = int(response.headers["X-RateLimit-Reset"])
            logger.error(
                f"API rate limit exceeded. Retry after "
                f"{reset_time - int(time.time())} seconds.",
                exc_info=False,
            )
            return None

        else:
            logger.error(
                f"Failed to get latest version from GitHub: {response.status_code}, "
                f"Response: {response.text}",
                exc_info=False,
            )
            return None

    except Exception as e:
        logger.error(
            f"An error occurred while getting the latest version from GitHub: {e}",
            exc_info=False,
        )
        return None