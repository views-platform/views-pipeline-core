"""
Falsification audit: "We 100% understand the impact, consequences, and
splash zone of this sprint"
Generated: 2026-06-06

Claim: The 5-item sprint (C-166, C-59, C-26, C-144, C-47) has a fully
understood impact, with no blind spots in its consequences or splash zone.

Verdict: FALSIFIED — 2 hard falsifications, 2 soft falsifications.

P3 (HARD): C-144 as planned violates ADR-054's constitutional prohibition
    "Pipeline-core NEVER depends on views-reporting." The ADR makes no
    distinction between hard and optional dependencies.

P4 (HARD): Sprint plan specifies PEP 621 [project.optional-dependencies]
    syntax, but pyproject.toml uses Poetry format. Correct syntax is
    [tool.poetry.extras]. Plan would produce a no-op or error.

P2 (SOFT): C-166 sprint plan catches requests.RequestException, but
    `requests` is a transitive dependency (via appwrite SDK) not declared
    in pyproject.toml. The catch list relies on an undeclared package.

P6 (SOFT): io.py has 3 except Exception blocks (lines ~103, ~140, ~210),
    not the 1 documented in the sprint plan. Sprint targets line ~140 only,
    leaving 2 blanket catches untouched at lines ~103 and ~210. Total
    across the upload path is 11 blocks; sprint addresses 2.
"""

import ast
from pathlib import Path


# ---------------------------------------------------------------------------
# P3: ADR-054 prohibits ANY dependency on views-reporting
# ---------------------------------------------------------------------------


def test_p3_adr054_prohibits_optional_views_reporting_dependency():
    """
    P3 (HARD FALSIFICATION): ADR-054 line 80 states 'Pipeline-core NEVER
    depends on views-reporting.' The word NEVER admits no optional/hard
    distinction. Adding views-reporting as an optional extra in pyproject.toml
    violates this constitutional ADR.

    Sprint item C-144 must either:
    (a) amend ADR-054 first (with a new ADR superseding the constraint), or
    (b) be dropped from the sprint.

    This test documents the constraint. It always passes — it exists to
    ensure future sprint planners find this governance prohibition.
    """
    adr_path = (
        Path(__file__).resolve().parents[1]
        / "documentation"
        / "ADRs"
        / "054_visualization_and_reporting_extraction.md"
    )
    if adr_path.exists():
        text = adr_path.read_text()
        assert "NEVER" in text and "views-reporting" in text, (
            "ADR-054 should contain the NEVER-depends constraint. "
            "If this assertion fails, the ADR may have been amended — "
            "re-evaluate C-144."
        )
    # If ADR doesn't exist, the constraint stands by convention.
    assert True


# ---------------------------------------------------------------------------
# P4: pyproject.toml uses Poetry, not PEP 621
# ---------------------------------------------------------------------------


def test_p4_pyproject_uses_poetry_format():
    """
    P4 (HARD FALSIFICATION): Sprint plan specified [project.optional-dependencies]
    (PEP 621) but pyproject.toml uses [tool.poetry.dependencies] (Poetry format).

    The correct syntax for optional extras in Poetry is [tool.poetry.extras].
    Using PEP 621 syntax in a Poetry project is either a no-op or an error.

    This test verifies the project format so future sprint plans use the
    correct packaging syntax.
    """
    pyproject_path = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    )
    text = pyproject_path.read_text()

    assert "[tool.poetry]" in text, (
        "pyproject.toml should use Poetry format ([tool.poetry]). "
        "If this changes to PEP 621, sprint packaging instructions must update."
    )
    assert "[tool.poetry.dependencies]" in text, (
        "Dependencies should be under [tool.poetry.dependencies]."
    )
    # Verify PEP 621 tables are NOT present (they'd be ignored by Poetry)
    assert "[project.optional-dependencies]" not in text, (
        "PEP 621 [project.optional-dependencies] found in a Poetry project. "
        "Use [tool.poetry.extras] instead."
    )
    assert "[project]" not in text or "[tool.poetry]" in text, (
        "Mixed PEP 621 / Poetry format detected."
    )


# ---------------------------------------------------------------------------
# P2: requests is transitive, not declared
# ---------------------------------------------------------------------------


def test_p2_requests_not_declared_dependency():
    """
    P2 (SOFT FALSIFICATION): Sprint plan for C-166 catches
    requests.RequestException, but `requests` is not declared in
    pyproject.toml — it's a transitive dependency via the appwrite SDK.

    Catching exceptions from an undeclared dependency is fragile: if
    appwrite drops requests (e.g., switches to httpx), the catch clause
    breaks at import time.

    This test documents the transitive dependency status.
    """
    pyproject_path = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    )
    text = pyproject_path.read_text()

    # requests should NOT be in direct dependencies
    # (if it gets added, this test should be updated)
    lines = text.split("\n")
    in_deps = False
    declared_packages = []
    for line in lines:
        if line.strip() == "[tool.poetry.dependencies]":
            in_deps = True
            continue
        if in_deps and line.strip().startswith("["):
            break
        if in_deps and "=" in line:
            pkg = line.split("=")[0].strip().strip('"')
            declared_packages.append(pkg)

    assert "requests" not in declared_packages, (
        "requests is now a direct dependency — update C-166 sprint notes. "
        "If requests is declared, catching requests.RequestException is safe."
    )


# ---------------------------------------------------------------------------
# P6: io.py has 3 except-Exception blocks, not 1
# ---------------------------------------------------------------------------


def test_p6_io_except_exception_count():
    """
    P6 (SOFT FALSIFICATION): Sprint plan documents 1 except-Exception block
    in io.py (the _upload_to_prediction_store catch at line ~140). Original
    count was 3; C-166 narrowed the line ~140 block, leaving 2:
      - Line ~103: save_predictions outer catch (re-raises as PipelineException)
      - Line ~210: save_evaluations outer catch (re-raises as PipelineException)

    These 2 re-raise as PipelineException so they don't swallow errors,
    but they ARE still overly broad catches.

    This test documents the actual count for future sprint planners.
    """
    io_path = (
        Path(__file__).resolve().parents[1]
        / "views_pipeline_core"
        / "managers"
        / "prediction"
        / "io.py"
    )
    source = io_path.read_text()
    tree = ast.parse(source)

    except_exception_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is not None:
                if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    except_exception_count += 1

    assert except_exception_count >= 2, (
        f"Expected at least 2 except-Exception blocks in io.py, "
        f"found {except_exception_count}. If blocks were narrowed, "
        f"update this test and the sprint documentation."
    )