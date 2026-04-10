"""Boundary enforcement tests — verify ADR-002 layer dependency rules.

Inner layers (data/, configs/, exceptions/) must not import from outer layers
(managers/, modules/). modules/ must not import from managers/.

Exemptions per ADR-002 Rule 4: TYPE_CHECKING imports and lazy importlib imports
are permitted for type annotation purposes only.
"""
import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG = _REPO_ROOT / "views_pipeline_core"


def _collect_imports(filepath: Path) -> list[tuple[str, bool]]:
    """Parse a Python file and return (module_path, in_type_checking) tuples.

    Returns all ``import X`` and ``from X import Y`` statements with a flag
    indicating whether they appear inside an ``if TYPE_CHECKING:`` block.
    """
    source = filepath.read_text()
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    results = []

    # Find TYPE_CHECKING block line ranges
    tc_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            is_tc = False
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                is_tc = True
            elif isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
                is_tc = True
            if is_tc:
                end = max(
                    getattr(child, "end_lineno", child.lineno)
                    for child in node.body
                )
                tc_ranges.append((node.lineno, end))

    def _in_tc(lineno: int) -> bool:
        return any(start <= lineno <= end for start, end in tc_ranges)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((alias.name, _in_tc(node.lineno)))
        elif isinstance(node, ast.ImportFrom) and node.module:
            results.append((node.module, _in_tc(node.lineno)))

    return results


def _check_layer_violations(
    source_dir: Path,
    forbidden_prefixes: list[str],
    layer_name: str,
) -> list[str]:
    """Scan source_dir for imports that violate the dependency rule."""
    violations = []
    for py_file in sorted(source_dir.rglob("*.py")):
        imports = _collect_imports(py_file)
        for module_path, in_type_checking in imports:
            if in_type_checking:
                continue  # ADR-002 Rule 4 exemption
            for prefix in forbidden_prefixes:
                if module_path.startswith(prefix):
                    rel = py_file.relative_to(_REPO_ROOT)
                    violations.append(
                        f"{rel}: imports '{module_path}' "
                        f"({layer_name} must not import from {prefix})"
                    )
    return violations


class TestInnerLayerBoundaries:
    """data/ and configs/ (Layers 0-1) must not import from managers/ or modules/."""

    def test_data_does_not_import_managers(self):
        violations = _check_layer_violations(
            _PKG / "data",
            ["views_pipeline_core.managers"],
            "data/ (Layer 1)",
        )
        assert violations == [], (
            "Layer violation: data/ imports managers/:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_data_does_not_import_modules(self):
        violations = _check_layer_violations(
            _PKG / "data",
            ["views_pipeline_core.modules"],
            "data/ (Layer 1)",
        )
        # Known deviations documented in ADR-002:
        # - handlers.py imports modules/statistics (MAP computation)
        # - handlers.py imports modules/visualizations (distribution plots)
        violations = [
            v for v in violations
            if "modules.statistics" not in v
            and "modules.visualizations" not in v
        ]
        assert violations == [], (
            "Layer violation: data/ imports modules/:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_configs_does_not_import_managers(self):
        violations = _check_layer_violations(
            _PKG / "configs",
            ["views_pipeline_core.managers"],
            "configs/ (Layer 0-1)",
        )
        # Known deviation documented in ADR-002:
        # - pipeline.py lazily imports managers/package/PackageManager for version fetching
        violations = [
            v for v in violations
            if "managers.package" not in v
        ]
        assert violations == [], (
            "Layer violation: configs/ imports managers/:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


class TestModuleLayerBoundaries:
    """modules/ (Layers 3-4) must not import from managers/ (Layers 5-6)."""

    def test_modules_does_not_import_managers(self):
        violations = _check_layer_violations(
            _PKG / "modules",
            ["views_pipeline_core.managers"],
            "modules/ (Layer 3-4)",
        )
        # Known deviations (all deferred imports, documented in ADR-002/risk register):
        # - appwrite/file.py imports managers.model (ModelPathManager for cache path)
        # - logging/logging.py imports managers.model (ModelPathManager for log path)
        # - validation/ensemble/check.py imports managers.model + managers.ensemble
        _known = {
            "modules/appwrite/file.py",
            "modules/logging/logging.py",
            "modules/validation/ensemble/check.py",
        }
        violations = [
            v for v in violations
            if not any(known in v for known in _known)
        ]
        assert violations == [], (
            "Layer violation: modules/ imports managers/:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
