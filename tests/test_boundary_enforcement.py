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


def _collect_imports(
    filepath: Path,
) -> list[tuple[str, tuple[str, ...], bool]]:
    """Parse a Python file and return (module_path, names, in_type_checking).

    ``names`` is the tuple of imported symbol names for ``from X import a, b``
    statements, or an empty tuple for bare ``import X`` statements. The
    ``in_type_checking`` flag indicates whether the import appears inside an
    ``if TYPE_CHECKING:`` block (ADR-002 Rule 4 exemption).
    """
    source = filepath.read_text()
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    results: list[tuple[str, tuple[str, ...], bool]] = []

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
                results.append((alias.name, (), _in_tc(node.lineno)))
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = tuple(alias.name for alias in node.names)
            results.append((node.module, names, _in_tc(node.lineno)))

    return results


def _check_layer_violations(
    source_dir: Path,
    forbidden_prefixes: list[str],
    layer_name: str,
    granular_exemptions: dict[tuple[str, str], set[str]] | None = None,
) -> list[str]:
    """Scan source_dir for imports that violate the dependency rule.

    ``granular_exemptions`` maps ``(file_suffix, forbidden_module_prefix)`` to
    the set of symbol names that may be imported from that prefix in that file.
    Any imported name not in the allowed set still counts as a violation — this
    prevents exemptions from silently widening over time.
    """
    granular_exemptions = granular_exemptions or {}
    violations = []
    for py_file in sorted(source_dir.rglob("*.py")):
        imports = _collect_imports(py_file)
        rel = py_file.relative_to(_REPO_ROOT)
        rel_str = str(rel)
        for module_path, names, in_type_checking in imports:
            if in_type_checking:
                continue  # ADR-002 Rule 4 exemption
            for prefix in forbidden_prefixes:
                if not module_path.startswith(prefix):
                    continue
                allowed: set[str] | None = None
                for (file_suffix, exempt_prefix), allowed_names in granular_exemptions.items():
                    if rel_str.endswith(file_suffix) and module_path.startswith(exempt_prefix):
                        allowed = allowed_names
                        break
                if allowed is not None and names and all(n in allowed for n in names):
                    continue
                if allowed is not None and names:
                    disallowed = [n for n in names if n not in allowed]
                    violations.append(
                        f"{rel}: imports {disallowed} from '{module_path}' "
                        f"({layer_name}: exempted names are {sorted(allowed)}, "
                        f"but {disallowed} are not exempt)"
                    )
                else:
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
        # C-50: validate_ensemble_model in modules/validation/ensemble/check.py
        # instantiates Layer 6 orchestrators (ModelManager, EnsembleManager,
        # EnsemblePathManager) to run partition-alignment validation. This is an
        # architectural smell (L2/L5 validator reaching up to L6 orchestrator)
        # tracked as a forward-looking risk; correct fix is moving the function
        # into managers/ensemble/ with a deprecation shim (separate PR).
        # The exemption is granular: any additional name imported from these
        # modules still fails the test.
        granular_exemptions = {
            (
                "modules/validation/ensemble/check.py",
                "views_pipeline_core.managers.model",
            ): {"ModelManager"},
            (
                "modules/validation/ensemble/check.py",
                "views_pipeline_core.managers.ensemble",
            ): {"EnsembleManager", "EnsemblePathManager"},
        }
        violations = _check_layer_violations(
            _PKG / "modules",
            ["views_pipeline_core.managers"],
            "modules/ (Layer 3-4)",
            granular_exemptions=granular_exemptions,
        )
        assert violations == [], (
            "Layer violation: modules/ imports managers/:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


class TestUtilityLayerBoundaries:
    """files/ (Layer 2) must not import from managers/ (Layers 5-7)."""

    def test_files_does_not_import_managers(self):
        violations = _check_layer_violations(
            _PKG / "files",
            ["views_pipeline_core.managers"],
            "files/ (Layer 2)",
        )
        assert violations == [], (
            "Layer violation: files/ imports managers/:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
