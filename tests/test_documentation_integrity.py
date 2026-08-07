"""Gate: documentation/validate_docs.sh must pass before shipping."""
import subprocess
from pathlib import Path


def test_validate_docs():
    script = Path(__file__).resolve().parents[1] / "documentation" / "validate_docs.sh"
    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"validate_docs.sh failed:\n{result.stdout}\n{result.stderr}"
    )