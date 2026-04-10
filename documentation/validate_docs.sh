#!/usr/bin/env bash
# Validates internal consistency of base_docs documentation set.
# Exit 0 if clean, exit 1 if issues found.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

errors=0

echo "=== base_docs validation ==="
echo ""

# 1. Check for unfilled template placeholders in accepted/active files
#    (skip files whose names contain "template" — those are expected to have placeholders)
#    These are warnings only (non-blocking) since in the template repo some
#    files are legitimately Accepted with placeholder dates.
echo "--- Checking for template placeholders in accepted/active files ---"
warnings=0
while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    [[ "$file" == *template* ]] && continue
    if grep -q 'YYYY-MM-DD' "$file"; then
        echo "  WARN: Unfilled date placeholder in $file"
        warnings=$((warnings + 1))
    fi
    if grep -q '<roles / team>' "$file"; then
        echo "  WARN: Unfilled deciders placeholder in $file"
        warnings=$((warnings + 1))
    fi
    if grep -q '<ClassName>' "$file"; then
        echo "  WARN: Unfilled ClassName placeholder in $file"
        warnings=$((warnings + 1))
    fi
done < <(grep -rl 'Status:.*\(Accepted\|Active\)' --include='*.md' . 2>/dev/null || true)
if [ "$warnings" -eq 0 ]; then
    echo "  OK"
fi

# 2. Verify CIC active contracts exist (skip blockquote/example lines)
echo "--- Checking CIC active contract references ---"
if [ -f "CICs/README.md" ]; then
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        contract=$(echo "$line" | sed -n 's/^- `\(.*\.md\)`.*$/\1/p')
        if [ -n "$contract" ] && [ ! -f "CICs/$contract" ]; then
            echo "  ERROR: CIC contract listed but missing: CICs/$contract"
            errors=$((errors + 1))
        fi
    done < <(grep -E '^- `[^`]+\.md`' CICs/README.md 2>/dev/null | grep -v '>' || true)
fi

# 3a. Cross-ADR reference integrity — verify ALL ADR-NNN text references resolve to files
echo "--- Checking cross-ADR references ---"
while IFS= read -r ref; do
    [[ -z "$ref" ]] && continue
    file=$(echo "$ref" | cut -d: -f1)
    # Check every ADR-NNN on the line, not just the first
    for adr_tag in $(echo "$ref" | grep -oE 'ADR-0[0-9][0-9]'); do
        adr_num=${adr_tag#ADR-}
        match_count=$(find ADRs -name "${adr_num}_*.md" 2>/dev/null | wc -l)
        if [ "$match_count" -eq 0 ]; then
            echo "  ERROR: $file references ${adr_tag} but no matching file found in ADRs/"
            errors=$((errors + 1))
        fi
    done
done < <(grep -rnE 'ADR-0[0-9][0-9]' --include='*.md' . 2>/dev/null || true)

# 3b. Verify ADR file path references (e.g., ADRs/042_foo.md) resolve to real files
#     Scans both documentation/ and repo root (for READMEs in source tree)
echo "--- Checking ADR file path references ---"
while IFS= read -r ref; do
    [[ -z "$ref" ]] && continue
    file=$(echo "$ref" | cut -d: -f1)
    for adr_path in $(echo "$ref" | grep -oE 'ADRs/[0-9][0-9a-z_]+\.md'); do
        if [ ! -f "$adr_path" ] && [ ! -f "documentation/$adr_path" ]; then
            echo "  ERROR: $file references $adr_path but file does not exist"
            errors=$((errors + 1))
        fi
    done
done < <(grep -rnE 'ADRs/[0-9][0-9a-z_]+\.md' --include='*.md' . "$SCRIPT_DIR/.." 2>/dev/null | sort -u || true)

# 3c. Verify relative ADR links (bare NNN_foo.md inside ADRs/) resolve to real files
echo "--- Checking relative ADR links ---"
while IFS= read -r ref; do
    [[ -z "$ref" ]] && continue
    file=$(echo "$ref" | cut -d: -f1)
    for adr_file in $(echo "$ref" | grep -oE '\(0[0-9][0-9]_[^)]+\.md\)' | tr -d '()'); do
        if [ ! -f "ADRs/$adr_file" ]; then
            echo "  ERROR: $file references $adr_file but file does not exist in ADRs/"
            errors=$((errors + 1))
        fi
    done
done < <(grep -rnE '\(0[0-9][0-9]_[^)]+\.md\)' --include='*.md' ADRs/ 2>/dev/null || true)

# 4. Check that referenced protocol files exist
echo "--- Checking protocol file references ---"
while IFS= read -r ref; do
    [[ -z "$ref" ]] && continue
    file=$(echo "$ref" | cut -d: -f1)
    proto=$(echo "$ref" | grep -oE 'contributor_protocols/[a-z_]+\.md' | head -1)
    if [ -n "$proto" ] && [ ! -f "$proto" ]; then
        echo "  ERROR: $file references $proto but file does not exist"
        errors=$((errors + 1))
    fi
done < <(grep -rn 'contributor_protocols/' --include='*.md' . 2>/dev/null || true)

# 5. Verify README.md file links resolve to real files
echo "--- Checking README.md file links ---"
ROOT_README="$SCRIPT_DIR/../README.md"
if [ -f "$ROOT_README" ]; then
    while IFS= read -r link; do
        [[ -z "$link" ]] && continue
        # Resolve relative to repo root
        target="$SCRIPT_DIR/../$link"
        if [ ! -e "$target" ]; then
            echo "  ERROR: README.md references $link but file does not exist"
            errors=$((errors + 1))
        fi
    done < <(grep -oE '\./views_pipeline_core/[^)]+' "$ROOT_README" 2>/dev/null || true)
fi

# 6. Report template status markers
echo "--- Checking template status markers ---"
template_count=$(grep -rl '\-\-template\-\-' --include='*.md' . 2>/dev/null | wc -l)
echo "  INFO: $template_count files still have --template-- status (expected in template repo)"

echo ""
if [ "$errors" -gt 0 ]; then
    echo "=== FAILED: $errors issue(s) found ==="
    exit 1
else
    echo "=== PASSED: no issues found ==="
    exit 0
fi
