#!/usr/bin/env bash
# Check every shipped frozen preregistration file (plans, predictions, specs, tolerances, code-hash records)
# against the sha256 recorded in prereg/SHA256SUMS. The frozen files listed in prereg/SHA256SUMS.withheld are
# not in this release; each of them is checked against its recorded sha256 as soon as it is present at its path.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if command -v sha256sum >/dev/null 2>&1; then
  check() { sha256sum -c "$@"; }
else
  check() { shasum -a 256 -c "$@"; }
fi
check prereg/SHA256SUMS
if [ -f prereg/SHA256SUMS.withheld ]; then
  while read -r sum path; do
    [ -n "${sum:-}" ] || continue
    if [ -f "$path" ]; then
      printf '%s  %s\n' "$sum" "$path" | check -
    else
      echo "$path: WITHHELD (not in this release)"
    fi
  done < prereg/SHA256SUMS.withheld
fi
