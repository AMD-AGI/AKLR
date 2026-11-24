# SPDX-License-Identifier: MIT
"""Print the contract IDs eligible for scoped op baseline/reference tests (empty = full)."""

import json
import os
import re
import subprocess
import sys

from aklr.core.contract.spec import CONTRACT_ID_PATTERN
from aklr.core.validation.stages.validation_scope import ValidationScope


def reference_contracts(base: str, head: str) -> set[str]:
    # Read immutable trees. A content-addressed kernel replacement is commonly an add + delete.
    diff = subprocess.check_output(
        ["git", "diff", "--raw", "--no-abbrev", "--no-renames", "-z", base, head, "--"]
    ).split(b"\0")
    blob_ids: set[bytes] = set()
    for header, path in zip(diff[:-1:2], diff[1::2], strict=True):
        path = os.fsdecode(path)
        old_mode, new_mode, old_oid, new_oid, status = header.removeprefix(b":").split()
        if status not in (b"A", b"M", b"D") or {old_mode, new_mode} - {b"000000", b"100644"}:
            return set()
        # Kernel PRs may also update a case note; these Markdown files are not test inputs.
        if re.fullmatch(r"harness/skills/kernel-optimization-book/cases/[a-zA-Z0-9_-]+\.md", path):
            continue
        if not path.startswith("aklr/plugins/aiter/") or ValidationScope.path_kind(path) != "schema_data":
            return set()
        if old_mode == b"100644":
            blob_ids.add(old_oid)
        if new_mode == b"100644":
            blob_ids.add(new_oid)

    contracts = set()
    for oid in blob_ids:
        schema = json.loads(subprocess.check_output(["git", "cat-file", "blob", oid]))
        if not isinstance(schema, dict) or not isinstance(schema.get("kernel"), dict):
            return set()
        contract = schema.get("contract_id")
        if (
            not schema["kernel"].get("sources")
            or not isinstance(contract, str)
            or not CONTRACT_ID_PATTERN.fullmatch(contract)
        ):
            return set()
        contracts.add(contract)
    return contracts


if __name__ == "__main__":
    try:
        contracts = reference_contracts(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "HEAD")
    except (subprocess.CalledProcessError, OSError, ValueError) as error:
        print(f"Could not determine reference scope ({error}); using all cases.", file=sys.stderr)
        contracts = set()
    print(" ".join(sorted(contracts)))
