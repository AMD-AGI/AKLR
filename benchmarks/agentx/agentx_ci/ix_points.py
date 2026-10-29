# SPDX-License-Identifier: MIT
"""Runs inside an InferenceX checkout: cwd = inferencex-e2e, PYTHONPATH = it + its pinned deps.

Calls only InferenceX's own functions (matrix generation, point binding, variant expansion,
golden acceptance, fingerprints) and prints one JSON document on stdout. Imports nothing
from this repository, so it runs under InferenceX's dependency set alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    config_file, key = sys.argv[1], sys.argv[2]
    from infx.matrix.fingerprint import engine_environment
    from infx.matrix.generate import generate_config_matrix, with_fingerprints
    from infx.matrix.validation import config_root, load_config_files, load_runner_file
    from infx.srt_slurm.generate import bound_variant, point_environment
    from infx.srt_slurm.synthetic_acceptance import acceptance_length, build_overrides
    from infx.srt_slurm.variants import expand_variants

    files = [config_file]
    master = load_config_files(files)
    runners = load_runner_file("configs/runners.yaml")
    root = config_root(files)
    rows = generate_config_matrix([key], master, runners, eval_mode="none", root=root)
    rows = with_fingerprints(rows, root)
    points = []
    for row in rows:
        env = point_environment(row)
        variant, recipe = bound_variant(row, env, Path(root), expand=expand_variants)
        points.append(
            {
                "row": row,
                "environment": env,
                "variant": variant,
                "bound_recipe": recipe,
                "golden_overrides": build_overrides(recipe, row["framework"], env),
                "acceptance_length": acceptance_length(recipe, row["framework"], env),
                "engine_environment": json.loads(
                    engine_environment(row, Path(root)) or "{}"
                ),
            }
        )
    json.dump(
        {"config_key": key, "entry": master[key], "points": points},
        sys.stdout,
        default=str,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
