# SPDX-License-Identifier: MIT
"""Bind points using the pinned InferenceX checkout."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    config_file, key = sys.argv[1], sys.argv[2]
    from infx.clusters import load_inventory
    from infx.clusters.slurm import slurm_settings
    from infx.launch.drivers.srt.config import HEALTH_CHECK
    from infx.launch.drivers.srt.lanes import srt_time_limit
    from infx.launch.request import LaunchRequest
    from infx.matrix.generate import generate_config_matrix, with_fingerprints
    from infx.matrix.validation import config_root, load_config_files, load_runner_file
    from infx.srt_slurm.generate import bound_variant, point_environment
    from infx.srt_slurm.synthetic_acceptance import acceptance_length, build_overrides
    from infx.srt_slurm.variants import expand_variants

    files = [config_file]
    master = load_config_files(files)
    runners = load_runner_file("configs/runners.yaml")
    inventory = load_inventory("configs/runners.yaml")
    root = config_root(files)
    rows = generate_config_matrix([key], master, runners, eval_mode="none", root=root)
    rows = with_fingerprints(rows, root)
    points = []
    for row in rows:
        env = point_environment(row)
        variant, recipe = bound_variant(row, env, Path(root), expand=expand_variants)
        clusters = {inventory.cluster_for(name).id for name in inventory.labels[row["runner"]]}
        if len(clusters) != 1:
            raise ValueError(f"runner label {row['runner']} does not identify one cluster")
        cluster = inventory.clusters[clusters.pop()]
        request = LaunchRequest(
            runner_name=row["runner"],
            model=row["model"],
            model_prefix=row["model-prefix"],
            framework=row["framework"],
            is_agentic=True,
            conc=row["conc"],
        )
        time_limit = srt_time_limit(cluster.id, request, None, slurm_settings(cluster).srt_slurm)
        runtime = {
            "time_limit": time_limit,
            "health_check": {**HEALTH_CHECK, **(recipe.get("health_check") or {})},
            "port": (recipe["roles"]["agg"].get("args") or {}).get("port"),
            "sources": {
                "time_limit": f"infx.launch.drivers.srt.lanes.srt_time_limit / configs/runners.yaml / {cluster.id}",
                "health_check": "infx.launch.drivers.srt.config.HEALTH_CHECK + bound recipe health_check",
                "dram": "infx.matrix.generate.agentic_dram_offload_gb / point_environment.TOTAL_CPU_DRAM_GB",
                "port": "bound recipe roles.agg.args.port; otherwise allocated on the execution host",
            },
        }
        points.append(
            {
                "row": row,
                "environment": env,
                "variant": variant,
                "bound_recipe": recipe,
                "golden_overrides": build_overrides(recipe, row["framework"], env),
                "acceptance_length": acceptance_length(recipe, row["framework"], env),
                "runtime": runtime,
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
