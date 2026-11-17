"""Which Points each model's serving calls, and how often: `harness/automations/shepherd/models/<model>.json`."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath

TABLES = "harness/automations/shepherd/models"


@dataclass(frozen=True)
class ModelCalls:
    """Every model's table at one commit: a schema's name, its workloads' axes (`m=1,n=32`, as AKLR's
    `axes_text` writes them), and the calls in the forward pass that reaches each. A 1 is a seed, not a
    measurement."""

    by_model: dict  # model: schema name: axes: calls

    @classmethod
    def load(cls, commit):
        """The tables at ``commit``, none when it has none; ``ValueError`` naming an unusable one."""
        tree = subprocess.run(
            ["git", "ls-tree", "--name-only", commit, f"{TABLES}/"], capture_output=True, check=True
        )
        by_model = {}
        for path in tree.stdout.decode().splitlines():
            model = PurePosixPath(path).stem
            blob = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True)
            try:
                table = json.loads(blob.stdout)
                if table["model"] != model:
                    raise ValueError(f"model {table['model']!r} is not the file's name")
                by_model[model] = table["calls"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}: no usable table: {exc!r}") from None
        return cls(by_model)

    def callers(self, name):
        """The models whose table holds a Point of the schema ``name``, sorted."""
        return sorted(model for model, calls in self.by_model.items() if name in calls)

    def calls(self, models, name, axes):
        """The calls of the schema ``name`` at a workload's ``axes`` over ``models``; 0: none calls it."""
        key = ",".join(f"{axis}={value}" for axis, value in axes.items())
        return sum(self.by_model[model].get(name, {}).get(key, 0) for model in models)
