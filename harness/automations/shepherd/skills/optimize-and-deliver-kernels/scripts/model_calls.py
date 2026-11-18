"""Which Points each model's serving calls, and how often: `harness/automations/shepherd/models/<model>.json`.

Run at the repository root.

`python -m aklr export --from <calls dir> --to <scratch dir> --language <any> | model_calls.py record --model
<model> --source <the run> --per <unit> --units <n>` writes the model's table from a serving run recorded with
`AKLR_EXPORT=1`: the calls of every Point the run reached, divided by the ``n`` units the run made (the forward
passes that reached it, say), so that each count is per one of them. A Point of the table it did not reach
keeps its value. Exit 2: error.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

TABLES = "harness/automations/shepherd/models"


@dataclass(frozen=True)
class ModelCalls:
    """Every model's table at one commit: a schema's name, its workloads' axes (`m=1,n=32`, as AKLR's
    `axes_text` writes them), and the calls in the forward pass that reaches each. A 1 is a seed, not a
    measurement."""

    by_model: dict  # model: schema name: axes: calls, per the unit the table's `per` names

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
                if not (isinstance(table["per"], str) and table["per"]):
                    raise ValueError("`per` does not name what one count is per")
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

    @staticmethod
    def recorded(export_output):
        """One model's calls in what `python -m aklr export` printed, by schema name and axes: its lines
        `served|added <op>/impls/<language>/<name>/<file>.json   <n> workloads  <axes>:<calls> ...`."""
        calls = {}
        for line in export_output.splitlines():
            status, *fields = line.split()
            if status not in ("served", "added"):
                continue
            points = calls.setdefault(PurePosixPath(fields[0]).parts[-2], {})
            for field in fields[3:]:
                axes, colon, n_calls = field.rpartition(":")
                if not (colon and n_calls.isdigit()):
                    break  # The note that ends some lines.
                points[axes] = points.get(axes, 0) + int(n_calls)
        return calls


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser(
        "record", help="write a model's table from `python -m aklr export` output on stdin"
    )
    record.add_argument("--model", required=True)
    record.add_argument("--source", required=True, help="the run: engine, layout, load, hardware, date")
    record.add_argument("--per", required=True, help="what one count is per, such as `forward pass`")
    record.add_argument("--units", required=True, type=int, help="how many of them the run made")
    args = parser.parse_args()
    try:
        measured = ModelCalls.recorded(sys.stdin.read())
        if not measured:
            raise ValueError("no `served` or `added` line on stdin")
        if args.units < 1:
            raise ValueError("--units is a positive number of --per")
        path = Path(TABLES, f"{args.model}.json")
        calls = {}
        if path.exists():
            table = json.loads(path.read_text(encoding="utf-8"))
            if table["per"] != args.per:
                raise ValueError(f"{path} counts per {table['per']!r}, not per {args.per!r}")
            calls = table["calls"]
        seeds = sum(axes not in measured.get(name, {}) for name, points in calls.items() for axes in points)
        for name, points in measured.items():
            per_unit = {
                axes: n // args.units if n % args.units == 0 else round(n / args.units, 3)
                for axes, n in points.items()
            }
            calls[name] = {**calls.get(name, {}), **per_unit}
        source = args.source + ("; the Points it did not reach keep their earlier value" if seeds else "")
        path.write_text(
            json.dumps({"model": args.model, "source": source, "per": args.per, "calls": calls}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps({"path": str(path), "measured": sum(map(len, measured.values())), "kept": seeds}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
