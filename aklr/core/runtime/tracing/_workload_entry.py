from dataclasses import dataclass
from typing import Any


@dataclass
class WorkloadEntry:
    schema_name: str

    axes: dict[str, int]

    inputs_to_dump: dict[str, Any]

    order: int

    cuda_graph_snapshot: dict[str, Any] | None = None
