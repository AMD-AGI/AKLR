from pathlib import Path

import pytest

from aklr.core.contract.registry import ContractRegistry


@pytest.fixture
def small_moe_schema():
    schema_path = "aklr/plugins/aiter/provider/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim512/690da7f963abb021.json"
    path = Path(schema_path)
    return ContractRegistry.default().parse_schema(path.read_bytes(), schema_path)
