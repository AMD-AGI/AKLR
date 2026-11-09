# Forge Optimization Report

- Campaign: `db97d5b1dcc444bcab41222e17933a1b`
- Status: best verified result
- mean case speedup: 1.067351x
- Improved overall: yes
- Baseline raw mean: 9.0115 ms
- Search-start raw mean: 501.4019 ms
- Selected candidate raw mean (diagnostic; not monotonic, but it withdraws the improvement above when it contradicts the score): 8.4092 ms
- Correctness: PASS
- Best iteration: 5
- Commit: `5bfbea64c0d72d191756b25f14d171e78e120703`
- Optimization: Stream stage-2 W2 weights non-temporally at small token counts.

## Changed Files

- `kernel.py`

## Round Budget

- Rounds planned (campaign total): 4
- Planning wall-clock (campaign total): 18.0 min
- Round wall-clock (campaign total): 171.0 min
- Campaign wall-clock: 210.4 min
- Planning share of campaign wall-clock: 9%

## Artifacts

- Patch: `best/iter_005/forge.patch`
- Validation: `best/iter_005/validation.txt`
- Benchmark: `best/iter_005/benchmark.json`
- Bundle: `best/iter_005`
