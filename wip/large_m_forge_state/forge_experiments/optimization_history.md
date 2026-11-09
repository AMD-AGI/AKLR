# Forge Optimization History

## Iteration 1 — REVERT_INTEGRITY

- Session: 3
- Experiment: `0c8f5098`
- Session end: `sdk_error`
- Turns: 
- Plan: unspecified
- Canonical correctness: FAIL/NOT RUN
- Candidate mean case speedup: 
- Best mean case speedup before: 
- Best mean case speedup after: 0.014490173973418355
- Candidate raw mean ms: 
- Commit: ``
- Archive: ``

Changed files:
- (none)

## Iteration 2 — KEEP

- Session: 3
- Experiment: `0c8f5098`
- Session end: `candidate_submitted`
- Turns: 40
- Plan: Inline aiter's FlyDSL fp4 MoE GEMMs, with torch routing sort and MXFP4 quantization
- Canonical correctness: PASS
- Candidate mean case speedup: 0.17940596985112736
- Best mean case speedup before: 0.014490173973418355
- Best mean case speedup after: 0.17940596985112736
- Candidate raw mean ms: 67.3979
- Commit: `6469848867cdb47e56695512ddf38d2872531b73`
- Archive: `candidates/iter_002/`

Changed files:
- `kernel.py`

## Iteration 3 — KEEP

- Session: 4
- Experiment: `2abb24e1`
- Session end: `candidate_submitted`
- Turns: 51
- Plan: Replace torch glue with FlyDSL sort and quant kernels, fp4 stage-1 output
- Canonical correctness: PASS
- Candidate mean case speedup: 0.9879742842630966
- Best mean case speedup before: 0.17940596985112736
- Best mean case speedup after: 0.9879742842630966
- Candidate raw mean ms: 10.5277
- Commit: `ccff3ae8b7a338b607c501efa3fb80e49ed9d87d`
- Archive: `candidates/iter_003/`

Changed files:
- `kernel.py`

## Iteration 4 — KEEP

- Session: 4
- Experiment: `2abb24e1`
- Session end: `candidate_submitted`
- Turns: 50
- Plan: Large-n stage-1/stage-2 tile and B-caching settings plus a multi-workgroup routing sort
- Canonical correctness: PASS
- Candidate mean case speedup: 1.0477255592584294
- Best mean case speedup before: 0.9879742842630966
- Best mean case speedup after: 1.0477255592584294
- Candidate raw mean ms: 8.4918
- Commit: `f25779ae8005d18e85bdaee42a85066a101950d2`
- Archive: `candidates/iter_004/`

Changed files:
- `kernel.py`

## Iteration 5 — KEEP

- Session: 4
- Experiment: `2abb24e1`
- Session end: `candidate_submitted`
- Turns: 18
- Plan: Stream stage-2 W2 weights non-temporally at small token counts.
- Canonical correctness: PASS
- Candidate mean case speedup: 1.0673514534925792
- Best mean case speedup before: 1.0477255592584294
- Best mean case speedup after: 1.0673514534925792
- Candidate raw mean ms: 8.4092
- Commit: `5bfbea64c0d72d191756b25f14d171e78e120703`
- Archive: `candidates/iter_005/`

Changed files:
- `kernel.py`
