# RFC: e2e validates accuracy on servers launched from the sglang cookbook

Status: draft, not implemented.

## Summary

The e2e stage stops measuring performance and checks instead that a model still answers correctly with AKLR on:

1. Each run fetches the **latest** launch recipe (env and server flags) from the sglang cookbook and starts `sglang.launch_server` with it unchanged.
2. Each scenario starts one server with AKLR off and one with AKLR on:
   - Phase 1, GSM8K, an A/B test: both sides run it; the scenario fails when on scores more than a threshold below off.
   - Phase 2, one prompt with an expected output: only the on side runs it; the scenario fails when the answer differs from the expected output.

The model-level benchmarks under `aklr/benchmarks/models/` are abandoned, but deleting them is outside this RFC; this RFC only removes e2e's dependence on them (`SCENARIO_DIR`, `main.py`).

The code does not restrict the deployment shape: unified serving, prefill/decode (PD) disaggregation, multiple nodes, overrides of a recipe's flags and env, and a pinned or local cookbook are all expressed by configuration. CI has one machine today, so `scenarios.yaml` holds only single-node unified scenarios; that is a choice of configuration, not a limit of the code.

## Motivation

The current e2e stage runs `aklr/benchmarks/models/main.py` in-process with teacher-forced decode and compares tokens/s with AKLR on and off. Nothing checks that a route replacing an sglang function leaves the model's output correct; the bench stage checks each op against its reference, but not the route's glue (argument translation, the steps the route runs itself, the fallbacks). Serving the model the way users do and grading what it says covers that.

## The cookbook

`sgl-project/sgl-cookbook` is no longer maintained (last commit 2026-05, models up to sglang v0.5.10; no GLM-5.2, DeepSeek-V4 or Kimi-K3). The cookbook now lives in the sglang repository:

- pages: `docs/cookbook/autoregressive/<Vendor>/<Model>.mdx`;
- launch recipes: `docs/src/snippets/configs/<vendor>/<model>.jsx`, an `export const config = {...}` whose `cells: [...]` is an array of plain literals, one per recipe:

  ```js
  {
    match: { hw: "mi355x", pdMode: "unified", strategy: "balanced" },  // the keys differ per model
    nnodes: 1, verified: false, redirect: true?, warn: "...",
    env:   ["SGLANG_USE_AITER=1", ...],
    flags: ["--model-path {{MODEL_NAME}}", "--tp-size 8", ..., "--host {{HOST_IP}}", "--port {{PORT}}"],
  }
  ```

A cell's `match` keys differ per model (Kimi-K3: `hw/pdMode/strategy`; GLM-5.2 and DeepSeek-V4: `hw/variant/quant/strategy/nodes`), so a scenario gives the whole `match` and a cell is selected by equality.

**Parsing.** The CI image has no node, and the cells hold no functions, spreads or JSX (backticks appear only in comments and strings). Python cuts out the text from `cells: [` to its matching `]`, removes `//` comments with a string-aware scan, quotes bare keys, drops trailing commas and calls `json.loads`. A format change fails the parse, reported as `PLATFORM_FAILURE` with the cookbook's SHA, rather than serving a wrong recipe.

**Source.** By default the `main` branch of the sglang repository, the latest each run; `--e2e-cookbook-source` selects a branch, tag or SHA, or a local sglang checkout (for offline runs).

**Latest, yet traceable.** For a remote source, `git ls-remote https://github.com/sgl-project/sglang refs/heads/main` resolves the SHA first, and the files are fetched at that SHA from `raw.githubusercontent.com/sgl-project/sglang/<sha>/...` (not the REST API, so no 60-requests-per-hour limit). Both sides of a validation run use the same SHA; the SHA, the selected cells and the final command lines go into the report.

## Scenarios

`aklr/core/validation/stages/_e2e/scenarios.yaml` defines scenarios only: which cells of which cookbook, and how to start their servers. How the two phases run and their parameters (question count, threshold, prompt, expected output, timeout) are code constants, the same for every scenario. The top level keeps only `scenarios` (`defaults` is removed):

```yaml
scenarios:
  - cookbook: moonshotai/kimi-k3          # docs/src/snippets/configs/<this>.jsx
    model_path: moonshotai/Kimi-K3        # replaces {{MODEL_NAME}}
    env:                                   # added over the cell's env
      AKLR_SGLANG_PLUGIN: "1"
    args:                                  # overrides the cell's flags: replaced by name, null removes, true is a switch
      --mem-fraction-static: 0.8
    matches:                               # each item is a scenario of its own
      - {hw: mi355x, pdMode: unified, strategy: balanced}          # unified: one match
      - prefill: {hw: a3, pdMode: prefill, strategy: default}      # PD: one match per role
        decode:  {hw: a3, pdMode: decode, strategy: balanced}
  - cookbook: zai-org/glm-5.2
    matches:
      - {hw: mi355x, variant: default, quant: mxfp4, strategy: low-latency, nodes: single}
    ...
```

- An item of `matches` is one match (unified serving) or a `prefill` and a `decode` match (PD). `env` and `args` apply to every role of the scenario; written as `env: {prefill: {...}, decode: {...}}` they apply per role, and `args` likewise.
- An entry expands into one scenario per item of `matches`, sharing the entry's other fields; each starts its own servers, is judged on its own and has its own row in the report.
- An entry has no `name`. A scenario's name is `<model>-<the match values in YAML order, joined by ->`, where the model is the last segment of `cookbook` and a PD scenario joins the prefill values, then the decode values: for example `kimi-k3-mi355x-unified-balanced` or `glm-5.2-mi355x-default-mxfp4-low-latency-single`. The report and `--e2e-scenario NAME` use that name; `--e2e-scenario glm-5.2`, a model alone, selects every scenario of that model. Two scenarios with one name fail reading the file.
- A cell is selected by `match` equality alone; no match or more than one is an error. The cell's other fields (`redirect`, `warn`, `verified: false`) are shown in the progress and the report, never rejected.
- After the overrides and the known placeholders are applied, a remaining unknown `{{...}}` or `<...>` placeholder (such as `<network-interface>`) fails the scenario, naming it; the scenario's `env` or `args` must supply it.

## Deployment

One side (on or off) of a scenario is a **deployment**: one or more roles, each of `nnodes` node processes, plus a router for PD.

- Unified: one role; `nnodes` comes from the cell (default 1).
- PD: a `prefill` and a `decode` role, each started from its own cell, then
  `python -m sglang_router.launch_router --pd-disaggregation --prefill http://<p>:<port> <bootstrap> --decode http://<d>:<port> --host ... --port ...`.
  The bootstrap port is the prefill role's `--disaggregation-bootstrap-port` (default 8998). AKLR writes the router command itself: the cookbook's router template sits in `playgroundFeatures`, mixed with the page's logic, and is not parsed.
- Multiple nodes: following the cookbook's convention, the launcher adds `--nnodes N --node-rank i --dist-init-addr <node 0>:<port>`, and `{{HOST_IP}}` and `{{PORT}}` become the process's node and assigned port.
- The phases send requests to the deployment's entry point: node 0's server when unified, the router for PD.

Nodes are a resource of the stage, not a property of a scenario:

- `python -m aklr validate --e2e-hosts h0,h1,...`: the nodes available; by default the local machine only.
- `--e2e-launch 'ssh {host}'` (or `'srun -w {host} --overlap'`, ...): the command prefix that starts a process on a node other than the local one; by default none. Every process starts as `env K=V ... python -m ...`, so local and remote processes take the same path.
- A scenario needing more nodes (the sum of its roles' `nnodes`; the router shares node 0) than `--e2e-hosts` gives fails with `PLATFORM_FAILURE`, stating how many it needs and has; the other scenarios still run.
- Remote nodes must see this run's AKLR checkout and work directory (`AKLR_MEASURED_DIR`, the call export directory) at the same paths: for multiple nodes, `--e2e-work-dir` points at a shared filesystem.

## The two phases

Constants of `_stage.py`, shared by every scenario:

| phase | how | constants | verdict |
|---|---|---|---|
| 1 GSM8K (off and on) | `python -m sglang.test.run_eval --eval-name gsm8k --base-url <entry>`, reading the score | all 1319 questions; `max-tokens` 16384 (enough for thinking models; others stop far earlier); threshold 0.02 | `on < off - 0.02` → `E2E_ACCURACY_DROP` |
| 2 prompt (on only) | `/v1/chat/completions`, `temperature 0`, reading `message.content` | one prompt with a short, unique answer, e.g. "What is the capital of France? Answer with one word." expecting `Paris`; `max_tokens` as above | the answer, stripped of surrounding whitespace and trailing punctuation and compared case-insensitively, differs from the expected output → `E2E_WRONG_ANSWER` |

- The on side runs both phases on the same server, one after the other, without a restart; phase 2 runs even when phase 1 fails, and both results go into the report.
- The whole answer is compared, not a substring. A thinking model's reasoning goes to `reasoning_content` through the cell's own `--reasoning-parser`; a cell without one puts the reasoning into `content`, and the answer is judged different. That is a configuration fault, and it should show.
- The GSM8K threshold of 0.02 is provisional: with 1319 questions shared by both sides, the standard error of the difference is about 0.006–0.008, so 0.02 is about 2.5σ. It is to be set from measured off-against-off noise.

## Flow

For each scenario:

1. Fetch the cookbook, select each role's cell, apply the scenario's `env` and `args`, replace `{{MODEL_NAME}}` and the other placeholders, and assign nodes and ports.
2. Off first, then on, once each:
   - env = the stage's env (`HIP_VISIBLE_DEVICES`, `PYTHONPATH` set to the checkout root, `AKLR_MEASURED_DIR`, so on serves this run's bench measurements) | the cell's env | the scenario's env | `AKLR_DISABLE=1` on the off side;
   - start every process of the deployment, each in its own process group, and poll the entry point's `/health_generate` until it is ready;
   - phase 1; on the on side, phase 2;
   - stop the deployment by killing each process group, with the current `ScenarioRun.execute` logic (killpg, wait, read the log tail); a remote process is stopped by killing its local ssh or srun prefix process.

   A side's total time, from start to stop, is capped by a code constant (5400 s).
3. Verdicts:
   - any failure on the off side (the cookbook cannot be fetched or parsed, the server does not start, an evaluation errors) → `PLATFORM_FAILURE`, and on does not run: the cookbook does not fit the image, or the model or the environment is at fault, not the PR;
   - the cookbook names each recipe's image (`dockerImages`); the report shows it next to the CI image, and when its sglang version differs from the installed one, an off-side failure is a non-blocking warning instead;
   - the on side's server crashes or an evaluation errors → `E2E_CRASH`;
   - phases 1 and 2 as in the table above.

**Hit and fallback counts.** Today a separate export run counts how many calls AKLR served, for the performance report. Accuracy needs the count as much: if every call on the on side fell back, "no accuracy drop" means nothing. So the on side starts with `AKLR_EXPORT=1`, and after the phases the call files of every node are read; `n_hit_calls == 0` is flagged in the progress and listed in the report. No extra run.

## Structure

| file | concept | classes | public methods/functions |
|---|---|---|---|
| `_e2e/_cookbook.py` | a cookbook model configuration | `Cookbook` (`source`, `sha`, `cells`); `CookbookCell` (frozen: `match`, `nnodes`, `env`, `flags`, `verified`, `warn`) | `Cookbook.fetch(source: str, config: str) -> Cookbook` (classmethod: resolve the source, the SHA, download or read locally, parse `cells`); `Cookbook.cell(match: Mapping) -> CookbookCell` |
| `_e2e/_deployment.py` | one deployment of one side (on or off) | `Deployment` (roles, router, node assignment, `run_dir`); `Role` (frozen: `cell`, `env`, `args`); `NodeProcess` (one process on one node: its own process group, its log); `RunNotKilledError` | `Deployment.plan(scenario, cookbook, hosts, launch, env) -> Deployment` (classmethod: apply overrides, replace placeholders, add the multi-node flags, assign nodes and ports; raises `ValueError` when nodes are short or a placeholder remains); `Deployment.start(timeout) -> str` (the entry URL); `Deployment.stop() -> str` (log tails); `Deployment.argvs` (every command line, for the report) |
| `_e2e/_config.py` (changed) | the e2e configuration file | `E2eConfig`; `Scenario` (one expanded scenario: `name`, `cookbook`, `roles` (role name to match), `model_path`, `env`, `args`) | `E2eConfig.read` (expands `matches`); `E2eConfig.selected` |
| `_e2e/_report.py` (changed) | the e2e report | `E2eReport`; `E2eRecord` (scenario name, cookbook SHA, the off and on `SideResult`, the difference); `SideResult` (`gsm8k_score`, `prompt_answer` (empty on off), hit and fallback counts, command lines) | `E2eRecord.table_row` (a column for whether phase 2 matched); `E2eRecord.console_line`; `E2eReport.record_blocks` (the table, and the on side's answer next to the expected output) |
| `_e2e/_stage.py` (changed) | the e2e stage | `E2eStage` (new constructor arguments `cookbook_source`, `hosts`, `launch`, `work_dir`); the phases' constants | `E2eStage.run` |
| `command.py` (changed) | the `python -m aklr validate` command | — | `main` gains `--e2e-cookbook-source`, `--e2e-hosts`, `--e2e-launch`, `--e2e-work-dir` |
| `_report/_finding.py` (changed) | why a stage did not pass | `FindingCode`: `E2E_NOT_FASTER` removed; `E2E_ACCURACY_DROP` and `E2E_WRONG_ANSWER` added | — |

A side's flow (start the deployment, phase 1, phase 2 on the on side, read the call files, stop) is a private method of `E2eStage` returning a `SideResult`.

## Removed

- `_e2e/_run.py` (`ScenarioRun`, `RunResult`); from `_config.py`, `SCENARIO_DIR`, `Scenario.argv` and `E2eSettings` (the whole class, with the YAML's `defaults`);
- from `_e2e/_stage.py`, `_compare_rounds` and the ratio computation (`ratio_lower_bound` itself stays: bench uses it);
- the e2e performance cases of `tests/core/validation/test_pipeline.py` (`E2E_NOT_FASTER`, `throughput_ratio_lower_bound`, `latency_ratio_lower_bound`, `ScenarioRun`).

Text updated with it:

- `.github/workflows/ci.yaml`: the validation job's comment on "the e2e texts, at the revisions aklr/benchmarks/models/main.py pins";
- `aklr/plugins/README.md`, "Correctness: validation", item 3: e2e compares accuracy.

## Tests

- `tests/core/validation/test_e2e_cookbook.py`: a fixture cut from the real `kimi-k3.jsx` (comments, trailing commas, `//` and backticks inside strings); asserts the parsed mi355x cell's env and flags; no match and several matches fail; a local-directory source. No network.
- The pure part of `Deployment.plan`: unified on one node; the multi-node flags `--nnodes/--node-rank/--dist-init-addr`; PD's router command and bootstrap port; `args` replacing, removing and switching; per-role `env`; a remaining placeholder fails; too few nodes fail. Starts no process.
- `E2eConfig.read`: valid and invalid files in the new format (replacing the current cases).
- `E2eStage`'s verdicts, with `Deployment.start` and the phases stubbed: off fails, on crashes, accuracy drops, wrong answer, pass. The current process-group cleanup cases move to `NodeProcess`.

## CI

The `ci:e2e` label goes away: whenever the validation job runs, e2e runs, with `--stages audit,bench,e2e` fixed. The validation job's own trigger is unchanged (a push to main, or a same-repository PR with `ci:gpu` and schemas to measure). A PR that changes code, routes included, measures every schema, so a route change always gets e2e; a PR changing only documents or only new schema data still runs no GPU job.

Changes to `.github/workflows/ci.yaml`:

- remove `workflow_dispatch.inputs.e2e`;
- remove code-lint's "Read the ci:e2e label" step and its `should_run_e2e` output;
- fix the validation job's stages at `audit,bench,e2e` (inline in the command, dropping the `STAGES` env). The step name becomes the fixed `Validate (audit,bench,e2e)`; review-tier's `ValidationGate.e2e_step` decides from that name whether e2e ran, and must accept it;
- re-estimate `timeout-minutes: 300` for e2e's time (two weight loads and GSM8K per scenario) and the number of scenarios.

Text mentioning `ci:e2e` is updated in `aklr/plugins/README.md`, `.github/pull_request_template.md`, `harness/skills/aklr-open-pr/SKILL.md` and `harness/skills/aklr-read-validation/references/ci.md`.

To confirm:

- the GPU container reaches `github.com` and `raw.githubusercontent.com` (the cookbook);
- GSM8K data: `run_eval` downloads `test.jsonl` from GitHub by default; if the container cannot, the file goes into the shared HF cache directory and phase 1 passes `--gsm8k-data-path` (a code constant too). The shared cache's e2e texts (`datasets--org--name`), kept for `main.py`, need no further upkeep.

## Risks and open questions

1. **The latest cookbook against a pinned image.** Today the cookbook names v0.5.19 for Kimi-K3 and v0.5.13 for GLM-5.2 FP8 on MI355X, against CI's v0.5.22, so their off-side failures only warn; GLM-5.2 MXFP4 and DeepSeek-V4 name v0.5.22 and block.
2. **Which cells to test.** The scenarios above were picked after the former performance scenarios (low-concurrency decode) and need confirming. DeepSeek-V4's low-latency cell runs `--speculative-algorithm DSPARK`, whose call shapes may lie outside the workloads AKLR has measured, showing as mostly fallback calls.
3. **Run time.** Each scenario loads its weights twice (about 1.5 TB for Kimi-K3) and runs 1319 GSM8K questions per side, longer for thinking models. Without `ci:e2e`, every code PR with `ci:gpu` pays that time, and GPU runner occupancy grows markedly. If too slow, phase 1 runs fewer questions (a code constant), at the cost of a wider threshold.
4. **PD and multiple nodes are untested in CI.** They have pure-function tests only; their first real use will likely need fixes. Stopping a remote process by killing a local ssh may not reach the remote side; srun, or an explicit remote cleanup, may be needed.
