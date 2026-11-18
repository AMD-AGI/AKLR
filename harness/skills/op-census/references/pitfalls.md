# What has broken a census before

Each item is something that gave a wrong or missing number on MI355X, with how to avoid it.

## Capture

- **A profiler that misses CUDA-graph kernels.** Decode replays graphs. Some torch/ROCm builds record the graph
  launch and none of its kernels (the vLLM K3 image, torch 2.13 + ROCm 7.2: 120 k kernels in 121 s, 77% NCCL), and
  in a multi-GPU server some ranks may record them and others not. `op_census probe` checks one process; the
  census's `steps_without_gpu_work` and the per-rank step counts check the real server. A point failing either is
  not a result.
- **rocprofv3 changes what it measures.** Its kernel trace makes `hipGraphLaunch` 30-100x slower on the host, so
  every traced decode looks launch-bound and the scheduler's batches change. Its pause does not stop a graph replay
  already queued (a 4 s window recorded 10.5 s). It has crashed a worker once in `release_signal`. On images that
  install ROCm from pip (`_rocm_sdk_*`) it does not work at all (two comgr copies: an LLVM abort, or a hang on exit
  without a trace). Use the server's torch profiler where the probe passes.
- **Labels by gaps.** Splitting a rank's trace into windows by idle gaps shifts the labels when one rank records an
  extra tail (it waited in a collective). Windows here are separate trace files; steps are labeled from the server's
  own spans.
- **Eager runs inflate communication.** Without graphs, a collective's kernel time includes the wait for the slowest
  rank's launch (K3 c4 eager: 86% of decode in all-reduce). Only graph runs, as the cookbook configures, are censuses.

## Serving the point as deployed

- **`apptainer --rocm`** binds the host's ROCm user-space libraries (libdrm, comgr, libnuma, OpenCL) over the
  image's, which is not the deployed stack and fails on images with an older glibc (`GLIBC_2.38 not found`). Run
  without it; `/dev/kfd` and `/dev/dri` are visible anyway. A `--writable` sandbox needs every bind destination to
  exist in it.
- **The host's environment leaks in.** An image's `LD_LIBRARY_PATH` default only applies when the variable is
  unset (`${LD_LIBRARY_PATH:-...}`), so a host value replaces it. Use `--cleanenv` and pass the recipe's variables.
- **Synthetic acceptance.** AgentX speculative runs do not verify drafts for real: InferenceX sets a golden
  acceptance length at launch (`SGLANG_SIMULATE_ACC_*`, vLLM `rejection_sample_method=synthetic`) and, for Kimi on
  vLLM, drops the tool and reasoning parsers. The plan applies the same overrides; a census without them measures a
  different decode.
- **Host pinned memory.** MI355X nodes may have less GTT than the cluster InferenceX ran on (1375 GiB here). A
  hierarchical-cache point that cannot allocate its host pool does not start; record any change made to start it.
- **Page cache and huge pages.** Reading 1 TB+ of weights fragments host memory; a vLLM offload pool allocated with
  transparent huge pages then stalls in compaction. Drop the weights' page cache after load before such a point.
- **A host smaller than the cluster's.** A container limited to some of the host's CPUs (96 of 236 on one MI355X host) breaks
  SGLang's per-rank CPU binding (`SGLANG_SET_CPU_AFFINITY=1`, set by every SGLang image): it computes each rank's
  CPUs from the host count, and a TP8 rank whose slice lies outside the allowed set exits at start. Turning the
  binding off is a deviation (host scheduling only), recorded in the point's report. A recipe that leaves vLLM's
  600 s engine-ready timeout may not start when model init is slower than on the cluster; raising the timeout is
  likewise a startup-only deviation.
- **Paths and arguments the cluster never exercised.** vLLM puts ZMQ sockets in `TMPDIR`; a long `TMPDIR` exceeds the
  107-byte AF_UNIX limit. A list-valued recipe argument (`cuda-graph-bs-prefill: [256, ...]`) is argparse `nargs`:
  one token per item, not JSON. Parse every point's command with the image's own parser before loading weights.
- **Cold weights.** A loader reading shards one by one from NFS can take 25+ minutes for a 117 GB checkpoint; read the
  files once in parallel first. Skip hidden directories (`.cache/huggingface`), which may not be readable.
- **First-start JIT.** Graph capture compiles per batch size: Qwen3.8's monokernel took ~7 minutes per size for an
  8-size capture. Give the ready timeout room; later starts reuse the cache in the image's home.
- **GPU memory not released.** vLLM's workers are `python -c from multiprocessing.spawn ...` processes that killing
  `vllm serve` does not reach. Before the next point, check that the GPUs hold no memory (`amd-smi`, `rocm-smi
  --showpids`). Kill by process group or PID, never by a broad `pkill`.
- **A queue started over a bare ssh PATH.** A driver started without `/opt/rocm/bin` on its PATH once lost its runner
  thread in the GPU-free check after a complete capture: no `status.json`, the GPUs never returned to the pool, the
  queue hung silently. The runner now looks for `amd-smi` in the ROCm locations too, writes a status whatever the
  teardown does, gives the GPUs back, and checks that a point's GPUs are free before starting it. A driver that
  waits with `pgrep -f "op_census run"` also waits for any shell whose command line merely contains that text.

## Reading the numbers

- **Running fewer requests than the concurrency.** SGLang DCP8 points ran 85-95% of max-running with the rest queued;
  the runner records the server's running and queued gauges at each window's edges.
- **One rank is not the step.** A rank that waits in collectives looks communication-heavy; `per_rank` keeps each
  rank's time with and without communication, and the critical path is the rank that waits least.
- **Prefill chunk sizes.** vLLM K3 prefill ran 7680-token chunks (with decode tokens riding along); an AgentX turn
  prefills about 4632 new tokens on a 1536-token cache block. Read sizes from the step labels, not from the recipe.
- **Kernel names.** A fused kernel (a MoE that includes its sort, a monokernel decode layer) cannot be split by name;
  its class is the class of its main work and its name stays in the table. AKLR's own kernels must be in the rules.
