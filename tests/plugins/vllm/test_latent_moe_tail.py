# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's latent-MoE tail on vLLM to AKLR's kimi_k3_latent_moe_tail, on the CPU: the reference against vLLM's ops
composed one by one; the committed kernel's layout run for eight ranks (every hand-off written once and read where it
was written, the MFMA lanes, the result against the reference and the same on every rank); what the route takes,
declines and checks, and that its decisions read nothing a rank has alone; the committed schema."""

import ast
import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec, FastKeyFnGenerator
from aklr.core.contract.locate import committed_schema_paths, op_dir, schema_relative_path, schema_uuid
from aklr.core.contract.op import OpContract
from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import vllm as plugin
from aklr.plugins.vllm.models.kimi_k3.latent_moe import api, runtime
from aklr.plugins.vllm.models.kimi_k3.latent_moe.contract import KIMI_K3_LATENT_MOE_TAIL
from aklr.plugins.vllm.models.kimi_k3.latent_moe.reference import kimi_k3_latent_moe_tail_reference

TP, HIDDEN, LATENT = 8, 7168, 3584
BF16 = torch.bfloat16


def bf16(x):
    return x.to(BF16).float()


def draw(n, hidden=HIDDEN, latent=LATENT, tp=TP, seed=0):
    """Every rank's args as the contract draws them: N(0, 1) partials and norm weights, LeCun up-projections."""
    gen = torch.Generator().manual_seed(seed)
    shard = hidden // tp
    return [
        dict(
            routed=torch.randn(n, latent, generator=gen).to(BF16),
            shared=torch.randn(n, hidden, generator=gen).to(BF16),
            up_weight=(torch.randn(shard, latent, generator=gen) / latent**0.5).to(BF16),
            norm_weight=torch.randn(latent, generator=gen).to(BF16),
            eps=1e-5,
        )
        for _ in range(tp)
    ]


def point(n, hidden=HIDDEN, latent=LATENT, tp=TP):
    return SimpleNamespace(
        axes={"n_tokens": n, "hidden_size": hidden, "latent_size": latent},
        attrs={"tp_size": tp},
        dtypes=dict.fromkeys(("routed", "shared", "up_weight", "norm_weight", "out"), "bfloat16"),
    )


def reference(group_args, rank=0):
    n, latent = group_args[0]["routed"].shape
    hidden = group_args[0]["shared"].shape[1]
    return kimi_k3_latent_moe_tail_reference(point(n, hidden, latent, len(group_args)), group_args, rank)


# ---------------------------------------------------------------- the reference


def test_the_reference_is_vllms_ops_one_by_one():
    """vLLM's all-reduce (a float32 sum, bf16 out), aiter's RMSNorm (bf16(x * rstd * w)), the addmm into a rank's own
    columns (float32 beta-add, bf16 out), the all-reduce of the shared outputs; each rank with its own norm weight."""
    group = draw(5, hidden=256, latent=128)
    latent = bf16(sum(a["routed"].float() for a in group))
    rstd = torch.rsqrt(latent.square().mean(-1, keepdim=True) + 1e-5)
    outputs = []
    for r, a in enumerate(group):
        s = a["shared"].float().clone()
        normed = bf16(latent * rstd * a["norm_weight"].float())
        s[:, r * 32 : (r + 1) * 32] = bf16(s[:, r * 32 : (r + 1) * 32] + normed @ a["up_weight"].float().T)
        outputs.append(s)
    want = sum(outputs)
    for rank in (0, 5):
        assert torch.equal(reference(group, rank), want)


def test_a_rank_outside_the_sum_changes_the_out():
    """Every rank's routed and shared partials count: no rank's slice is left out."""
    group = draw(3, hidden=256, latent=128)
    base = reference(group)
    for r in range(TP):
        for name in ("routed", "shared"):
            changed = [dict(a) for a in group]
            changed[r][name] = torch.zeros_like(group[r][name])
            assert not torch.equal(reference(changed), base), (r, name)


# ---------------------------------------------------------------- the committed kernel's layout, for eight ranks


def committed_schema():
    contract = ContractRegistry.default().get("kimi_k3_latent_moe_tail@1.0")
    (path,) = committed_schema_paths(op_dir(contract.op_package))
    return path


def kernel_source(name):
    sources = json.loads(committed_schema().read_text())["kernel"]["sources"]
    return next(s["content"] for s in sources if s["path"] == name)


@pytest.fixture(scope="module")
def layout():
    module = types.ModuleType("kimi_k3_latent_moe_tail_layout")
    exec(compile(kernel_source("layout.py"), "layout.py", "exec"), module.__dict__)
    return module


class _Memory:
    """A rank's bytes start .. end as the kernel's hand-offs use them: units of ``unit`` bytes (16: four bf16 in
    tagged pairs; 8: four plain bf16, or one tagged f32 or flag), each written at most once a launch inside its
    region's bytes and read only once written. Under the flag protocol a payload unit keeps the byte offset of the flag
    its writer stores after it, and its reader names the flag it polled first: they must be the same."""

    def __init__(self, start, end, unit, width):
        self.start, self.unit = start, unit
        self.values = torch.zeros((end - start) // unit, width)
        self.writes = torch.zeros((end - start) // unit, dtype=torch.int64)
        self.flags = torch.full(((end - start) // unit,), -1, dtype=torch.int64)

    def _index(self, offsets):
        assert (offsets % self.unit == 0).all() and (offsets >= self.start).all()
        return (offsets - self.start) // self.unit

    def write(self, offsets, values, region, flag=None):
        start, end = region
        flag = None if flag is None else torch.broadcast_to(torch.as_tensor(flag), offsets.shape).reshape(-1)
        offsets = offsets.reshape(-1)
        assert (offsets >= start).all() and (offsets + self.unit <= end).all()
        index = self._index(offsets)
        assert (self.writes[index] == 0).all() and len(set(index.tolist())) == len(index)
        self.values[index] = values.reshape(len(index), -1)
        self.writes[index] = 1
        if flag is not None:
            self.flags[index] = flag

    def read(self, offsets, flag=None):
        flag = None if flag is None else torch.broadcast_to(torch.as_tensor(flag), offsets.shape).reshape(-1)
        index = self._index(offsets.reshape(-1))
        assert (self.writes[index] == 1).all()
        if flag is not None:
            assert torch.equal(self.flags[index], flag)
        return self.values[index]


def regions(L):
    """Each mailbox region's bytes, (start, end)."""
    return {
        "lred": (L.LRED, L.SRED),
        "sred": (L.SRED, L.LAG),
        "lag": (L.LAG, L.LSQ),
        "lsq": (L.LSQ, L.YAG),
        "yag": (L.YAG, L.LRED_D),
        "lred_d": (L.LRED_D, L.SRED_D),
        "sred_d": (L.SRED_D, L.LAG_D),
        "lag_d": (L.LAG_D, L.YAG_D),
        "yag_d": (L.YAG_D, L.LRED_F),
        "lred_f": (L.LRED_F, L.SRED_F),
        "sred_f": (L.SRED_F, L.YAG_F),
        "yag_f": (L.YAG_F, L.PEER_END),
        "nbuf": (L.NBUF, L.SCRATCH_END),
    }


def _poll_flags(memory, first):
    """A row's 8 sources' flags from byte ``first``, two a 16-byte poll, as the kernel loads them."""
    for i in range(TP // 2):
        for second in (0, 8):
            memory.read(first + 16 * i + second)


def run_layout(L, group):
    """The committed kernel's launch on every rank, its arithmetic in float32 (the MFMAs' sums in another order):
    every hand-off through the layout's offsets, in the protocol the row count picks (tagged pairs, or from
    L.FLAG_ROWS rows data then flag). Returns each rank's out."""
    n = group[0]["routed"].shape[0]
    flagged = n >= L.FLAG_ROWS
    a = torch.arange
    region = regions(L)
    # the tagged regions' quads (not LSQ's bytes)
    quads = [_Memory(L.LRED, L.LRED_D, 16, 4) for _ in range(TP)]
    pairs = [_Memory(L.LSQ, L.YAG, 8, 1) for _ in range(TP)]
    plain = [_Memory(L.LRED_D, L.LRED_F, 8, 4) for _ in range(TP)]
    flags = [_Memory(L.LRED_F, L.PEER_END, 8, 1) for _ in range(TP)]
    nbuf = [_Memory(L.NBUF, L.SCRATCH_END, 16, 4) for _ in range(TP)]
    rows = a(n)[:, None]
    # push: the pushing workgroups' jobs (set 0 of the up projection pushes nothing), each job once
    jobs = L.push_job(a(L.POOL)[:, None, None], a(L.WAVES)[None, :, None], a(2)[None, None, :], n).reshape(-1)
    assert len(set(jobs.tolist())) == len(jobs) and set(range(L.push_jobs(n)[1])) <= set(jobs.tolist())
    n_latent, n_jobs = L.push_jobs(n)
    lanes = a(L.SLICE)[None, :]
    for r, args in enumerate(group):
        for name, kind, sel in (
            ("routed", L.latent_job, jobs < n_latent),
            ("shared", L.shared_job, (jobs >= n_latent) & (jobs < n_jobs)),
        ):
            job = jobs[sel][:, None] - (0 if name == "routed" else n_latent)
            element, owner, t, col = (torch.broadcast_to(x, (len(job), L.SLICE)) for x in kind(job, lanes))
            data = args[name].float().reshape(-1)[element[..., None] + a(8)]
            latent = name == "routed"
            # the job's flag, as its lane 0 names it
            flag = L.lred_f(t[:, :1], r) if latent else L.sred_f(t[:, :1], col[:, :1] // L.LS, r)
            flag = torch.broadcast_to(flag, t.shape)
            for p in range(TP):
                own = owner == p
                if flagged:
                    offset = (L.lred_d if latent else L.sred_d)(r, t[own], col[own])
                    target = region["lred_d" if latent else "sred_d"]
                    plain[p].write(offset, data[own][:, :4], target, flag[own])
                    plain[p].write(offset + 8, data[own][:, 4:], target, flag[own])
                    lead = own[:, 0]
                    flag_region = region["lred_f" if latent else "sred_f"]
                    flags[p].write(flag[lead, 0], torch.ones(int(lead.sum()), 1), flag_region)
                else:
                    offset = (L.lred if latent else L.sred)(r, t[own], col[own])
                    target = region["lred" if latent else "sred"]
                    quads[p].write(offset, data[own][:, :4], target)
                    quads[p].write(offset + 16, data[own][:, 4:], target)
    # lsum: owner r's columns of every row
    col = 4 * a(L.LSUM_LANES)[None, :]
    for r in range(TP):
        if flagged:
            _poll_flags(flags[r], L.lred_f(rows, 0))
            vals = [plain[r].read(L.lred_d(src, rows, col), L.lred_f(rows, src)) for src in range(TP)]
        else:
            vals = [quads[r].read(L.lred(src, rows, col)) for src in range(TP)]
        acc = vals[0]
        for v in vals[1:]:
            acc = acc + v
        lat = bf16(acc)
        for p in range(TP):
            if flagged:  # the row's LSQ pair, stored after every thread's payload, is its flag
                plain[p].write(L.lag_d(rows, L.LS * r + col), lat, region["lag_d"], L.lsq(rows, r))
            else:
                quads[p].write(L.lag(rows, L.LS * r + col), lat, region["lag"])
            pairs[p].write(L.lsq(a(n), r), lat.view(n, -1).square().sum(-1), region["lsq"])
    # gather: every rank's normalized rows
    j = a(L.QUADS)[None, :]
    for r, args in enumerate(group):
        # two owners' pairs a 16-byte poll, as the kernel loads them
        total = None
        for i in range(TP // 2):
            for second in (0, 8):
                x = pairs[r].read(L.lsq(a(n), 2 * i) + second)[:, 0]
                total = x if total is None else total + x
        rstd = torch.rsqrt(total / L.LATENT + args["eps"])
        if flagged:
            lat = plain[r].read(L.lag_d(rows, 4 * j), L.lsq(rows, 4 * j // L.LS))
        else:
            lat = quads[r].read(L.lag(rows, 4 * j))
        lat = lat.view(n, L.QUADS, 4)
        normed = bf16(lat * rstd[:, None, None] * args["norm_weight"].float().view(L.QUADS, 4))
        # tagged quads, no flag after them: the up lanes poll these very units
        nbuf[r].write(L.nbuf(rows, 4 * j), normed, region["nbuf"])
    # up: set s's workgroup j, 16 columns of the set's token groups, its waves' MFMAs as the lanes hold them, the
    # epilogue's items of the set's rows
    groups, sets = (n + L.TILE - 1) // L.TILE, L.up_sets(n)
    owned = [s + sets * a(L.set_groups(s, n)) for s in range(sets)]
    assert torch.equal(torch.cat(owned).sort().values, a(groups))
    ups, waves, steps, lane = a(L.UP_CTAS), a(L.WAVES), a(L.K_STEPS), a(64)
    t_item, q_item = L.up_item(a(L.THREADS))
    live = t_item < n
    t_item, q_item = t_item[live], q_item[live]
    item_lane, item_group = L.item_lane(t_item, q_item)
    item_set = item_group % sets  # the set whose workgroups hold the item's accumulators
    outs = []
    for r, args in enumerate(group):
        w_index = L.weight_element(
            ups[:, None, None, None], waves[None, :, None, None], lane, steps[None, None, :, None]
        )
        w = args["up_weight"].float().reshape(-1)[w_index[..., None] + a(8)]  # [j, wave, step, lane, 8]
        a_tile = (
            w.view(L.UP_CTAS, L.WAVES, L.K_STEPS, 4, 16, 8)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(*w.shape[:3], 16, 32)
        )
        lds = torch.full((sets, L.UP_CTAS, L.LDS_RED), float("nan"))  # each set's workgroups' own LDS
        for s, gs in enumerate(owned):
            row = torch.clamp(L.TILE * gs[:, None] + lane % L.TILE, max=n - 1)
            b_index = L.latent_element(
                row[None, :, None, :], waves[:, None, None, None], lane, steps[None, None, :, None]
            )
            # a lane's operand of a step: two tagged quads from its first byte, its pilot the first step's first pair
            first = L.nbuf(0, b_index)
            nbuf[r].read(first[:, :, 0].reshape(-1))
            b = torch.cat([nbuf[r].read(first.reshape(-1)), nbuf[r].read(first.reshape(-1) + 16)], dim=-1)
            b = b.view(*b_index.shape, 8)  # [wave, group, step, lane, 8]
            b_tile = (
                b.view(L.WAVES, len(gs), L.K_STEPS, 4, 16, 8)
                .permute(0, 1, 2, 3, 5, 4)
                .reshape(*b.shape[:3], 32, 16)
            )
            d = torch.einsum("jwsik,wgskc->jwgic", a_tile, b_tile)  # [j, wave, group, 16 rows, 16 tokens]
            # lane l holds rows 4 (l / 16) .. + 3 of token l % 16
            acc = (
                d.view(L.UP_CTAS, L.WAVES, len(gs), 4, 4, 16)
                .permute(0, 1, 2, 3, 5, 4)
                .reshape(L.UP_CTAS, L.WAVES, len(gs), 64, 4)
            )
            lds_index = L.accumulator(waves[:, None, None], gs[None, :, None], lane)
            lds[s][:, lds_index[..., None] + a(4)] = acc
        y = sum(
            lds[
                item_set[None, :, None],
                ups[:, None, None],
                (L.accumulator(w_, item_group, item_lane)[:, None] + a(4)),
            ]
            for w_ in range(L.WAVES)
        )
        assert not torch.isnan(y).any()  # every item's accumulators from its own workgroup's MFMAs
        local = L.up_column(ups[:, None], q_item[None, :])
        if flagged:  # the sources' flags of the half of the item's row holding its columns
            _poll_flags(flags[r], L.sred_f(t_item[None, :], local // L.LS, 0))
        shared = None
        for src in range(TP):
            if flagged:
                v = plain[r].read(
                    L.sred_d(src, t_item[None, :], local), L.sred_f(t_item[None, :], local // L.LS, src)
                )
            else:
                v = quads[r].read(L.sred(src, t_item[None, :], local))
            v = v.view(L.UP_CTAS, -1, 4)
            shared = v if shared is None else shared + v
        for p in range(TP):
            if flagged:  # flagged by the item's workgroup, a flag a token group of its set
                yag_flag = L.yag_f(r, item_group[None, :], ups[:, None])
                plain[p].write(
                    L.yag_d(t_item[None, :], L.HS * r + local), bf16(shared + y), region["yag_d"], yag_flag
                )
            else:
                quads[p].write(L.yag(t_item[None, :], L.HS * r + local), bf16(shared + y), region["yag"])
        if flagged:  # each set's workgroup j, once its payload is out: thread i the flag of the set's group i
            for gs in owned:
                for p in range(TP):
                    yag_flags = L.yag_f(r, gs[:, None], ups[None, :])
                    flags[p].write(yag_flags, torch.ones(yag_flags.numel(), 1), region["yag_f"])
    # out: every workgroup's quads of YAG
    for r in range(TP):
        total, per, i0 = L.out_items(a(L.GRID)[:, None], a(L.THREADS)[None, :], n)
        has_i0 = (a(L.THREADS)[None, :] < per) & (i0 < total)
        has_i1 = (a(L.THREADS)[None, :] + L.THREADS < per) & (i0 + L.THREADS < total)
        items = torch.cat([i0[has_i0], (i0 + L.THREADS)[has_i1]])
        assert torch.equal(items.sort().values, a(n * L.OUT_QUADS))
        out = torch.empty(n * L.OUT_QUADS, 4)
        if flagged:
            flags[r].read(L.out_flag(items))
            out[items] = plain[r].read(L.yag_d(0, 4 * items), L.out_flag(items))
        else:
            out[items] = quads[r].read(L.yag(0, 4 * items))
        outs.append(out.view(n, L.HIDDEN))
    for r in range(TP):
        # one protocol a launch: the other's regions untouched
        assert (quads[r].writes.sum() == 0) == flagged and (plain[r].writes.sum() > 0) == flagged
        assert (flags[r].writes.sum() > 0) == flagged
        # every source's every quad of every row, once
        if flagged:
            live = plain[r].writes[: (L.SRED_D - L.LRED_D) // 8].view(TP, L.MAX_ROWS, -1)[:, :n]
        else:
            live = quads[r].writes[: (L.SRED - L.LRED) // 16].view(TP, L.MAX_ROWS, -1)[:, :n]
        assert (live == 1).all()
    return outs


@pytest.mark.parametrize("n", [1, 10, 17, 23, 24, 40, 72, 128])
def test_the_kernels_layout_computes_the_op_on_every_rank(layout, n):
    """Run through the committed kernel's own offsets: each hand-off written once and read where it was written, the
    MFMA operands and accumulators where the lanes hold them; the out within the contract's bound of the reference
    (rounding a rank's own columns once, where vLLM rounds them twice) and the same bits on every rank."""
    group = draw(n, seed=n)
    outs = run_layout(layout, group)
    for out in outs[1:]:
        assert torch.equal(out, outs[0])
    got = bf16(outs[0])
    want = reference(group)
    bound = KIMI_K3_LATENT_MOE_TAIL.outputs["out"].error_bounds[0][1]
    allowed = bound.allowed_error(want.double(), 2.0**-8, None, None)
    assert ((got.double() - want.double()).abs() <= allowed).all()


def test_the_up_sets_split_the_token_groups(layout):
    """At every row count: each set of up workgroups 56 workgroups, one a column block; set 0 outside the pushing
    pool, the others in it; every token group in exactly one set."""
    L = layout
    assert L.up_sets(0) == 0
    roles = [L.up_role(bid) for bid in range(L.GRID)]
    for n in range(1, L.MAX_ROWS + 1):
        groups, sets = (n + L.TILE - 1) // L.TILE, L.up_sets(n)
        assert sets == min(groups, L.MAX_SETS)
        covered = []
        for s in range(sets):
            members = [bid for bid, (s_, _) in enumerate(roles) if s_ == s]
            assert sorted(roles[bid][1] for bid in members) == list(range(L.UP_CTAS))
            assert all((bid >= L.POOL) == (s == 0) for bid in members)
            covered += [s + sets * i for i in range(L.set_groups(s, n))]
        assert sorted(covered) == list(range(groups))


def test_the_layout_fits_the_runtimes_buffers(layout):
    """Every peer region in order, 16-byte aligned, inside the peer buffer: the tagged ones, the flag protocol's
    payloads and its flags apart, so neither protocol ever polls a word the other wrote."""
    bounds = sorted(regions(layout).items(), key=lambda kv: kv[1])
    peer = [b for name, b in bounds if name != "nbuf"]
    assert peer[0][0] == 0 and all(prev[1] == nxt[0] for prev, nxt in zip(peer, peer[1:]))
    assert all(start % 16 == 0 for start, _ in peer) and peer[-1][1] == layout.PEER_END
    assert layout.GRID == runtime.GRID
    assert layout.PEER_END <= runtime.PEER_BYTES and layout.SCRATCH_END <= runtime.SCRATCH_BYTES
    assert layout.LDS_WORDS * 4 <= 160 * 1024


# ---------------------------------------------------------------- the route


class _Op:
    """The op as the route sees it: its calls, what its dispatcher answers, its out."""

    def __init__(self, reason=None):
        self.calls, self.asked, self.reason = [], [], reason

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return torch.ones(args[0].shape[0], HIDDEN)

    def fallback_reason(self, *args, **kwargs):
        self.asked.append((args, kwargs))
        return self.reason


@pytest.fixture
def vllm_tp(monkeypatch):
    """vLLM's TP group as the route reads it: TP 8, this rank's index settable; and the setups it was asked for."""
    tp = SimpleNamespace(rank_in_group=0, world_size=TP, cpu_group=None, setups=[])
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_tp_group = lambda: tp
    distributed.get_tensor_model_parallel_rank = lambda: tp.rank_in_group
    distributed.get_tensor_model_parallel_world_size = lambda: tp.world_size
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)

    def setup(cpu_group, rank, world, device):
        tp.setups.append((rank, world, device))
        return SimpleNamespace(cpu_group=cpu_group)

    monkeypatch.setattr(runtime, "setup", setup)
    return tp


@pytest.fixture(autouse=True)
def _fresh_route(monkeypatch):
    monkeypatch.setattr(api, "_SEEN", set())
    monkeypatch.setattr(api, "_GROUPS", {})
    monkeypatch.setattr(api, "_CHECKED", {})
    monkeypatch.setattr(api, "_ENABLED", True)
    monkeypatch.setattr(api, "_CHECK", False)


@pytest.fixture
def op(monkeypatch):
    fake = _Op()
    monkeypatch.setattr(api, "kimi_k3_latent_moe_tail", fake)
    return fake


def runner(preshard=True, rank=0, reduced=False):
    """ROCmLatentMoERunner as _shard_up_proj_tail reads it: the norm, the up-projection (this rank's rows, or all)."""
    gen = torch.Generator().manual_seed(rank)
    rows = HIDDEN // TP if preshard else HIDDEN
    transform = SimpleNamespace(
        norm=SimpleNamespace(weight=torch.randn(LATENT, generator=gen).to(BF16), variance_epsilon=1e-5),
        up_proj=SimpleNamespace(weight=torch.randn(rows, LATENT, generator=gen).to(BF16)),
        row_sharded=preshard,
    )
    return SimpleNamespace(
        routed_output_transform=transform,
        _up_proj_preshard=preshard,
        _up_proj_shard_size=HIDDEN // TP,
        _fused_output_is_reduced=reduced,
    )


def outputs(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, LATENT, generator=gen).to(BF16), torch.randn(n, HIDDEN, generator=gen).to(BF16)


def test_the_route_takes_vllms_parameters():
    import inspect

    assert list(inspect.signature(api._shard_up_proj_tail).parameters) == [
        "self", "fused_output", "shared_output", "trunc_size",
    ]  # fmt: skip


def _changed(case):
    run, (fused, shared), trunc = runner(), outputs(4), None
    if case == "padded":
        trunc = HIDDEN
    elif case == "reduced":
        run = runner(reduced=True)
    elif case == "no_norm":
        run.routed_output_transform.norm = None
    elif case == "rows_0":
        fused, shared = outputs(0)
    elif case == "rows_129":
        fused, shared = outputs(129)
    elif case == "fp16":
        fused = fused.half()
    elif case == "strided":
        fused = torch.randn(LATENT, 4).to(BF16).T
    elif case == "shared_width":
        shared = shared[:, : HIDDEN // 2].contiguous()
    elif case == "norm_fp32":
        run.routed_output_transform.norm.weight = run.routed_output_transform.norm.weight.float()
    return run, fused, shared, trunc


@pytest.mark.parametrize(
    "case",
    ["padded", "reduced", "no_norm", "rows_0", "rows_129", "fp16", "strided", "shared_width", "norm_fp32"],
)
def test_what_no_contract_states_goes_back_to_vllm_untouched(vllm_tp, op, case):
    """Declined before the dispatcher is asked, the group set up or the op run."""
    run, fused, shared, trunc = _changed(case)
    assert api._shard_up_proj_tail(run, fused, shared, trunc) is FallbackReason.OUT_OF_CONTRACT
    assert op.asked == [] and op.calls == [] and vllm_tp.setups == []


def test_another_tp_size_goes_back_to_vllm(vllm_tp, op):
    vllm_tp.world_size = 4
    run, fused, shared, trunc = _changed(None)
    assert api._shard_up_proj_tail(run, fused, shared, trunc) is FallbackReason.OUT_OF_CONTRACT
    assert op.asked == [] and vllm_tp.setups == []


def test_a_call_the_dispatcher_declines_goes_back_before_the_setup(vllm_tp, op):
    op.reason = FallbackReason.NOT_QUALIFIED
    fused, shared = outputs(10)
    assert api._shard_up_proj_tail(runner(), fused, shared, None) is FallbackReason.NOT_QUALIFIED
    assert len(op.asked) == 1 and op.calls == [] and vllm_tp.setups == []


@pytest.mark.gpu  # the route asks torch.cuda whether a graph is being captured
def test_a_supported_call_hands_vllms_operands_over(vllm_tp, op):
    """vLLM's own tensors, no copy: the routed and shared partials, this rank's up rows, the norm's weight and eps;
    the group set up once, over vLLM's TP group."""
    run = runner()
    fused, shared = outputs(10)
    out = api._shard_up_proj_tail(run, fused, shared, None)
    assert out.shape == (10, HIDDEN)
    api._shard_up_proj_tail(run, *outputs(3), None)
    ((args, kwargs), _) = op.calls
    assert args[0] is fused and args[1] is shared
    assert args[2] is run.routed_output_transform.up_proj.weight
    assert args[3] is run.routed_output_transform.norm.weight
    assert kwargs == {"tp_size": TP, "eps": 1e-5}
    assert vllm_tp.setups == [(0, TP, fused.device)]


@pytest.mark.gpu  # the route asks torch.cuda whether a graph is being captured
@pytest.mark.parametrize("rank", [0, 3, 7])
def test_a_replicated_up_projection_hands_this_ranks_rows_over(vllm_tp, op, rank):
    vllm_tp.rank_in_group = rank
    run = runner(preshard=False)
    api._shard_up_proj_tail(run, *outputs(5), None)
    up = op.calls[0][0][2]
    shard = HIDDEN // TP
    assert up.shape == (shard, LATENT) and up.is_contiguous()
    assert up.data_ptr() == run.routed_output_transform.up_proj.weight[rank * shard].data_ptr()


@pytest.mark.gpu  # the route asks torch.cuda whether a graph is being captured
def test_every_rank_decides_alike(vllm_tp, op):
    """Ranks with their own tensors (values, up rows) take the same decision at every call, and set the group up at
    the same call: nothing a rank has alone reaches a decision."""
    calls = [outputs(n, seed=n) for n in (200, 10, 10, 3, 0, 128)]
    decisions = []
    for rank in range(TP):
        vllm_tp.rank_in_group, vllm_tp.setups = rank, []
        api._GROUPS.clear()
        run = runner(preshard=rank % 2 == 0, rank=rank)
        seen = []
        for fused, shared in calls:
            result = api._shard_up_proj_tail(run, fused + rank, shared - rank, None)
            seen.append((result if isinstance(result, FallbackReason) else "served", len(vllm_tp.setups)))
        decisions.append(seen)
    assert all(d == decisions[0] for d in decisions)
    assert decisions[0][0] == (FallbackReason.OUT_OF_CONTRACT, 0) and decisions[0][1] == ("served", 1)


def test_before_the_setup_a_capture_goes_back_to_vllm(vllm_tp, op, monkeypatch):
    """The peer memory is set up outside a graph capture: the capture records vLLM's tail."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert api._shard_up_proj_tail(runner(), *outputs(10), None) is FallbackReason.NOT_WARMED
    assert op.calls == [] and vllm_tp.setups == []


@pytest.mark.gpu  # the route asks torch.cuda whether a graph is being captured
def test_a_group_the_ranks_could_not_set_up_keeps_every_call_with_vllm(vllm_tp, op, monkeypatch):
    monkeypatch.setattr(runtime, "setup", lambda *args: vllm_tp.setups.append(args))
    for n in (10, 40):
        assert api._shard_up_proj_tail(runner(), *outputs(n), None) is FallbackReason.OUT_OF_CONTRACT
    assert op.calls == [] and len(vllm_tp.setups) == 1


@pytest.fixture
def vllms_tail(monkeypatch):
    """vLLM's tail, as the route's first-call check runs it: what it returns, its calls."""
    tail = SimpleNamespace(calls=[], scale=1.0)

    def call(self, fused_output, shared_output, trunc_size):
        tail.calls.append(fused_output.shape[0])
        return torch.ones(fused_output.shape[0], HIDDEN) * tail.scale

    monkeypatch.setattr(api, "_original", lambda route: call)
    monkeypatch.setattr(api, "_CHECK", True)
    return tail


@pytest.mark.gpu  # the route asks torch.cuda whether a graph is being captured
def test_the_first_eager_call_of_a_bucket_is_checked_against_vllms_tail(vllm_tp, op, vllms_tail, monkeypatch):
    agreements = []
    monkeypatch.setattr(runtime, "tp_agree", lambda ok, group: agreements.append(ok) or ok)
    for n in (10, 15, 16, 31):
        out = api._shard_up_proj_tail(runner(), *outputs(n), None)
        assert torch.equal(out, torch.ones(n, HIDDEN))  # the op's
    # the check's buckets, rows of one bit length: 8 .. 15, 16 .. 31
    assert vllms_tail.calls == [10, 16] and agreements == [True, True] and len(op.calls) == 4


@pytest.mark.gpu  # the route asks torch.cuda whether a graph is being captured
def test_a_bucket_any_rank_finds_different_stays_with_vllm(vllm_tp, op, vllms_tail, monkeypatch):
    """vLLM's out for the checked call; the bucket's later calls go back to vLLM unserved, other buckets are checked
    on their own."""
    vllms_tail.scale = 8.0  # as if a reduction were missing
    out = api._shard_up_proj_tail(runner(), *outputs(10), None)
    assert torch.equal(out, torch.full((10, HIDDEN), 8.0))
    assert api._shard_up_proj_tail(runner(), *outputs(12), None) is FallbackReason.INACCURATE
    assert len(op.calls) == 1
    vllms_tail.scale = 1.0
    monkeypatch.setattr(runtime, "tp_agree", lambda ok, group: False)  # another rank differs
    out = api._shard_up_proj_tail(runner(), *outputs(40), None)
    assert torch.equal(out, torch.ones(40, HIDDEN)) and api._CHECKED[(out.device, 6)] is False


@pytest.mark.gpu  # the route asks torch.cuda whether a graph is being captured
def test_the_handed_over_operands_are_a_point_of_the_contract(vllm_tp, op):
    """The contract's own key reads the route's call as a Point of the committed schema's (no device check on the
    CPU)."""
    api._shard_up_proj_tail(runner(), *outputs(10), None)
    ((args, kwargs),) = op.calls
    codec = FastKeyCodec.from_contract(KIMI_K3_LATENT_MOE_TAIL)
    call = dict(zip(("routed", "shared", "up_weight", "norm_weight"), args)) | kwargs
    call |= {"out_dtype": None, "out": None}
    assert FastKeyFnGenerator(codec, None).generate()(call) is not None


def test_the_plugins_step_hook_runs_the_registered_watches(monkeypatch):
    seen = []
    monkeypatch.setattr(plugin, "AFTER_STEP", [lambda: seen.append("watch")])
    assert plugin._execute(lambda x: seen.append("step") or x, 3) == 3
    assert seen == ["step", "watch"]


def test_one_rank_agrees_with_itself():
    assert runtime.tp_agree(True, None) and not runtime.tp_agree(False, None)


# ---------------------------------------------------------------- the committed schema


def test_the_committed_schema_is_its_canonical_text_at_its_path():
    path = committed_schema()
    text = path.read_bytes()
    schema = ContractRegistry.default().parse_schema(text, str(path))
    assert schema.schema_text().encode() == text
    uuid = schema_uuid(schema.name, OpContract.authored_parts(json.loads(text)))
    assert path == op_dir(schema.op_package) / schema_relative_path(schema.name, "flydsl", uuid)


def test_the_committed_schema_serves_every_row_bucket_up_to_128():
    schema = json.loads(committed_schema().read_text())
    rows = sorted(w["axes"]["n_tokens"] for w in schema["workloads"])
    assert {n.bit_length() for n in rows} == set(range(1, 9)) and max(rows) == 128
    assert {10, 40, 120} <= set(rows)
    kernel = schema["kernel"]
    assert kernel["requires"] == {"flydsl": ">=0.3.2"} and kernel["entry_point"] == "kernel.py::run"
    assert [s["path"] for s in kernel["sources"]] == [
        "kernel.py",
        "kernel_small.py",
        "layout.py",
        "layout_small.py",
    ]


def test_the_kernel_takes_the_call_keys_and_nothing_from_aiter():
    tree = ast.parse(kernel_source("kernel.py"))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    assert [a.arg for a in run.args.kwonlyargs] == list(KIMI_K3_LATENT_MOE_TAIL.kernel_call_keys)
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not any(m and m.split(".")[0] in ("aiter", "vllm") for m in imported)
    assert "RANGE_SAFE = True" in kernel_source("kernel.py")
