#!/usr/bin/env python3
"""A FlyDSL kernel's register and LDS budget on gfx950, and the occupancy it reaches.

    waves per SIMD     = min(8, floor(512 / roundup8(align4(arch VGPRs) + AGPRs)))
    workgroups per CU  = min(by registers, floor(163840 / roundup1280(LDS bytes)))   W = waves per workgroup
    by registers       = floor(4 * waves per SIMD / W) for W <= 4, floor(waves per SIMD / ceil(W / 4)) above

The register rule is `gemm/mainloop.md` §7 (one 512-entry pool per SIMD, granule 8, arch rounded
to 4 before the AGPRs); the workgroup terms and the 1280-byte LDS block are `gemm/grid_pricing.md` §1 and §4
(measured at W = 1-6, 8 and 16; other W above 4 that are not a multiple of 4 are assumed to follow W = 5 and 6);
the 160 KiB LDS limit and the fact that a larger allocation still compiles are in `../flydsl/language/limits.md`.
The compiler's `; Occupancy:` comment skips the workgroup step. Also lists the register ceiling of every reachable
occupancy. Other per-CU limits (workgroup slots, SGPRs, barriers) are not modelled.

    python analysis/budget.py --arch-vgpr 128 --waves 8
    python analysis/budget.py --arch-vgpr 256 --agpr 81 --waves 4 --lds 65536
    python analysis/budget.py --isa 21_final_isa.s --waves 8      # resources from a FLYDSL_DUMP_IR=1 dump
    python analysis/budget.py --selftest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isa_diff import parse

POOL = 512  # VGPR entries per SIMD lane, arch and AGPR together
GRANULE = 8
MAX_WAVES_PER_SIMD = 8
SIMDS_PER_CU = 4
MAX_ARCH_VGPR = 256
LDS_BYTES = 163840
LDS_BLOCK = 1280  # LDS is allocated in blocks of 1280 bytes


def _up(value: int, step: int) -> int:
    return -(-value // step) * step


def registers(arch: int, agpr: int) -> int:
    """VGPR entries one wave takes from its SIMD's pool."""
    return _up(_up(arch, 4) + agpr, GRANULE)


def waves_per_simd(arch: int, agpr: int) -> int:
    return min(MAX_WAVES_PER_SIMD, POOL // max(registers(arch, agpr), GRANULE))


def _per_simd(groups: int, waves: int) -> int:
    """Wave slots the busiest SIMD needs for `groups` W-wave workgroups."""
    if waves <= SIMDS_PER_CU:
        return -(-groups * waves // SIMDS_PER_CU)
    return groups * -(-waves // SIMDS_PER_CU)


def workgroups_per_cu(arch: int, agpr: int, waves: int, lds: int) -> tuple[int, str]:
    per_simd = waves_per_simd(arch, agpr)
    if waves <= SIMDS_PER_CU:
        by_regs = SIMDS_PER_CU * per_simd // waves
    else:
        by_regs = per_simd // -(-waves // SIMDS_PER_CU)
    by_lds = LDS_BYTES // _up(lds, LDS_BLOCK) if lds else by_regs
    return min(by_regs, by_lds), "LDS" if by_lds < by_regs else "registers"


def ceilings(waves: int) -> list[tuple[int, int]]:
    """(workgroups per CU, most VGPR entries per wave) for every occupancy a W-wave workgroup can reach."""
    out = []
    for groups in range(1, SIMDS_PER_CU * MAX_WAVES_PER_SIMD // waves + 1):
        per_simd = _per_simd(groups, waves)
        if per_simd <= MAX_WAVES_PER_SIMD:
            out.append((groups, POOL // per_simd // GRANULE * GRANULE))
    return out


def report(arch: int, agpr: int, waves: int, lds: int) -> str:
    groups, limit = workgroups_per_cu(arch, agpr, waves, lds)
    lines = [
        (
            f"arch VGPRs {arch}, AGPRs {agpr}: {registers(arch, agpr)} entries per wave, "
            f"{waves_per_simd(arch, agpr)} waves per SIMD by registers"
        ),
        (
            f"{waves}-wave workgroups, {lds} B static LDS: {groups} workgroups per CU "
            f"({_per_simd(groups, waves)} waves on the busiest SIMD), set by {limit}"
        ),
    ]
    if arch > MAX_ARCH_VGPR:
        lines.append(f"over the {MAX_ARCH_VGPR} arch VGPRs a wave can address: the excess spills")
    if lds > LDS_BYTES:
        lines.append(f"over the {LDS_BYTES} B LDS limit: the compile still succeeds, the launch cannot fit")
    lines.append("ceilings (workgroups per CU: most VGPR entries per wave, arch rounded to 4 plus AGPRs):")
    lines += [f"  {g}: {regs}" for g, regs in ceilings(waves)]
    return "\n".join(lines)


def selftest() -> int:
    checks = [
        # grid_pricing §4: 96 VGPRs in 8-wave workgroups reads occupancy 5 per the compiler, 4 waves per SIMD real
        (
            waves_per_simd(96, 0) == 5 and workgroups_per_cu(96, 0, 8, 0)[0] == 2,
            "96 VGPRs, 8 waves: 2 workgroups",
        ),
        # grid_pricing §4: 128 -> 96 registers buys nothing with 8-wave workgroups
        (
            workgroups_per_cu(128, 0, 8, 0) == workgroups_per_cu(96, 0, 8, 0),
            "128 and 96 registers tie at 8 waves",
        ),
        # attention registers_occupancy §1: 256 arch + 81 AGPRs is one wave per SIMD
        (
            registers(256, 81) == 344 and waves_per_simd(256, 81) == 1,
            "256 + 81 registers: one wave per SIMD",
        ),
        # attention registers_occupancy §1: an 8-wave forward at 4 waves per SIMD needs at most 128 registers
        (
            dict(ceilings(8))[2] == 128 and workgroups_per_cu(129, 0, 8, 0)[0] == 1,
            "8 waves: 2 groups need <= 128",
        ),
        # ../flydsl/language/limits.md: 192 KiB of LDS does not fit
        (
            workgroups_per_cu(64, 0, 4, 196608)[0] == 0,
            "192 KiB of LDS: no workgroup fits",
        ),
        # grid_pricing §1: LDS in 1280 B blocks, 15361-16640 B hold 9 per CU, not 10
        (
            [workgroups_per_cu(32, 0, 1, b)[0] for b in (15360, 15361, 16000, 16640, 16641, 17920, 17921)]
            == [10, 9, 9, 9, 9, 9, 8],
            "LDS rounded up to 1280 B blocks",
        ),
        # grid_pricing §1 (measured): W = 5 at 97 + 31 registers holds 1, W = 6 at 80 holds 3
        (
            workgroups_per_cu(97, 31, 5, 0)[0] == 1
            and workgroups_per_cu(80, 0, 6, 0)[0] == 3
            and dict(ceilings(5))[2] == 128
            and dict(ceilings(6))[3] == 80,
            "W = 5 and 6: the busiest SIMD sets the count",
        ),
    ]
    for ok, what in checks:
        print(("ok    " if ok else "FAIL  ") + what)
    return 0 if all(ok for ok, _ in checks) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arch-vgpr", type=int, help="arch VGPRs per wave")
    parser.add_argument("--agpr", type=int, default=0, help="AGPRs per wave")
    parser.add_argument("--waves", type=int, help="waves per workgroup")
    parser.add_argument("--lds", type=int, help="static LDS bytes per workgroup")
    parser.add_argument("--isa", type=Path, help="read VGPRs, AGPRs and LDS from a NN_final_isa.s")
    parser.add_argument("--kernel", help="kernel in --isa, if it holds several")
    parser.add_argument("--selftest", action="store_true", help="check the rules on the cards' examples")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    arch, agpr, lds = args.arch_vgpr, args.agpr, args.lds
    if args.isa:
        kernels = parse(args.isa.read_text())
        name = args.kernel or next(iter(kernels), None)
        if name not in kernels:
            print(f"no kernel {name} in {args.isa}")
            return 1
        res = kernels[name].resources
        arch = res.get("arch_vgpr", arch)
        agpr = res.get("agpr", agpr)
        lds = res.get("lds_bytes", 0) if lds is None else lds
        print(f"{name}: {res}")
    if arch is None or not args.waves:
        parser.error("give --arch-vgpr (or --isa) and --waves")
    print(report(arch, agpr, args.waves, lds or 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
