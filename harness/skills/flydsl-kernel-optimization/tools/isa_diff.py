#!/usr/bin/env python3
"""Compare two FlyDSL ISA dumps without the noise of register renaming and labels.

`FLYDSL_DUMP_IR=1` writes `NN_final_isa.s` per kernel. Two builds of one kernel usually differ in register
numbers and branch labels on every line, which hides the change that matters. This prints, per kernel:

- the resources of each side: arch VGPRs, AGPRs, SGPRs, spills, static LDS, scratch;
- the opcode counts that differ;
- with --stream, a unified diff of the instruction streams with every register number and label replaced.

    python tools/isa_diff.py before/21_final_isa.s after/21_final_isa.s
    python tools/isa_diff.py a.s b.s --kernel gemm_kernel_0 --stream
    python tools/isa_diff.py --selftest
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REG_RANGE_RE = re.compile(r"\b([vsa])\[\d+:\d+\]")
REG_RE = re.compile(r"\b([vsa])\d+\b")
LABEL_RE = re.compile(r"\.L\w+")
DIRECTIVE_RE = re.compile(r"^\s*\.(amdhsa_\w+)\s+(\S+)")
META_RE = re.compile(r"^\s*(?:-\s+)?\.(\w+):\s+(\S+)")
# amdhsa directive -> resource name; on gfx950 AGPRs start at accum_offset in one register file
DIRECTIVES = {
    "amdhsa_next_free_vgpr": "vgpr_total",
    "amdhsa_accum_offset": "arch_vgpr",
    "amdhsa_next_free_sgpr": "sgpr_next_free",
    "amdhsa_group_segment_fixed_size": "lds_bytes",
    "amdhsa_private_segment_fixed_size": "scratch_bytes",
}
METADATA = ("vgpr_spill_count", "sgpr_spill_count")
RESOURCES = (
    "arch_vgpr",
    "agpr",
    "vgpr_total",
    "sgpr_next_free",
    "vgpr_spill_count",
    "sgpr_spill_count",
    "lds_bytes",
    "scratch_bytes",
)


@dataclass
class Kernel:
    name: str
    lines: list[str] = field(default_factory=list)  # instructions, normalized
    resources: dict[str, int] = field(default_factory=dict)

    @property
    def opcodes(self) -> Counter:
        return Counter(line.split()[0] for line in self.lines)


def normalize(instruction: str) -> str:
    """One instruction with register numbers and labels replaced, so renaming leaves it unchanged."""
    text = REG_RANGE_RE.sub(r"\1[#]", instruction)
    text = REG_RE.sub(r"\1#", text)
    text = LABEL_RE.sub(".L", text)
    return " ".join(text.replace(",", ", ").split()).replace(" ,", ",")


def parse(text: str) -> dict[str, Kernel]:
    """Kernels of one ISA dump by name: their normalized instructions and resources."""
    kernels: dict[str, Kernel] = {}
    names = re.findall(r"^\s*\.globl\s+(\S+)", text, re.MULTILINE)
    lines = text.splitlines()
    for name in names:
        kernel = kernels.setdefault(name, Kernel(name))
        try:
            start = lines.index(f"{name}:") + 1
        except ValueError:
            continue
        for line in lines[start:]:
            code = re.split(r"\s;|//", line, maxsplit=1)[0].strip()
            if code.startswith(".Lfunc_end"):
                break
            if not code or code.endswith(":") or code.startswith("."):
                continue
            kernel.lines.append(normalize(code))
    current = None
    for line in lines:
        directive = DIRECTIVE_RE.match(line)
        if line.strip().startswith(".amdhsa_kernel "):
            current = kernels.get(line.split()[1])
        elif directive and current and directive.group(1) in DIRECTIVES:
            current.resources[DIRECTIVES[directive.group(1)]] = int(directive.group(2), 0)
    meta, entry = text.split("amdhsa.kernels:", 1)[-1], {}
    for line in meta.splitlines() + ["  - "]:
        if line.startswith("  - ") and entry.get("name") in kernels:
            for key in METADATA:
                if key in entry:
                    kernels[entry["name"]].resources[key] = int(entry[key])
        if line.startswith("  - "):
            entry = {}
        match = META_RE.match(line)
        if match:
            entry[match.group(1)] = match.group(2)
    for kernel in kernels.values():
        res = kernel.resources
        if "vgpr_total" in res and "arch_vgpr" in res:
            # accum_offset is the arch count rounded up to 4, where AGPRs start; with none, it can exceed the total
            res["agpr"] = max(0, res["vgpr_total"] - res["arch_vgpr"])
            res["arch_vgpr"] = min(res["arch_vgpr"], res["vgpr_total"])
    return kernels


def report(a: Kernel, b: Kernel, stream: bool) -> str:
    out = [f"== {a.name}", f"{'':20} {'A':>10} {'B':>10}"]
    for key in RESOURCES:
        va, vb = a.resources.get(key, "-"), b.resources.get(key, "-")
        mark = "" if va == vb else "   *"
        out.append(f"{key:20} {va!s:>10} {vb!s:>10}{mark}")
    out.append(f"{'instructions':20} {len(a.lines):>10} {len(b.lines):>10}")
    ca, cb = a.opcodes, b.opcodes
    deltas = sorted(
        ((op, cb[op] - ca[op]) for op in set(ca) | set(cb) if ca[op] != cb[op]),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    if deltas:
        out.append("opcode counts that differ (B - A):")
        out += [f"  {op:40} {ca[op]:>6} -> {cb[op]:<6} {delta:+d}" for op, delta in deltas]
    else:
        out.append("same opcode counts")
    if stream:
        diff = list(difflib.unified_diff(a.lines, b.lines, "A", "B", lineterm="", n=2))
        out += diff if diff else ["same instruction stream after normalization"]
    return "\n".join(out)


SELFTEST_A = """\t.globl\tk
k:
\ts_load_dwordx4 s[8:11], s[0:1], 0x0
.LBB0_1:
\tv_mfma_f32_16x16x32_bf16 a[0:3], v[4:7], v[8:11], a[0:3] ; comment
\ts_cbranch_scc1 .LBB0_1
.Lfunc_end0:
\t.amdhsa_kernel k
\t\t.amdhsa_group_segment_fixed_size 0
\t\t.amdhsa_next_free_vgpr 144
\t\t.amdhsa_accum_offset 144
\t.end_amdhsa_kernel
amdhsa.kernels:
  - .agpr_count:     0
    .name:           k
    .vgpr_spill_count: 0
"""
SELFTEST_B = (
    SELFTEST_A.replace("s[8:11]", "s[12:15]")
    .replace("v[4:7]", "v[20:23]")
    .replace("LBB0_1", "LBB0_7")
    .replace("\ts_cbranch", "\tscratch_store_dword off, v3, s0\n\ts_cbranch")
    .replace("next_free_vgpr 144", "next_free_vgpr 136")
    .replace("accum_offset 144", "accum_offset 128")
    .replace("vgpr_spill_count: 0", "vgpr_spill_count: 94")
)


def selftest() -> int:
    a, b = parse(SELFTEST_A)["k"], parse(SELFTEST_B)["k"]
    renamed = parse(SELFTEST_A.replace("v[4:7]", "v[40:43]").replace("LBB0_1", "LBB0_9"))["k"]
    checks = [
        (
            a.lines == renamed.lines,
            "register and label renaming leaves the stream unchanged",
        ),
        (
            b.opcodes - a.opcodes == Counter({"scratch_store_dword": 1}),
            "an added spill store is the only opcode delta",
        ),
        (
            a.resources
            == {
                "vgpr_total": 144,
                "arch_vgpr": 144,
                "agpr": 0,
                "lds_bytes": 0,
                "vgpr_spill_count": 0,
            },
            "resources of A",
        ),
        (
            b.resources["agpr"] == 8 and b.resources["vgpr_spill_count"] == 94,
            "AGPRs from accum_offset, spills from metadata",
        ),
        (
            parse(
                SELFTEST_A.replace("next_free_vgpr 144", "next_free_vgpr 23").replace(
                    "accum_offset 144", "accum_offset 24"
                )
            )["k"].resources
            == {
                "vgpr_total": 23,
                "arch_vgpr": 23,
                "agpr": 0,
                "lds_bytes": 0,
                "vgpr_spill_count": 0,
            },
            "no AGPRs and an arch count not a multiple of 4: AGPRs 0",
        ),
    ]
    for ok, what in checks:
        print(("ok    " if ok else "FAIL  ") + what)
    return 0 if all(ok for ok, _ in checks) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("a", nargs="?", type=Path, help="first NN_final_isa.s")
    parser.add_argument("b", nargs="?", type=Path, help="second NN_final_isa.s")
    parser.add_argument("--kernel", help="compare only this kernel")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="also diff the normalized instruction streams",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="check the normalization on built-in snippets",
    )
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.a and args.b):
        parser.error("give two ISA files, or --selftest")
    ka, kb = parse(args.a.read_text()), parse(args.b.read_text())
    names = [args.kernel] if args.kernel else sorted(set(ka) & set(kb))
    missing = sorted(set(ka) ^ set(kb))
    if missing and not args.kernel:
        print(f"kernels in one file only: {', '.join(missing)}")
    for name in names:
        if name not in ka or name not in kb:
            print(f"no kernel {name} in both files")
            return 1
        print(report(ka[name], kb[name], args.stream))
    return 0


if __name__ == "__main__":
    sys.exit(main())
