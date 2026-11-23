"""Check a candidate implementation of the MLIP block contracts against captures.

A candidate is a Python module exposing any of

    conv_tp(x, y, w, spec)                    -> [E, dim_out]        (A1)
    symmetric_contraction(x, species, spec)   -> [N, sum C * d_out]  (A2)
    gather(h, idx)                            -> [E, F]              (B1)
    segment_sum(m, idx, num_segments)         -> [N, F]              (B2)

with the semantics of ``blocks.py`` (the reference, used when no candidate
is given). Inputs may be moved to a device with ``--device``. Gradients are
taken with ``torch.autograd.grad`` through the candidate (a device kernel
must register its backward) and compared with the captured cotangent
propagation: grad_x, grad_y, grad_w for A1, grad_x for A2, grad_src for B.

    python check.py fixtures/mace_mp0b2_tiny.pt                  # reference vs capture
    python check.py ~/.xtmp/mlip/captures/*.pt --candidate mypkg.mlip_kernels --device neuron
    python check.py fixtures/mace_mp0b2_tiny.pt --edge-cases     # synthetic B / A edge cases

Pass/fail uses the relative Frobenius error ``|got - ref| / |ref|``
against ``--rtol`` (default 2e-5 for fp32 captures; reductions in a
different order than MACE's CPU path are expected to land near 1e-6).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blocks as reference  # noqa: E402


def load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def rel(got, ref) -> float:
    got = got.detach().to("cpu", torch.float64)
    ref = ref.detach().to("cpu", torch.float64)
    if got.shape != ref.shape:
        raise AssertionError(f"shape {tuple(got.shape)} != {tuple(ref.shape)}")
    den = ref.norm()
    num = (got - ref).norm()
    if den == 0:
        return float(num)
    return float(num / den)


class Checker:
    def __init__(self, impl, device: str, rtol: float) -> None:
        self.impl, self.device, self.rtol = impl, torch.device(device), rtol
        self.rows: list = []
        self.failed = 0

    def record(
        self, name: str, quantity: str, err: float, rtol: float | None = None
    ) -> None:
        tol = self.rtol if rtol is None else rtol
        ok = err <= tol
        self.failed += 0 if ok else 1
        self.rows.append((name, quantity, err, "ok" if ok else "FAIL"))

    def dev(self, t, grad=False):
        if not torch.is_tensor(t):
            return t  # spec dicts stay on the host; the candidate places constants
        t = t.to(self.device)
        return t.requires_grad_(True) if grad and t.is_floating_point() else t

    def vjp(self, fn, inputs: dict, grad_out, want: list[str]):
        ins = {k: self.dev(v, grad=k in want) for k, v in inputs.items()}
        out = fn(**ins)
        if not want:
            return out, {}
        gs = torch.autograd.grad(
            out, [ins[k] for k in want], self.dev(grad_out), allow_unused=True
        )
        return out, dict(zip(want, gs))

    # A1
    def conv_tp(self, cap: dict) -> None:
        fn = getattr(self.impl, "conv_tp", None)
        if fn is None:
            return
        for a in cap["A1"]:
            name = f"A1 conv_tp L{a['layer']}"
            want = [k for k in ("x", "y", "w") if a.get("grad_" + k) is not None]
            out, gs = self.vjp(
                lambda x, y, w, spec: fn(x, y, w, spec),
                {"x": a["x"], "y": a["y"], "w": a["w"], "spec": a["spec"]},
                a["grad_out"],
                want,
            )
            self.record(name, "out", rel(out, a["out"]))
            for k in want:
                self.record(name, "grad_" + k, rel(gs[k], a["grad_" + k]))

    # A2
    def symcon(self, cap: dict) -> None:
        fn = getattr(self.impl, "symmetric_contraction", None)
        if fn is None:
            return
        for a in cap["A2"]:
            name = f"A2 symmetric_contraction L{a['layer']}"
            out, gs = self.vjp(
                lambda x, species, spec: fn(x, species, spec),
                {"x": a["x"], "species": a["species"], "spec": a["spec"]},
                a["grad_out"],
                ["x"],
            )
            self.record(name, "out", rel(out, a["out"]))
            self.record(name, "grad_x", rel(gs["x"], a["grad_x"]))

    # B
    def graph(self, cap: dict) -> None:
        ss = getattr(self.impl, "segment_sum", None)
        ga = getattr(self.impl, "gather", None)
        if ss is not None:
            for s in cap["B_scatter"]:
                name = f"B2 segment_sum {s['name']} ({s['index_role']}, F={s['src'].shape[1] if s['src'].dim() > 1 else 1})"
                want = ["m"] if s["grad_src"] is not None else []
                out, gs = self.vjp(
                    lambda m, idx: ss(m, idx, s["dim_size"]),
                    {"m": s["src"], "idx": s["index"]},
                    s["grad_out"],
                    want,
                )
                self.record(name, "out", rel(out, s["out"]))
                if want:
                    self.record(name, "grad_src", rel(gs["m"], s["grad_src"]))
            # force assembly: forces from dE/d(edge vector), two segment sums
            g = self.dev(cap["edge_vectors_grad"])
            snd, rcv = self.dev(cap["sender"]), self.dev(cap["receiver"])
            n = cap["stats"]["atoms"]
            f = ss(g, snd, n) - ss(g, rcv, n)
            self.record("B2 force assembly (F=3)", "forces", rel(f, cap["forces"]))
        if ga is not None:
            for gth in cap["B_gather"]:
                name = f"B1 gather {gth['name']} (F={gth['src'].shape[1]})"
                want = ["h"] if gth["grad_src"] is not None else []
                out, gs = self.vjp(
                    lambda h, idx: ga(h, idx),
                    {"h": gth["src"], "idx": gth["index"]},
                    gth["grad_out"],
                    want,
                )
                self.record(name, "out", rel(out, gth["out"]))
                if want:
                    self.record(name, "grad_src", rel(gs["h"], gth["grad_src"]))

    def report(self, title: str) -> None:
        print(f"\n== {title}")
        for name, q, err, st in self.rows:
            print(f"  {st:4s} {err:9.2e}  {name}: {q}")
        self.rows = []


def edge_cases(impl, device: str, checker: Checker) -> None:
    """Synthetic cases a kernel must get right beyond the captured graphs."""
    dev = torch.device(device)
    ss = getattr(impl, "segment_sum", None)
    ga = getattr(impl, "gather", None)
    g = torch.Generator().manual_seed(0)

    def both(name, m, idx, n):
        ref = reference.segment_sum(m.double(), idx, n)
        mm = m.to(dev).requires_grad_(True)
        out = ss(mm, idx.to(dev), n)
        checker.record(name, "out", rel(out, ref), 1e-5)
        go = torch.randn(out.shape, generator=g, dtype=m.dtype)
        (gm,) = torch.autograd.grad(out, mm, go.to(dev))
        checker.record(name, "grad_src", rel(gm, go[idx]), 1e-6)
        if device != "cpu":
            again = ss(m.to(dev), idx.to(dev), n)
            checker.record(
                name,
                "deterministic (bitwise rerun)",
                float((again.cpu() != out.detach().cpu()).sum()),
                0,
            )

    if ss is not None:
        F = 64
        # empty segments: atoms 0, 5 and the last have no in-edges
        idx = torch.randint(1, 30, (200,), generator=g)
        idx[idx == 5] = 6
        both(
            "B2 empty segments (N=31, rows 0, 5, 30 empty)",
            torch.randn(200, F, generator=g),
            idx,
            31,
        )
        # no edges at all
        both("B2 zero edges", torch.zeros(0, F), torch.zeros(0, dtype=torch.long), 7)
        # heavy repetition: every edge on one receiver
        both(
            "B2 one segment, 4096 repeats",
            torch.randn(4096, F, generator=g),
            torch.zeros(4096, dtype=torch.long),
            3,
        )
        # unsorted vs receiver-sorted index give the same sums
        idx = torch.randint(0, 50, (2000,), generator=g)
        m = torch.randn(2000, F, generator=g)
        both("B2 unsorted index", m, idx, 50)
        perm = torch.argsort(idx, stable=True)
        both("B2 receiver-sorted index", m[perm], idx[perm], 50)
        # F = 1 and F = 3 (density, forces)
        both(
            "B2 F=1",
            torch.randn(500, 1, generator=g),
            torch.randint(0, 40, (500,), generator=g),
            40,
        )
        both(
            "B2 F=3",
            torch.randn(500, 3, generator=g),
            torch.randint(0, 40, (500,), generator=g),
            40,
        )
        # flat index E*F beyond 2^24 (a 2^24 boundary in flattened scatter lowering)
        E, F = 4100, 4096 + 1024  # E*F = 20.99M > 2^24 = 16.78M
        idx = torch.randint(0, 120, (E,), generator=g)
        both("B2 E*F > 2^24 (E=4100, F=5120)", torch.randn(E, F, generator=g), idx, 120)
        # atom index beyond 2^24 (N > 2^24 rows: index values not fp32-exact)
        N = (1 << 24) + 3
        idx = torch.tensor([N - 1, N - 2, (1 << 24) + 1, 1 << 24, 0, N - 1])
        m = torch.randn(len(idx), 1, generator=g)
        both("B2 index values > 2^24 (N = 2^24 + 3, F=1)", m, idx, N)
    if ga is not None:
        h = torch.randn(40, 16, generator=g)
        idx = torch.tensor([3, 3, 3, 0, 39, 7, 3])
        out = ga(h.to(dev).requires_grad_(True), idx.to(dev))
        checker.record("B1 repeated indices", "out", rel(out, h[idx]), 0)
        hh = h.to(dev).requires_grad_(True)
        go = torch.randn(len(idx), 16, generator=g)
        (gh,) = torch.autograd.grad(ga(hh, idx.to(dev)), hh, go.to(dev))
        checker.record(
            "B1 repeated indices",
            "grad_src",
            rel(gh, reference.segment_sum(go, idx, 40)),
            1e-6,
        )
        out = ga(h.to(dev), torch.zeros(0, dtype=torch.long, device=dev))
        checker.record("B1 zero edges", "shape", float(tuple(out.shape) != (0, 16)), 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("captures", nargs="+")
    ap.add_argument(
        "--candidate", default=None, help="module path; default: the reference"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rtol", type=float, default=2e-5)
    ap.add_argument("--edge-cases", action="store_true")
    ap.add_argument("--blocks", default="A1,A2,B")
    args = ap.parse_args()
    impl = importlib.import_module(args.candidate) if args.candidate else reference
    chk = Checker(impl, args.device, args.rtol)
    want = set(args.blocks.split(","))
    for path in args.captures:
        cap = load(path)
        if "A1" in want:
            chk.conv_tp(cap)
        if "A2" in want:
            chk.symcon(cap)
        if "B" in want:
            chk.graph(cap)
        st = cap["stats"]
        chk.report(
            f"{path} ({cap['meta']['structure']}: {st['atoms']} atoms, {st['edges']} edges)"
        )
    if args.edge_cases:
        edge_cases(impl, args.device, chk)
        chk.report("edge cases")
    print(f"\n{chk.failed} failed")
    sys.exit(1 if chk.failed else 0)


if __name__ == "__main__":
    main()
