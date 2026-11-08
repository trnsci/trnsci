#!/usr/bin/env python
"""Report where the umbrella site and packaging trail PyPI (trnsci/trnsci#151).

The mkdocs hook (``hooks/versions.py``) keeps generated version cells current,
but it can't rewrite prose. This script finds the places that need a person:

1. ``[[described:<lib>=X]]`` tokens in ``docs/`` and ``README.md`` where PyPI
   has a newer release than the one the hand-written row describes.
2. ``pyproject.toml`` extras whose minimum for a library is below that
   library's latest minor release on PyPI, and per-library extras that
   disagree with ``[all]``.
3. The published ``trnsci`` meta-package: its ``[all]`` minimums differ from
   the repo's ``pyproject.toml`` (the repo was bumped but not released).
4. The umbrella ``[torchneuron]`` extra differs from a library's own
   ``[torchneuron]`` extra in its latest release.

Output is a Markdown report on stdout (or ``--output FILE``).

Exit status: 0 no drift, 1 drift found, 2 PyPI could not be read. Standard
library only, Python >= 3.11.

    python scripts/check_docs_drift.py
    python scripts/check_docs_drift.py --output drift.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBS = ("trnfft", "trnblas", "trnrand", "trnsolver", "trnsparse", "trntensor")
# Per-library extra name in the umbrella pyproject.toml.
EXTRA = {
    "trnfft": "fft",
    "trnblas": "blas",
    "trnrand": "rand",
    "trnsolver": "solver",
    "trnsparse": "sparse",
    "trntensor": "tensor",
}
DESCRIBED = re.compile(r"\[\[described:([a-z]+)=([0-9][0-9A-Za-z.]*)\]\]")
REQ = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(?:\[[^\]]*\])?\s*([^;]*?)\s*(?:;\s*(.*))?$")
MIN = re.compile(r">=\s*([0-9][0-9A-Za-z.]*)")


class PyPIError(RuntimeError):
    pass


def vkey(v: str) -> tuple[int, ...]:
    parts = [int(p) for p in re.findall(r"\d+", v)[:3]]
    return tuple(parts + [0] * (3 - len(parts)))


def pypi(name: str) -> dict:
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.load(r)
    except Exception as e:
        raise PyPIError(f"{url}: {e}") from e


def parse_req(req: str) -> tuple[str, str, str]:
    """Return (name, specifier, marker) for a PEP 508 string."""
    m = REQ.match(req)
    if not m:
        return req.strip(), "", ""
    return m.group(1).lower(), m.group(2).replace(" ", ""), (m.group(3) or "").strip()


def extra_reqs(requires_dist: list[str], extra: str) -> set[str]:
    """The requirements of one extra, from a PyPI ``requires_dist`` list."""
    out = set()
    for req in requires_dist or []:
        name, spec, marker = parse_req(req)
        if re.search(rf"extra\s*==\s*['\"]{re.escape(extra)}['\"]", marker):
            out.add(f"{name}{spec}")
    return out


def minimum(reqs: list[str], lib: str) -> str | None:
    for req in reqs:
        name, spec, _ = parse_req(req)
        if name == lib:
            m = MIN.search(spec)
            return m.group(1) if m else None
    return None


def described_tokens() -> list[tuple[str, int, str, str]]:
    files = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]
    found = []
    for path in files:
        rel = path.relative_to(ROOT)
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for m in DESCRIBED.finditer(line):
                found.append((str(rel), lineno, m.group(1), m.group(2)))
    return found


def check() -> list[str]:
    latest: dict[str, str] = {}
    lib_info: dict[str, dict] = {}
    for lib in LIBS:
        lib_info[lib] = pypi(lib)
        latest[lib] = lib_info[lib]["info"]["version"]
    umbrella = pypi("trnsci")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project.get("optional-dependencies", {})
    drift: list[str] = []

    # 1. Hand-written rows that describe an older release.
    for rel, lineno, lib, described in described_tokens():
        if lib not in latest:
            drift.append(
                f"`{rel}:{lineno}`: `[[described:{lib}=...]]` names an unknown library"
            )
        elif vkey(latest[lib]) > vkey(described):
            drift.append(
                f"`{rel}:{lineno}`: row describes {lib} {described}; PyPI has "
                f"[{latest[lib]}](https://pypi.org/project/{lib}/{latest[lib]}/). "
                "Update the row (and its #35 validation link)."
            )

    # 2. Repo extras below the latest minor, or inconsistent with [all].
    all_reqs = extras.get("all", [])
    for lib in LIBS:
        lo = minimum(all_reqs, lib)
        if lo is None:
            drift.append(f"`pyproject.toml`: `[all]` has no `>=` minimum for {lib}")
            continue
        if vkey(lo)[:2] < vkey(latest[lib])[:2]:
            drift.append(
                f"`pyproject.toml`: `[all]` requires {lib}>={lo}; latest is {latest[lib]} "
                "(a newer minor). Bump the minimum if the new release is the one to require."
            )
        per = minimum(extras.get(EXTRA[lib], []), lib)
        if per != lo:
            drift.append(
                f"`pyproject.toml`: `[{EXTRA[lib]}]` requires {lib}>={per}, `[all]` >={lo}"
            )

    # 3. The published meta-package lags the repo.
    published = umbrella["info"]["version"]
    pub_all = extra_reqs(umbrella["info"].get("requires_dist") or [], "all")
    repo_all = {"{}{}".format(*parse_req(r)[:2]) for r in all_reqs}
    if pub_all != repo_all:
        drift.append(
            f"PyPI `trnsci` {published} `[all]` is `{', '.join(sorted(pub_all)) or '(none)'}`; "
            f"the repo has `{', '.join(sorted(repo_all))}`. Release the meta-package."
        )

    # 4. [torchneuron] parity with the libraries.
    repo_tn = {"{}{}".format(*parse_req(r)[:2]) for r in extras.get("torchneuron", [])}
    for lib in LIBS:
        lib_tn = extra_reqs(
            lib_info[lib]["info"].get("requires_dist") or [], "torchneuron"
        )
        if lib_tn and lib_tn != repo_tn:
            drift.append(
                f"`[torchneuron]`: {lib} {latest[lib]} has `{', '.join(sorted(lib_tn))}`; "
                f"the umbrella has `{', '.join(sorted(repo_tn)) or '(none)'}`"
            )

    return drift


def report(drift: list[str]) -> str:
    if not drift:
        return "No drift: the site's described releases and the extras match PyPI.\n"
    lines = [
        "`scripts/check_docs_drift.py` found places where the umbrella site or "
        "packaging trails PyPI:",
        "",
        *[f"- [ ] {d}" for d in drift],
        "",
        "The generated cells (`[[pypi:]]`, `[[min:]]`) update on their own; these "
        "need a person. Release checklist: each library's release flow has an "
        '"Update umbrella docs/extras" item.',
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path, help="write the Markdown report here")
    args = ap.parse_args(argv)
    try:
        drift = check()
    except PyPIError as e:
        print(f"check_docs_drift: PyPI lookup failed: {e}", file=sys.stderr)
        return 2
    text = report(drift)
    if args.output:
        args.output.write_text(text)
    sys.stdout.write(text)
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
