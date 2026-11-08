"""mkdocs hook: fill in sub-project versions at build time, so they can't go stale.

Pages use three tokens, replaced before Markdown rendering:

- ``[[pypi:trnfft]]``   -> the latest PyPI release, linked to its PyPI page.
- ``[[min:trnfft]]``    -> the umbrella's ``trnsci[all]`` minimum (``>=0.29``),
  read from ``pyproject.toml``.
- ``[[described:trnfft=0.29.0]]`` -> ``0.29.0``, the release a hand-written
  row describes. If PyPI has a newer release, a short note naming it is
  appended, so prose tied to an old release flags itself instead of silently
  aging.

Versions come from the PyPI JSON API. If PyPI can't be reached (offline
builds, or ``TRNSCI_DOCS_OFFLINE=1``), the hook falls back to the version in
the sub-project's ``pyproject.toml`` that docs.yml clones next to this repo,
and then to a plain PyPI link. A token naming an unknown library is a
warning, so ``mkdocs build --strict`` catches typos.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tomllib
import urllib.request
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.trnsci_versions")

# PyPI name -> directory name inside the umbrella layout (see docs.yml).
LIBS = {
    "trnfft": "trnfft",
    "trnblas": "trnblas",
    "trnrand": "trnrand",
    "trnsolver": "trnsolve",
    "trnsparse": "trnsparse",
    "trntensor": "trntensor",
}

TOKEN = re.compile(r"\[\[(pypi|min|described):([a-z]+)(?:=([0-9][0-9A-Za-z.]*))?\]\]")

_latest: dict[str, str | None] = {}
_minimum: dict[str, str | None] = {}


def _vkey(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", v)[:3])


def _from_pypi(name: str) -> str | None:
    if os.environ.get("TRNSCI_DOCS_OFFLINE") == "1":
        return None
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.load(r)["info"]["version"]
    except Exception as e:  # network, HTTP or JSON error: fall back
        log.info("trnsci_versions: PyPI lookup for %s failed (%s)", name, e)
        return None


def _from_pyproject(path: Path) -> str | None:
    try:
        return tomllib.loads(path.read_text())["project"]["version"]
    except Exception:
        return None


def on_config(config, **kwargs):
    root = Path(config.config_file_path).parent
    for name, d in LIBS.items():
        _latest[name] = _from_pypi(name) or _from_pyproject(root / d / "pyproject.toml")
    try:
        extras = tomllib.loads((root / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]["all"]
    except Exception:
        extras = []
    for req in extras:
        m = re.match(r"([A-Za-z0-9_.-]+)\s*(>=\s*[0-9][0-9.]*)", req)
        if m and m.group(1) in LIBS:
            _minimum[m.group(1)] = m.group(2).replace(" ", "")
    log.info("trnsci_versions: latest %s; minimums %s", _latest, _minimum)
    return config


def _render(kind: str, name: str, described: str | None, page) -> str:
    latest = _latest.get(name)
    pypi = f"https://pypi.org/project/{name}/"
    if kind == "pypi":
        return f"[{latest}]({pypi}{latest}/)" if latest else f"[PyPI]({pypi})"
    if kind == "min":
        return f"`{_minimum[name]}`" if _minimum.get(name) else "(see `pyproject.toml`)"
    # described
    if not described:
        log.warning(
            "trnsci_versions: %s: [[described:%s]] needs =VERSION",
            page.file.src_uri,
            name,
        )
        return name
    if latest and _vkey(latest) > _vkey(described):
        log.info(
            "trnsci_versions: %s describes %s %s; PyPI has %s",
            page.file.src_uri,
            name,
            described,
            latest,
        )
        return f"{described} (newer on PyPI: [{latest}]({pypi}{latest}/))"
    return described


def on_page_markdown(markdown, page, config, files, **kwargs):
    def sub(m: re.Match) -> str:
        kind, name, described = m.group(1), m.group(2), m.group(3)
        if name not in LIBS:
            log.warning(
                "trnsci_versions: %s: unknown library %r", page.file.src_uri, name
            )
            return m.group(0)
        return _render(kind, name, described, page)

    return TOKEN.sub(sub, markdown)
