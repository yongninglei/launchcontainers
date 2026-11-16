"""Serve the rating page over one analysis's ``qc/`` folder (ARCHITECTURE.md §6, layer 2).

Moved from QC-registration's viewer and generalised: it shows every item that
``launchcontainers.core.slices.render_item`` pre-rendered (``qc/<item>/sub-*/ses-*[/<run>]/``
with a ``session.json``), stacks the item's layer PNGs per slice, and saves
True/False + comment per (item, sub, ses, run) to ``qc/qc_rater-<name>.tsv``
(``launchcontainers.core.ratings``). Nothing else is written. Stdlib only.

With a ``render`` hook (e.g. QC-registration), items on the work list that
are not rendered yet, or are stale, are rendered when first opened; the
hook is fingerprint-guarded, so a fresh item costs a few stat() calls.

Routes: ``/`` (page), ``/api/index``, ``/api/item`` (one item's session.json),
``/api/qc`` (GET ?rater=, POST),
``/png/<dir>/<file>.png`` (files under ``qc/`` only).

    lc qc --serve --analysis-dir D [--port 8000] [--host 127.0.0.1]
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import typer
from rich.console import Console

from launchcontainers.core import LCError
from launchcontainers.core.ratings import QC_DIR, load_ratings, save_rating
from launchcontainers.core.slices import find_items, item_dir, item_key, read_meta

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
_META_KEYS = (
    "item",
    "sub",
    "ses",
    "run",
    "dir",
    "question",
    "layers",
    "cuts_mm",
    "fingerprint",
    "resampled",
    "inputs",
)


RenderHook = Callable[
    [str, str, str, str], Any
]  # (item, sub, ses, run) -> render (fingerprint-guarded)
Key = tuple[str, str, str, str]


class Viewer:
    """What the handler serves: rendered items, plus ``expected`` ones a ``render``
    hook can render on first view (QC-registration's on-demand rendering)."""

    def __init__(
        self,
        analysis_dir: Path,
        title: str | None = None,
        pairs: set[tuple[str, str]] | None = None,
        expected: Iterable[Key] | None = None,
        render: RenderHook | None = None,
        question: str = "",
    ) -> None:
        self.root = Path(analysis_dir).resolve()
        self.qc = self.root / QC_DIR
        self.title = title or self.root.name
        self.pairs = pairs
        self.expected = [tuple(k) for k in (expected or ())]
        self.render = render
        self.question = question
        self.qc_lock = threading.Lock()
        self._guard = threading.Lock()
        self._locks: dict[Key, threading.Lock] = {}
        self.reload()

    def reload(self) -> None:
        keys = {item_key(m) for m in find_items(self.root)}
        if self.render is not None:
            keys |= set(self.expected)
        self.keys = sorted(
            k for k in keys if self.pairs is None or (k[1], k[2]) in self.pairs
        )

    def index(self) -> dict:
        self.reload()  # `lc qc` may have rendered more since the page was opened
        return {
            "analysis": self.title,
            "analysis_dir": str(self.root),
            "question": self.question,
            "items": [
                {"item": i, "sub": s, "ses": e, "run": r} for i, s, e, r in self.keys
            ],
        }

    def item(self, key: Key) -> dict:
        """The item's session.json (rendered first, with a hook, when missing or stale)."""
        if key not in self.keys:
            raise LCError(f"not in this analysis: {key}")
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        out = item_dir(self.root, *key)
        rendered = False
        with lock:
            if self.render is not None:
                t0 = time.time()
                res = self.render(*key)
                rendered = bool(getattr(res, "rendered", False))
                if rendered:
                    console.print(
                        f"  [green]rendered[/green] {out.relative_to(self.root)} ({time.time() - t0:.1f} s)"
                    )
            meta = read_meta(out)
        if meta is None:
            raise LCError(
                f"{out.relative_to(self.root)} is not rendered yet: run `lc qc` first"
            )
        meta["dir"] = out.relative_to(self.qc).as_posix()
        return {**{k: meta.get(k) for k in _META_KEYS}, "rendered_now": rendered}


def make_handler(viewer: Viewer) -> type[BaseHTTPRequestHandler]:
    static_dir = resources.files("launchcontainers.core") / "static"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(
            self, code: int, body: bytes, ctype: str, cache: str = "no-cache"
        ) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: object, code: int = 200) -> None:
            self._send(code, json.dumps(data).encode(), "application/json")

        def _error(self, code: int, message: str) -> None:
            self._json({"error": message}, code)

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in STATIC:
                    name, ctype = STATIC[url.path]
                    self._send(200, (static_dir / name).read_bytes(), ctype)
                elif url.path == "/api/index":
                    self._json(viewer.index())
                elif url.path == "/api/item":
                    self._json(
                        viewer.item(
                            (
                                q.get("item", ""),
                                q.get("sub", ""),
                                q.get("ses", ""),
                                q.get("run", ""),
                            )
                        )
                    )
                elif url.path == "/api/qc":
                    rows = load_ratings(viewer.root, q.get("rater", ""))
                    self._json({"rows": list(rows.values())})
                elif url.path.startswith("/png/"):
                    path = (viewer.qc / unquote(url.path[5:])).resolve()
                    if (
                        path.suffix != ".png"
                        or not path.is_relative_to(viewer.qc)
                        or not path.is_file()
                    ):
                        return self._error(404, "no such image")
                    # URLs carry ?v=<fingerprint>, so a re-render gets a new URL.
                    self._send(200, path.read_bytes(), "image/png", "max-age=86400")
                else:
                    self._error(404, "not found")
            except LCError as exc:
                self._error(400, str(exc))
            except Exception as exc:  # noqa: BLE001 — show it in the page
                console.print(f"  [red]ERROR[/red] {url.path}: {exc}")
                self._error(500, str(exc))

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/qc":
                return self._error(404, "not found")
            try:
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                )
                key = (
                    body.get("item"),
                    body.get("sub"),
                    body.get("ses"),
                    body.get("run") or "",
                )
                if key not in viewer.keys:
                    raise LCError("not an item of this analysis")
                with viewer.qc_lock:
                    row = save_rating(
                        viewer.root,
                        body.get("rater", ""),
                        key,
                        body.get("ok"),
                        body.get("comment", ""),
                    )
                self._json(row)
            except (LCError, json.JSONDecodeError, AttributeError, TypeError) as exc:
                self._error(400, str(exc))
            except OSError as exc:
                console.print(f"  [red]ERROR[/red] saving rating: {exc}")
                self._error(500, f"could not save: {exc}")

    return Handler


def serve(
    analysis_dir: Path | Viewer,
    port: int = 8000,
    host: str = "127.0.0.1",
    title: str | None = None,
    pairs: set[tuple[str, str]] | None = None,
    out: Console | None = None,
) -> None:
    """Blocking: serve until Ctrl-C. Pass a Viewer to serve one with a render hook."""
    out = out or console
    viewer = (
        analysis_dir
        if isinstance(analysis_dir, Viewer)
        else Viewer(analysis_dir, title, pairs)
    )
    server = ThreadingHTTPServer((host, port), make_handler(viewer))
    out.print(f"{viewer.title}: {len(viewer.keys)} item(s)  [dim]{viewer.qc}[/dim]")
    out.print(
        f"Open [bold]http://localhost:{port}/[/bold]  "
        f"[dim](remote: ssh -L {port}:localhost:{port} <host>)[/dim]"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        out.print("\n[yellow]viewer stopped[/yellow]")
    finally:
        server.server_close()


@app.command()
def main(
    analysis_dir: Path = typer.Option(
        ..., "--analysis-dir", "-w", help="Analysis directory (with qc/)"
    ),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="127.0.0.1 = this machine only (use an ssh tunnel)"
    ),
) -> None:
    """Serve the rating page for the pre-rendered QC items of one analysis."""
    console.rule("[bold]lc qc --serve[/bold]")
    serve(analysis_dir, port, host)


if __name__ == "__main__":
    app()
