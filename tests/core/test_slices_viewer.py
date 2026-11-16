"""Pre-rendered QC items (slices) and the rating server (viewer)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")
pytest.importorskip("PIL")
pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

from launchcontainers.core.ratings import load_ratings  # noqa: E402
from launchcontainers.core.slices import Layer, find_items, render_item  # noqa: E402
from launchcontainers.core.viewer import Viewer, make_handler  # noqa: E402


def _imgs(tmp_path):
    rng = np.random.default_rng(0)
    head = np.zeros((20, 24, 18), np.float32)
    head[4:16, 4:20, 3:15] = 100 + rng.random((12, 16, 12)) * 50
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(head, aff), t1)
    lab = (head > 0).astype(np.int16) * 3
    lab[4:10] = 2 * (head[4:10] > 0)
    labels = tmp_path / "aseg.nii.gz"
    nib.save(nib.Nifti1Image(lab, aff), labels)
    low = tmp_path / "b0.nii.gz"  # coarser grid: resampled for display
    nib.save(
        nib.Nifti1Image(head[::2, ::2, ::2].copy(), np.diag([4.0, 4.0, 4.0, 1.0])), low
    )
    return t1, labels, low


def test_render_item_is_fingerprinted(tmp_path):
    t1, labels, low = _imgs(tmp_path)
    ana = tmp_path / "ana"
    layers = [
        Layer("b0", "b0", "map", low, cmap="hot"),
        Layer("aseg", "aseg", "labels", labels),
        Layer("edge", "edge", "outline", labels, show=False),
    ]
    r = render_item(
        ana, "dwi_on_t1", "01", "02", base=t1, layers=layers, question="ok?", n_slices=3
    )
    assert r.rendered and r.preview.is_file()
    assert "coronal-0_base.png" in r.files and "axial-2_edge.png" in r.files
    meta = json.loads((r.out / "session.json").read_text())
    assert [ly["key"] for ly in meta["layers"]] == ["base", "b0", "aseg", "edge"]
    assert meta["resampled"] == ["b0"] and set(meta["inputs"]) == {
        "base",
        "b0",
        "aseg",
        "edge",
    }
    again = render_item(
        ana, "dwi_on_t1", "01", "02", base=t1, layers=layers, n_slices=3
    )
    assert not again.rendered  # same inputs and settings: nothing re-rendered
    changed = render_item(
        ana, "dwi_on_t1", "01", "02", base=t1, layers=layers, n_slices=5
    )
    assert changed.rendered
    items = find_items(ana)
    assert [(m["item"], m["sub"], m["ses"], m["run"]) for m in items] == [
        ("dwi_on_t1", "01", "02", "")
    ]


def test_viewer_serves_items_and_saves_ratings(tmp_path):
    t1, labels, _ = _imgs(tmp_path)
    ana = tmp_path / "ana"
    render_item(
        ana,
        "seg",
        "01",
        "01",
        base=t1,
        layers=[Layer("aseg", "aseg", "labels", labels)],
        n_slices=3,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Viewer(ana)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        idx = json.load(urllib.request.urlopen(f"{base}/api/index"))
        assert idx["items"] == [{"item": "seg", "sub": "01", "ses": "01", "run": ""}]
        it = json.load(
            urllib.request.urlopen(f"{base}/api/item?item=seg&sub=01&ses=01&run=")
        )
        assert it["layers"][1]["key"] == "aseg" and it["rendered_now"] is False
        png = urllib.request.urlopen(f"{base}/png/{it['dir']}/axial-0_aseg.png").read()
        assert png[:4] == b"\x89PNG"
        assert b"app.js" in urllib.request.urlopen(f"{base}/").read()
        body = json.dumps(
            {
                "rater": "tiger",
                "item": "seg",
                "sub": "01",
                "ses": "01",
                "run": "",
                "ok": False,
                "comment": "bad  cut",
            }
        ).encode()
        req = urllib.request.Request(
            f"{base}/api/qc", body, {"Content-Type": "application/json"}
        )
        assert json.load(urllib.request.urlopen(req))["ok"] == "False"
        assert (
            load_ratings(ana, "tiger")[("seg", "01", "01", "")]["comment"] == "bad cut"
        )
        with pytest.raises(urllib.error.HTTPError):  # outside qc/
            urllib.request.urlopen(f"{base}/png/../../t1.nii.gz")
    finally:
        server.shutdown()


def test_viewer_renders_expected_items_on_demand(tmp_path):
    t1, labels, _ = _imgs(tmp_path)
    ana = tmp_path / "ana"
    calls = []

    def hook(item, sub, ses, run):
        calls.append((item, sub, ses, run))
        return render_item(
            ana,
            item,
            sub,
            ses,
            run=run,
            base=t1,
            layers=[Layer("aseg", "aseg", "labels", labels)],
            n_slices=3,
        )

    v = Viewer(ana, expected=[("seg", "01", "01", "01")], render=hook)
    assert [k for k in v.keys] == [("seg", "01", "01", "01")]  # listed before it exists
    first = v.item(("seg", "01", "01", "01"))
    assert first["rendered_now"] and first["dir"] == "seg/sub-01/ses-01/01"
    assert not v.item(("seg", "01", "01", "01"))[
        "rendered_now"
    ]  # fresh: not re-rendered
    assert len(calls) == 2
    with pytest.raises(Exception, match="not in this analysis"):
        v.item(("seg", "02", "01", "01"))
