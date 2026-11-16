"""Pre-render one QC item to PNG slices (ARCHITECTURE.md §6, layer 2).

Ported from QC-registration's ``render.py`` and generalised: an *item* is a
grayscale **base** image plus any number of **layers** drawn on top of it:

- ``map``      continuous image, colormap + opacity (e.g. b0 on T1, a qMRI map)
- ``labels``   label volume, one colour per label (e.g. aparc+aseg)
- ``outline``  the borders of a label volume / mask, as thin lines
- ``surface``  FreeSurfer/GIfTI surfaces cut by each slice (e.g. white matter)

Per view (coronal, sagittal, axial) and slice, one PNG per layer is written:
``<view>-<i>_base.png`` and ``<view>-<i>_<layer key>.png`` — the viewer
stacks them, so toggling a layer costs nothing. ``preview.png`` (every layer
flattened, middle slice of each view) is for the static report.

Output folder (``item_dir``): ``<analysis>/qc/<item>/sub-XX/ses-YY[/<run>]/``
with ``session.json``: item, sub, ses, run, question, layers, slice positions
in mm, the PNG list and a **fingerprint** of the inputs (path, size, mtime)
and settings. A call whose fingerprint matches and whose PNGs are all there
does nothing, so nothing is re-rendered without a reason. ``session.json`` is
written last: a crash mid-way leaves an item that is not fresh.

Needs numpy, nibabel, scipy, matplotlib and pillow, imported only inside the
functions: core's runtime dependencies stay pyyaml/rich/typer, and the tool
that renders brings them (or ``pip install "launchcontainers[render]"``).
Slice positions follow fMRIPrep's report (niworkflows ``cuts_from_bbox``)
on the base image's brain mask (given, or a threshold of the base).
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from launchcontainers.core.io import PathLike, atomic_write_text, bare
from launchcontainers.core.ratings import QC_DIR

VIEWS = ("coronal", "sagittal", "axial")
VIEW_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
CROP_MARGIN_MM = 15.0
LAYER_KINDS = ("map", "labels", "outline", "surface")
# Bump when a code change alters the pixels: every item's fingerprint changes.
RENDER_VERSION = 1

ImageSource = Union[
    PathLike, Callable[[], Any]
]  # a path, or a function returning a nibabel image


@dataclass
class Layer:
    key: str  # short id, used in PNG names and the page
    label: str  # what the rater sees
    kind: str  # map | labels | outline | surface
    source: Any = None  # ImageSource; for "surface": a list of surface paths
    cmap: str = "viridis"
    opacity: float = 0.55
    color: str = "#ffd400"  # outline / surface colour
    show: bool = True  # on when the page opens
    percentiles: tuple[float, float] = (2.0, 98.0)
    background: float | None = 0.0  # map: this value is transparent (None: nothing is)
    interpolation: str = (
        "linear"  # map: linear | nearest when resampled to the base grid
    )

    def __post_init__(self) -> None:
        if self.kind not in LAYER_KINDS:
            raise ValueError(
                f"Layer.kind must be one of {LAYER_KINDS}, got {self.kind!r}"
            )
        if (
            not self.key.replace("-", "").replace("_", "").isalnum()
            or self.key == "base"
        ):
            raise ValueError(
                f"Layer.key must be alphanumeric and not 'base': {self.key!r}"
            )
        if self.interpolation not in ("linear", "nearest"):
            raise ValueError(
                f"Layer.interpolation must be linear or nearest, got {self.interpolation!r}"
            )


@dataclass
class RenderResult:
    out: Path
    rendered: bool  # False: was fresh, nothing written
    files: list[str] = field(default_factory=list)
    preview: Path | None = None
    notes: list[str] = field(default_factory=list)


def item_dir(
    analysis_dir: PathLike, item: str, sub: str, ses: str, run: str = ""
) -> Path:
    d = (
        Path(analysis_dir)
        / QC_DIR
        / item
        / f"sub-{bare(sub, 'sub')}"
        / f"ses-{bare(ses, 'ses')}"
    )
    return d / run if run else d


# -- provenance --------------------------------------------------------------------
def _stat(path: PathLike) -> dict[str, Any]:
    p = Path(path)
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def fingerprint(
    inputs: Mapping[str, PathLike], settings: Mapping[str, Any]
) -> tuple[str, dict]:
    """sha1 of the inputs' (path, size, mtime) and the settings; no image is read."""
    stats = {k: _stat(v) for k, v in sorted(inputs.items())}
    blob = json.dumps(
        {"inputs": stats, "settings": settings, "render_version": RENDER_VERSION},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(blob.encode()).hexdigest(), stats


def read_meta(out: PathLike) -> dict[str, Any] | None:
    try:
        return json.loads((Path(out) / "session.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def is_fresh(out: PathLike, fp: str) -> bool:
    meta = read_meta(out)
    return bool(
        meta
        and meta.get("fingerprint") == fp
        and all((Path(out) / f).is_file() for f in meta.get("files", []))
    )


# -- images ------------------------------------------------------------------------
def _load(src: ImageSource, what: str):
    import nibabel as nib
    import numpy as np

    img = src() if callable(src) else nib.load(str(src))
    img = nib.as_closest_canonical(img)
    if img.ndim == 4 and img.shape[3] == 1:
        img = img.slicer[..., 0]
    if img.ndim != 3:
        raise ValueError(f"{what}: expected a 3-D image, got shape {img.shape}")
    return nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, img.header)


def _on_grid(img, ref, order: int):
    import numpy as np
    from nibabel.processing import resample_from_to

    if img.shape == ref.shape and np.allclose(img.affine, ref.affine, atol=1e-3):
        return img, False
    return resample_from_to(img, ref, order=order, mode="constant", cval=0), True


def robust_range(
    data, lo: float = 2.0, hi: float = 98.0, background: float | None = 0.0
) -> tuple[float, float]:
    import numpy as np

    vals = data[np.isfinite(data)]
    if background is not None:
        vals = vals[vals != background]
    if vals.size == 0:
        return 0.0, 1.0
    a, b = (float(x) for x in np.percentile(vals, [lo, hi]))
    return (a, b) if b > a else (a, a + 1.0)


def brain_mask(data):
    import numpy as np

    pos = data[np.isfinite(data) & (data > 0)]
    if pos.size == 0:
        return np.ones(data.shape, bool)
    return data > 0.2 * float(np.percentile(pos, 99))


def cuts_from_mask(mask, n: int) -> dict[str, list[int]]:
    """niworkflows ``cuts_from_bbox`` in voxel indices of a canonical grid."""
    import numpy as np

    counts = (mask.sum(2).sum(1), mask.sum(2).sum(0), mask.sum(1).sum(0))
    thresholds = np.ceil(
        [
            mask.shape[1] * mask.shape[2] * 0.2,
            mask.shape[0] * mask.shape[2] * 0.1,
            mask.shape[0] * mask.shape[1] * 0.3,
        ]
    ).astype(int)
    out: dict[str, list[int]] = {}
    for view, axis in VIEW_AXIS.items():
        lo, hi = 0, mask.shape[axis] - 1
        hits = np.argwhere(counts[axis] > thresholds[axis])
        if hits.size < n:
            hits = np.argwhere(counts[axis] > 0)
        if hits.size:
            lo, hi = int(hits.min()), int(hits.max())
        out[view] = [int(round(v)) for v in np.linspace(lo, hi, n + 2)[1:-1]]
    return out


def crop_box(mask, zooms) -> list[tuple[int, int]]:
    import numpy as np

    box = []
    for axis in range(3):
        hits = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
        lo, hi = (
            (int(hits.min()), int(hits.max()))
            if hits.size
            else (0, mask.shape[axis] - 1)
        )
        pad = int(np.ceil(CROP_MARGIN_MM / float(zooms[axis])))
        box.append((max(lo - pad, 0), min(hi + pad, mask.shape[axis] - 1)))
    return box


def _plane(data, view: str, index: int, box):
    import numpy as np

    axis = VIEW_AXIS[view]
    rest = [a for a in range(3) if a != axis]
    plane = np.take(data, index, axis=axis)
    (r0, r1), (c0, c1) = box[rest[0]], box[rest[1]]
    return np.flipud(plane[r0 : r1 + 1, c0 : c1 + 1].T)


def _resize(img, view: str, zooms, scale: int = 1):
    """Physical aspect for anisotropic voxels, times ``scale``."""
    from PIL import Image

    rest = [a for a in range(3) if a != VIEW_AXIS[view]]
    sx, sy = float(zooms[rest[0]]), float(zooms[rest[1]])
    base = min(sx, sy)
    size = (round(img.width * sx / base) * scale, round(img.height * sy / base) * scale)
    if size == img.size:
        return img
    return img.resize(size, Image.NEAREST if img.mode == "RGBA" else Image.BILINEAR)


def _gray(plane, vrange):
    import numpy as np
    from PIL import Image

    lo, hi = vrange
    scaled = np.clip((np.nan_to_num(plane) - lo) / (hi - lo), 0, 1)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L")


def _map(plane, vrange, cmap: str, opacity: float, background: float | None = 0.0):
    import numpy as np
    from matplotlib import colormaps
    from PIL import Image

    lo, hi = vrange
    valid = np.isfinite(plane)
    if background is not None:
        valid &= plane != background
    rgba = colormaps[cmap](np.clip((np.nan_to_num(plane) - lo) / (hi - lo), 0, 1))
    rgba[..., 3] = np.where(valid, opacity, 0.0)
    return Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")


def _labels(plane, opacity: float):
    import numpy as np
    from matplotlib import colormaps
    from PIL import Image

    lab = np.nan_to_num(plane).astype(np.int64)
    rgba = colormaps["tab20"](
        (lab * 7) % 20
    )  # spread neighbouring label ids over the palette
    rgba[..., 3] = np.where(lab > 0, opacity, 0.0)
    return Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")


def _outline(plane, color: str):
    import numpy as np
    from matplotlib.colors import to_rgba
    from PIL import Image

    lab = np.nan_to_num(plane).astype(np.int64)
    edge = np.zeros(lab.shape, bool)
    edge[:-1, :] |= lab[:-1, :] != lab[1:, :]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    rgba = np.zeros((*lab.shape, 4))
    rgba[edge] = to_rgba(color)
    return Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")


def _segments(vox, faces, axis: int, index: int):
    """Intersection of a triangle mesh with the plane ``vox[axis] == index``: (n, 2, 3)."""
    import numpy as np

    d = vox[:, axis] - (index + 1e-4)
    fd = d[faces]
    hit = (fd.min(axis=1) < 0) & (fd.max(axis=1) > 0)
    tri, fd = faces[hit], fd[hit]
    if not len(tri):
        return np.empty((0, 2, 3))
    points, valid = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = fd[:, a], fd[:, b]
        crosses = (da * db) < 0
        t = np.where(crosses, da / np.where(crosses, da - db, 1.0), 0.0)
        pa, pb = vox[tri[:, a]], vox[tri[:, b]]
        points.append(pa + t[:, None] * (pb - pa))
        valid.append(crosses)
    points, valid = np.stack(points, 1), np.stack(valid, 1)
    order = np.argsort(~valid, axis=1, kind="stable")[:, :2]
    return np.take_along_axis(points, order[:, :, None], axis=1)


def _surface(meshes, view: str, index: int, box, size, color: str, scale: int = 2):
    from PIL import Image, ImageDraw

    axis = VIEW_AXIS[view]
    rest = [a for a in range(3) if a != axis]
    (c0, c1), (r0, r1) = box[rest[0]], box[rest[1]]
    width, height = size[0] * scale, size[1] * scale
    sx, sy = width / (c1 - c0 + 1), height / (r1 - r0 + 1)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for vox, faces in meshes:
        seg = _segments(vox, faces, axis, index)
        x = (seg[:, :, rest[0]] - c0 + 0.5) * sx
        y = (r1 - seg[:, :, rest[1]] + 0.5) * sy
        for (x0, x1), (y0, y1) in zip(x, y):
            draw.line((x0, y0, x1, y1), fill=color, width=scale)
    return img


def _meshes(paths: Sequence[PathLike], ref):
    import nibabel as nib
    import numpy as np

    to_vox = np.linalg.inv(ref.affine)
    out = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".gii":
            coords, faces = nib.load(str(p)).agg_data(("pointset", "triangle"))
        else:  # FreeSurfer surface (lh.white): tkr RAS → scanner RAS via the volume's c_ras
            coords, faces, meta = nib.freesurfer.read_geometry(
                str(p), read_metadata=True
            )
            coords = coords + np.asarray(meta.get("cras", (0, 0, 0)))
        out.append(
            (nib.affines.apply_affine(to_vox, coords), np.asarray(faces, np.int64))
        )
    return out


# -- the unit ----------------------------------------------------------------------
def render_item(
    analysis_dir: PathLike,
    item: str,
    sub: str,
    ses: str,
    *,
    base: ImageSource,
    layers: Sequence[Layer] = (),
    run: str = "",
    inputs: Mapping[str, PathLike] | None = None,
    mask: ImageSource | None = None,
    question: str = "",
    base_label: str = "base",
    n_slices: int = 7,
    base_percentiles: tuple[float, float] = (2.0, 98.0),
    settings: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> RenderResult:
    """Render one item of one session (and run) unless its PNGs are fresh. No printing.

    ``inputs`` names every file the pixels depend on (it is what the
    fingerprint hashes); by default the path sources of base, mask and layers.
    """
    out = item_dir(analysis_dir, item, sub, ses, run)
    if inputs is None:
        inputs = {}
        for k, src in (
            ("base", base),
            ("mask", mask),
            *((ly.key, ly.source) for ly in layers),
        ):
            if src is None or callable(src):
                continue
            if isinstance(src, (list, tuple)):
                inputs.update({f"{k}_{i}": s for i, s in enumerate(src)})
            else:
                inputs[k] = src
    layer_settings = [
        {
            k: getattr(ly, k)
            for k in (
                "key",
                "kind",
                "cmap",
                "opacity",
                "color",
                "percentiles",
                "background",
                "interpolation",
            )
        }
        for ly in layers
    ]
    all_settings = {
        "n_slices": n_slices,
        "base_percentiles": list(base_percentiles),
        "layers": layer_settings,
        **(settings or {}),
    }
    fp, stats = fingerprint(inputs, all_settings)
    res = RenderResult(out, rendered=False)
    if not overwrite and is_fresh(out, fp):
        res.files = (read_meta(out) or {}).get("files", [])
        res.preview = out / "preview.png" if (out / "preview.png").is_file() else None
        return res

    import numpy as np
    from PIL import Image

    ref = _load(base, "base")
    data = ref.get_fdata(dtype=np.float32)
    zooms = ref.header.get_zooms()[:3]
    if mask is not None:
        m, _ = _on_grid(_load(mask, "mask"), ref, order=0)
        mask_arr = np.asanyarray(m.dataobj) > 0
    else:
        mask_arr = brain_mask(data)
    cuts = cuts_from_mask(mask_arr, n_slices)
    box = crop_box(mask_arr, zooms)
    base_range = robust_range(data, *base_percentiles)

    volumes: dict[str, Any] = {}
    ranges: dict[str, tuple[float, float]] = {}
    meshes: dict[str, Any] = {}
    resampled = []
    for ly in layers:
        if ly.kind == "surface":
            meshes[ly.key] = _meshes(ly.source, ref)
            continue
        order = 1 if ly.kind == "map" and ly.interpolation == "linear" else 0
        img, moved = _on_grid(_load(ly.source, ly.label), ref, order=order)
        if moved:
            resampled.append(ly.key)
        volumes[ly.key] = img.get_fdata(dtype=np.float32)
        if ly.kind == "map":
            ranges[ly.key] = robust_range(
                volumes[ly.key], *ly.percentiles, background=ly.background
            )

    # Nothing is deleted (core rule): PNGs are overwritten in place; session.json
    # lists the current ones, so a leftover from an earlier, larger render is ignored.
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    preview_rows = []
    for view in VIEWS:
        for i, index in enumerate(cuts[view]):
            tile = _resize(
                _gray(_plane(data, view, index, box), base_range), view, zooms
            )
            stack = [("base", tile)]
            for ly in layers:
                if ly.kind == "surface":
                    img = _surface(
                        meshes[ly.key], view, index, box, tile.size, ly.color
                    )
                else:
                    plane = _plane(volumes[ly.key], view, index, box)
                    if ly.kind == "map":
                        img = _resize(
                            _map(
                                plane,
                                ranges[ly.key],
                                ly.cmap,
                                ly.opacity,
                                ly.background,
                            ),
                            view,
                            zooms,
                        )
                    elif ly.kind == "labels":
                        img = _resize(_labels(plane, ly.opacity), view, zooms)
                    else:
                        img = _resize(_outline(plane, ly.color), view, zooms)
                stack.append((ly.key, img))
            for key, img in stack:
                name = f"{view}-{i}_{key}.png"
                img.save(out / name, optimize=False, compress_level=1)
                files.append(name)
            if i == len(cuts[view]) // 2:
                flat = tile.convert("RGBA")
                for _key, img in stack[1:]:
                    flat.alpha_composite(
                        img.resize(flat.size, Image.BILINEAR)
                        if img.size != flat.size
                        else img
                    )
                preview_rows.append(flat)
    width = sum(t.width for t in preview_rows)
    height = max(t.height for t in preview_rows)
    preview = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    x = 0
    for t in preview_rows:
        preview.paste(t, (x, (height - t.height) // 2))
        x += t.width
    preview.convert("RGB").save(out / "preview.png")
    files.append("preview.png")

    aff = ref.affine
    cuts_mm = {
        v: [
            round(float(aff[VIEW_AXIS[v], VIEW_AXIS[v]] * c + aff[VIEW_AXIS[v], 3]), 1)
            for c in cuts[v]
        ]
        for v in VIEWS
    }
    meta = {
        "item": item,
        "sub": bare(sub, "sub"),
        "ses": bare(ses, "ses"),
        "run": run,
        "question": question,
        "layers": [{"key": "base", "label": base_label, "show": True}]
        + [{"key": ly.key, "label": ly.label, "show": ly.show} for ly in layers],
        "cuts_vox": cuts,
        "cuts_mm": cuts_mm,
        "resampled": resampled,
        "rendered": datetime.datetime.now().isoformat(timespec="seconds"),
        "render_version": RENDER_VERSION,
        "fingerprint": fp,
        "inputs": stats,
        "settings": all_settings,
        "files": files,
    }
    atomic_write_text(out / "session.json", json.dumps(meta, indent=1, default=str))
    res.rendered, res.files, res.preview = True, files, out / "preview.png"
    if resampled:
        res.notes.append(
            f"resampled to the base grid for display: {', '.join(resampled)}"
        )
    return res


def item_key(meta: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(meta["item"]),
        bare(str(meta["sub"]), "sub"),
        bare(str(meta["ses"]), "ses"),
        str(meta.get("run") or ""),
    )


def find_items(analysis_dir: PathLike) -> list[dict[str, Any]]:
    """Every rendered item under ``<analysis>/qc/``: the contents of its session.json + ``dir``."""
    qc = Path(analysis_dir) / QC_DIR
    found = []
    for meta_path in sorted(qc.glob("*/sub-*/ses-*/session.json")) + sorted(
        qc.glob("*/sub-*/ses-*/*/session.json")
    ):
        meta = read_meta(meta_path.parent)
        if meta and meta.get("item"):
            meta["dir"] = meta_path.parent.relative_to(qc).as_posix()
            found.append(meta)
    return sorted(
        found, key=lambda m: (m["item"], m["sub"], m["ses"], m.get("run", ""))
    )
