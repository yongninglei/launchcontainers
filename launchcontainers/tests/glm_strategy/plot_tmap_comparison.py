#!/usr/bin/env python
"""Plot a mosaic of t-maps across acq/strategy combos for a single contrast.

Example:
    python plot_tmap_comparison.py \
        --base /bcbl/home/public/Gari/IRAKEINU \
        -s pilot02,01 \
        --analysis-name T1w_MESE_compare_basic_MC \
        --task BfLocVideo \
        --contrast CH_RWvsCH_SC \
        --threshold 3.1
"""

from pathlib import Path

import typer
from rich.console import Console
from nilearn import plotting, image

app = typer.Typer(add_completion=False)
console = Console()

# (label, acq, desc) — desc="" means no desc token in the filename
COMBOS = [
    ("ME / denoised", "ME", "denoised"),
    ("ME / optcom", "ME", "optcom"),
    ("ME / echo2", "ME", "echo2"),
    ("SE", "SE", ""),
]


def _tmap_path(l1_dir: Path, sub: str, ses: str, task: str, acq: str, desc: str, contrast: str) -> Path:
    desc_tok = f"_desc-{desc}" if desc else ""
    fname = (
        f"sub-{sub}_ses-{ses}_task-{task}_acq-{acq}_space-T1w{desc_tok}"
        f"_contrast-{contrast}_stat-t_statmap.nii.gz"
    )
    return l1_dir / fname


@app.command()
def main(
    base: str = typer.Option(..., "--base", help="Project base dir, e.g. /bcbl/home/public/Gari/IRAKEINU"),
    subses: str = typer.Option(..., "-s", "--subses", help="sub,ses  e.g. pilot02,01"),
    analysis_name: str = typer.Option(..., "--analysis-name", help="L1 analysis name, e.g. T1w_MESE_compare_basic_MC"),
    task: str = typer.Option("BfLocVideo", "--task"),
    contrast: str = typer.Option("CH_RWvsCH_SC", "--contrast", help="Contrast name to plot"),
    threshold: float = typer.Option(3.1, "--threshold", help="t-value threshold; sub-threshold voxels shown transparent"),
    fp_ana_name: str = typer.Option("25.1.4_IRpilot", "--fp-ana-name", help="fMRIprep derivatives folder suffix (for anat bg)"),
    display_mode: str = typer.Option("mosaic", "--display-mode", help="nilearn display_mode, e.g. mosaic | z | x | y"),
    cmap: str = typer.Option("cold_hot", "--cmap"),
    output: str = typer.Option("", "--output", help="Output png path (default: alongside L1 dir)"),
):
    sub, ses = [x.strip() for x in subses.split(",")]

    l1_dir = Path(base) / "derivatives" / "l1_surface" / f"analysis-{analysis_name}" / f"sub-{sub}" / f"ses-{ses}"
    anat_img = (
        Path(base) / "derivatives" / f"fmriprep-{fp_ana_name}" / f"sub-{sub}" / f"ses-{ses}" / "anat"
        / f"sub-{sub}_ses-{ses}_run-01_desc-preproc_T1w.nii.gz"
    )

    if not anat_img.exists():
        console.print(f"[red]Error:[/red] anat background not found: {anat_img}")
        raise typer.Exit(code=1)

    import matplotlib.pyplot as plt
    from PIL import Image
    import tempfile

    if output:
        out_path = Path(output)
    else:
        out_path = l1_dir / f"tmap_comparison_{task}_{contrast}_thr-{threshold}.png"

    panel_pngs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for label, acq, desc in COMBOS:
            tmap_path = _tmap_path(l1_dir, sub, ses, task, acq, desc, contrast)
            if not tmap_path.exists():
                console.print(f"[yellow]Skipping[/yellow] {label}: not found ({tmap_path.name})")
                continue

            console.print(f"[green]Plotting[/green] {label}: {tmap_path.name}")

            fig = plt.figure(figsize=(16, 4))
            plotting.plot_stat_map(
                str(tmap_path),
                bg_img=str(anat_img),
                display_mode=display_mode,
                threshold=threshold,
                cmap=cmap,
                colorbar=True,
                title=f"sub-{sub} ses-{ses}  {label}  contrast-{contrast}  (t > {threshold})",
                figure=fig,
            )
            panel_path = Path(tmpdir) / f"{label.replace(' / ', '-').replace(' ', '')}.png"
            fig.savefig(panel_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            panel_pngs.append(panel_path)

        if not panel_pngs:
            console.print("[red]Error:[/red] no t-maps found, nothing to plot")
            raise typer.Exit(code=1)

        panels = [Image.open(p) for p in panel_pngs]
        width = max(p.width for p in panels)
        height = sum(p.height for p in panels)
        composite = Image.new("RGB", (width, height), "white")
        y = 0
        for p in panels:
            composite.paste(p, (0, y))
            y += p.height

        composite.save(out_path)

    console.print(f"[bold green]Saved:[/bold green] {out_path}")


if __name__ == "__main__":
    app()
