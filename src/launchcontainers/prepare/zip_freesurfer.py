#!/usr/bin/env python3
"""
Zip FreeSurfer outputs from freesurferator-with_t2 into anatrois-compatible format.
Creates freesurferator_Sxxx.zip in sub-xx/ses-01/output/ with Sxxx/ as root directory.
"""

from pathlib import Path
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from launchcontainers.log_setup import console

FREESURFERATOR_BASE = Path("/scratch/tlei/VOTCLOC/BIDS/derivatives/freesurfer-with_t2")
OUTPUT_BASE = Path(
    "/scratch/tlei/VOTCLOC/BIDS/derivatives/freesurfer-with_t2/analysis-prefs"
)
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 12)]


def zip_subject(subject: str) -> tuple[str, bool, str]:
    """Zip FreeSurfer output for one subject."""

    s_id = "Sxxx"

    source_dir = FREESURFERATOR_BASE / subject
    output_dir = OUTPUT_BASE / subject / "ses-01" / "output"
    zip_file = output_dir / f"freesurferator_{s_id}.zip"

    if not source_dir.exists():
        return subject, False, f"Source not found: {source_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temp directory with Sxxx structure
    temp_dir = output_dir / f"temp_{s_id}"
    temp_target = temp_dir / s_id

    try:
        # Clean up any existing temp
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        # Create Sxxx directory and copy contents
        temp_target.mkdir(parents=True)

        # Copy all contents from source to Sxxx/
        for item in source_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, temp_target / item.name, symlinks=True)
            else:
                shutil.copy2(item, temp_target / item.name)

        # Zip from temp directory
        subprocess.run(
            ["zip", "-r", "-q", str(zip_file), s_id],
            cwd=temp_dir,
            check=True,
            capture_output=True,
        )

        # Clean up
        shutil.rmtree(temp_dir)

        return subject, True, f"Created {zip_file}"

    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return subject, False, f"Error: {str(e)}"


def main():
    console.print("[bold cyan]FreeSurfer Output Zipper[/bold cyan]")
    console.print(f"Processing {len(SUBJECTS)} subjects\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Zipping...", total=len(SUBJECTS))

        success_count = 0
        failed = []

        with ProcessPoolExecutor(max_workers=11) as executor:
            futures = {executor.submit(zip_subject, subj): subj for subj in SUBJECTS}

            for future in as_completed(futures):
                subject, success, message = future.result()

                if success:
                    success_count += 1
                    console.print(f"[green]✓[/green] {subject}: {message}")
                else:
                    failed.append((subject, message))
                    console.print(f"[red]✗[/red] {subject}: {message}")

                progress.advance(task)

    console.print(f"\n[bold]Summary: {success_count}/{len(SUBJECTS)} successful[/bold]")

    if failed:
        console.print("\n[bold red]Failed:[/bold red]")
        for subj, msg in failed:
            console.print(f"  {subj}: {msg}")


if __name__ == "__main__":
    main()
