#!/usr/bin/env python3
"""
Create symbolic links from ses-02 to ses-10 pointing to ses-01 for each subject.
"""
import typer
from pathlib import Path
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def create_links(
    base_dir: Path = typer.Argument(
        ...,
        help="Base directory containing sub-01 to sub-11 folders"
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--execute",
        help="Preview links without creating them"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force overwrite existing symlinks and directories"
    )
):
    """
    Create symbolic links from ses-02 to ses-10 pointing to ses-01 for each subject.
    
    Example:
        # Dry run (preview)
        python create_session_links.py /bcbl/home/public/Gari/VOTCLOC/main_exp/BIDS/derivatives/freesurferator/analysis-01
        
        # Execute
        python create_session_links.py /path/to/analysis-01 --execute
        
        # Force overwrite existing links
        python create_session_links.py /path/to/analysis-01 --execute --force
    """
    
    # Validate base directory
    if not base_dir.exists():
        console.print(f"[red]Error: Base directory does not exist: {base_dir}[/red]")
        return
    
    # Convert to absolute path
    base_dir = base_dir.resolve()
    
    console.print(f"[bold]Creating session symbolic links[/bold]")
    console.print(f"  Base directory: {base_dir}")
    console.print(f"  Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    console.print(f"  Force: {force}\n")
    
    # Create links for all subjects and sessions
    created = 0
    overwritten = 0
    skipped = 0
    errors = 0
    
    for sub_num in range(1, 12):  # sub-01 to sub-11
        sub = f"sub-{sub_num:02d}"
        sub_dir = base_dir / sub
        
        # Check if subject directory exists
        if not sub_dir.exists():
            console.print(f"[yellow]⚠ {sub} directory not found, skipping[/yellow]")
            continue
        
        # Check if ses-01 exists (absolute path)
        ses_01_dir = sub_dir / "ses-01"
        if not ses_01_dir.exists():
            console.print(f"[yellow]⚠ {sub}/ses-01 not found, skipping subject[/yellow]")
            continue
        
        console.print(f"[cyan]Processing {sub}...[/cyan]")
        
        for ses_num in range(2, 11):  # ses-02 to ses-10
            ses = f"ses-{ses_num:02d}"
            
            # Link path (absolute path)
            link_path = sub_dir / ses
            
            # Target path (absolute path to ses-01)
            target = ses_01_dir.resolve()
            
            # Check if link already exists
            if link_path.exists() or link_path.is_symlink():
                if not force:
                    # Skip without force
                    if link_path.is_symlink():
                        existing_target = link_path.readlink()
                        # Check if it points to the correct target
                        if existing_target == target or link_path.resolve() == target:
                            console.print(f"  [dim]{ses} -> {target} (already exists, correct)[/dim]")
                            skipped += 1
                            continue
                        else:
                            console.print(f"  [yellow]⚠ {ses} -> {existing_target} (exists, different target, use --force to overwrite)[/yellow]")
                            skipped += 1
                            continue
                    else:
                        console.print(f"  [yellow]⚠ {ses} exists as regular directory (use --force to overwrite)[/yellow]")
                        skipped += 1
                        continue
                else:
                    # Force mode: remove existing
                    if dry_run:
                        if link_path.is_symlink():
                            console.print(f"  [yellow]Would remove existing symlink: {ses}[/yellow]")
                        else:
                            console.print(f"  [yellow]Would remove existing directory: {ses}[/yellow]")
                    else:
                        try:
                            if link_path.is_symlink():
                                link_path.unlink()
                                console.print(f"  [yellow]Removed existing symlink: {ses}[/yellow]")
                            elif link_path.is_dir():
                                import shutil
                                shutil.rmtree(link_path)
                                console.print(f"  [yellow]Removed existing directory: {ses}[/yellow]")
                            else:
                                link_path.unlink()
                                console.print(f"  [yellow]Removed existing file: {ses}[/yellow]")
                            overwritten += 1
                        except Exception as e:
                            console.print(f"  [red]✗ Error removing {ses}: {e}[/red]")
                            errors += 1
                            continue
            
            # Create symbolic link
            if dry_run:
                console.print(f"  [cyan]Would create: {ses} -> {target}[/cyan]")
            else:
                try:
                    link_path.symlink_to(target)
                    console.print(f"  [green]✓ Created: {ses} -> {target}[/green]")
                except Exception as e:
                    console.print(f"  [red]✗ Error creating {ses}: {e}[/red]")
                    errors += 1
                    continue
            
            created += 1
        
        console.print()
    
    # Summary
    console.print(f"[bold]Summary:[/bold]")
    console.print(f"  Links created: {created}")
    if overwritten > 0:
        console.print(f"  Overwritten: {overwritten}")
    console.print(f"  Skipped: {skipped}")
    if errors > 0:
        console.print(f"  [red]Errors: {errors}[/red]")
    
    if dry_run:
        console.print(f"\n[yellow]DRY RUN - No links were actually created[/yellow]")
        console.print(f"[yellow]Use --execute to create the links[/yellow]")


if __name__ == "__main__":
    app()