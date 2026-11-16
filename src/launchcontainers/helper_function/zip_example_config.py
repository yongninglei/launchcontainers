from __future__ import annotations

import os
import os.path as op
import subprocess
import zipfile
from importlib.metadata import version
from pathlib import Path


def get_git_root():
    """
    Return the repository root detected from Git.

    Returns
    -------
    pathlib.Path | None
        Repository root path, or ``None`` if the command is not run inside a
        Git working tree.
    """
    try:
        git_root = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
            )
            .strip()
            .decode("utf-8")
        )
        return Path(git_root)
    except subprocess.CalledProcessError:
        return None


def do_zip_configs():
    """
    Package the repository example configuration directory as a zip archive.
    """
    # Paths
    repo_dir = get_git_root()
    package_version = version("launchcontainers")
    example_configs_dir = op.join(repo_dir, "example_configs")  # type: ignore
    output_zip = op.join(
        repo_dir,
        "launchcontainers",
        "example_configs",  # type: ignore
        f"example_configs_{package_version}.zip",
    )

    # Create the src/configs directory if it doesn't exist
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)

    # Zip the folder
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(example_configs_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, example_configs_dir)
                zipf.write(filepath, arcname)
    print("successfully zip your all your example configs into the package")


if __name__ == "__main__":
    do_zip_configs()
