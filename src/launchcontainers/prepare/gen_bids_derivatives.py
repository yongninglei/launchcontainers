"""
Already use heudiconv to generate empty dataset_descriptions

Maybe it's time to use this to write and note down the corresponding containers

This one is WIP
"""

from __future__ import annotations

import json
import os

import pandas as pd
from launchcontainers.log_setup import console


def gen_dataset_desc_json(output_dir):
    """
    Create a placeholder ``dataset_description.json`` file.

    Parameters
    ----------
    output_dir : str or path-like
        Directory where the JSON file should be written.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    dataset_description = {
        "Acknowledgements": "TODO: whom you want to acknowledge",
        "Authors": ["TODO:", "First1 Last1", "First2 Last2", "..."],
        "BIDSVersion": "1.10.0",
        "DatasetDOI": "TODO: eventually a DOI for the dataset",
        "Funding": ["TODO", "GRANT #1", "GRANT #2"],
        "HowToAcknowledge": "TODO: describe how to acknowledge -- either cite a corresponding paper, \
        or just in acknowledgement section",
        "License": "TODO: choose a license, e.g. PDDL (http://opendatacommons.org/licenses/pddl/)",
        "Name": "TODO: name of the dataset",
        "ReferencesAndLinks": ["TODO", "List of papers or websites"],
    }

    output_path = os.path.join(output_dir, "dataset_description.json")
    with open(output_path, "w") as f:
        json.dump(dataset_description, f, indent=4)

    console.print(f"dataset_description.json created at: {output_path}", style="cyan")


def gen_participant_json(output_dir):
    """
    Create a placeholder ``participant.json`` sidecar file.

    Parameters
    ----------
    output_dir : str or path-like
        Directory where the JSON file should be written.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    participant = {
        "participant_id": {
            "Description": "Participant identifier",
        },
        "age": {
            "Description": "Age in years (TODO - verify) as in the initial session, \
                    might not be correct for other sessions",
        },
        "sex": {
            "Description": "self-rated by participant, M for male/F for female (TODO: verify)",
        },
        "group": {
            "Description": "(TODO: adjust - by default everyone is in control group)",
        },
    }

    output_path = os.path.join(output_dir, "participant.json")
    with open(output_path, "w") as f:
        json.dump(participant, f, indent=4)

    console.print(f"participant.json created at: {output_path}", style="cyan")


def gen_readme(output_dir, content=""):
    """
    Create a README file for a derivative dataset directory.

    Parameters
    ----------
    output_dir : str or path-like
        Directory where the README file should be written.
    content : str, default=''
        README body text to write.
    """

    os.makedirs(output_dir, exist_ok=True)
    readme_path = os.path.join(output_dir, "README")

    with open(readme_path, "w") as f:
        f.write(content)

    console.print(f"Empty README created at: {readme_path}", style="cyan")


def gen_participant_tsv(output_dir, columns):
    """
    Create an empty ``participants.tsv`` file with the requested columns.

    Parameters
    ----------
    output_dir : str or path-like
        Directory where the TSV file should be written.
    columns : list[str] | None
        Column names for the TSV file. If ``None``, a default participant
        schema is used.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    if columns is None:
        columns = ["participant_id", "age", "sex", "group"]

    participants_path = os.path.join(output_dir, "participants.tsv")

    df = pd.DataFrame(columns=columns)
    df.to_csv(participants_path, sep="\t", index=False)

    console.print(f"participants.tsv created at: {participants_path}", style="cyan")
