# """
# MIT License
# Copyright (c) 2020-2025 Garikoitz Lerma-Usabiaga
# Copyright (c) 2020-2022 Mengxing Liu
# Copyright (c) 2022-2023 Leandro Lecca
# Copyright (c) 2022-2025 Yongning Lei
# Copyright (c) 2023 David Linhardt
# Copyright (c) 2023 Iñigo Tellaetxe
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# """
from __future__ import annotations

from pathlib import Path

from launchcontainers import cli as lc_parser
from launchcontainers.log_setup import console


def gen_subseslist(basedir: str, output_name: str, output_dir=None) -> None:
    """
    Generate a simple ``subseslist`` file from a BIDS-style directory tree.

    Parameters
    ----------
    basedir : str
        Root directory containing ``sub-*`` folders.
    output_name : str
        Output filename stem without the ``.txt`` suffix.
    output_dir : str, optional
        Destination directory for the generated file. If omitted, ``basedir``
        is used.
    """
    base = Path(basedir)
    if not output_dir:
        out_dir = base
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{output_name}.txt"

    # Write header
    with out_file.open("w") as f:
        f.write("sub,ses,RUN,anat,dwi,func\n")

    # Iterate subjects (e.g. sub-01, sub-02, ...)
    for subj_dir in sorted(base.glob("sub-*")):
        if not subj_dir.is_dir():
            continue
        subj = subj_dir.name.split("sub-")[-1]

        # Iterate sessions under each subject
        for ses_dir in sorted(subj_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            ses = ses_dir.name.split("ses-")[-1]

            # Check for modality folders
            anat = (ses_dir / "anat").is_dir()
            dwi = (ses_dir / "dwi").is_dir()
            func = (ses_dir / "func").is_dir()

            # Append row
            with out_file.open("a") as f:
                f.write(f"{subj},{ses},True,{anat},{dwi},{func}\n")


def main():
    """
    Run the ``gen_subses`` helper from the shared launchcontainers parser.
    """
    parser_namespace, parse_dict = lc_parser.get_parser()

    # Check if download_configs argument is provided

    console.print("You are copying configs to target place")
    basedir = parser_namespace.basedir
    output_name = parser_namespace.name
    output_dir = parser_namespace.output_dir
    # Check if download_configs argument is provided
    if basedir:
        gen_subseslist(basedir, output_name, output_dir)


# #%%
if __name__ == "__main__":
    main()
