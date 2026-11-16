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

import os

from launchcontainers import cli as lc_parser
from launchcontainers import utils as do
from launchcontainers.log_setup import console


def copy_example_configs(copy_configs):
    """
    Copy packaged example configuration files into a destination directory.

    Parameters
    ----------
    copy_configs : str
        Destination directory that will receive the example files.
    """
    # Ensure the directory exists
    if not os.path.exists(copy_configs):
        os.makedirs(copy_configs)

    do.copy_configs(copy_configs)
    console.print("\n Step 0; Copy example_configs to the specified place")

    return


def main():
    """
    Run the ``copy_configs`` helper from the shared launchcontainers parser.
    """
    parser_namespace, parse_dict = lc_parser.get_parser()

    # Check if download_configs argument is provided

    print("You are copying configs to target place")
    copy_configs = parser_namespace.output
    # Check if download_configs argument is provided
    if copy_configs:
        copy_example_configs(copy_configs)


# #%%
if __name__ == "__main__":
    main()
