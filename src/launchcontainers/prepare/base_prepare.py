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

import os.path as op

from launchcontainers.log_setup import console


class BasePrepare:
    """
    Base class for all launchcontainers prepare pipelines.

    Provides shared infrastructure that every ``XxxPrepare`` subclass inherits:

    * Config parsing — ``lc_config`` dict split into ``_general`` and a
      pipeline-specific sub-dict.
    * Common path properties — :attr:`basedir` and :attr:`bidsdir`.
    * Example-config generation — :meth:`write_example_config` writes a
      ready-to-use ``lc_config_example.yaml``; subclasses supply the content
      via :meth:`_example_config_dict`.

    Parameters
    ----------
    lc_config : dict or None
        Parsed launchcontainers YAML configuration.  Pass ``None`` (or omit)
        only when calling :meth:`write_example_config` without a real config.
    """

    def __init__(self, lc_config: dict | None = None):
        self.lc_config = lc_config or {}
        self._general = self.lc_config.get("general", {})

    # ------------------------------------------------------------------
    # Common path properties
    # ------------------------------------------------------------------

    @property
    def basedir(self) -> str:
        """Absolute path to the project root (``general.basedir``)."""
        return self._general["basedir"]

    @property
    def bidsdir(self) -> str:
        """Absolute path to the BIDS directory (``<basedir>/<bidsdir_name>``)."""
        return op.join(self.basedir, self._general["bidsdir_name"])

    # ------------------------------------------------------------------
    # Example config generator
    # ------------------------------------------------------------------

    @classmethod
    def _example_config_dict(cls) -> dict:
        """
        Return the example config dictionary for this pipeline.

        Subclasses **must** override this method and return a plain Python
        ``dict`` that represents a fully annotated ``lc_config.yaml`` for
        their pipeline.  The base implementation raises
        :exc:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _example_config_dict()."
        )

    @classmethod
    def write_example_config(cls, output_path: str = "lc_config_example.yaml") -> str:
        """
        Write an annotated example ``lc_config.yaml`` for this pipeline.

        Calls :meth:`_example_config_dict` to obtain the content, then
        serialises it as YAML.  The destination file is created (or
        overwritten) at *output_path*.

        Parameters
        ----------
        output_path : str
            Destination file path.  Defaults to ``lc_config_example.yaml``
            in the current working directory.

        Returns
        -------
        str
            Absolute path to the written file.
        """
        import yaml  # lazy import — not needed at module level

        example = cls._example_config_dict()
        out_path = op.abspath(output_path)
        with open(out_path, "w") as fh:
            yaml.dump(
                example,
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        console.print(f"\n### Example config written → {out_path}", style="bold green")
        return out_path
