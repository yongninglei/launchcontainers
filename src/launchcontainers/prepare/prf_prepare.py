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

import csv
import glob
import os
import os.path as op
from datetime import datetime, timedelta

from launchcontainers.log_setup import console
from launchcontainers.utils import parse_hms


# ---------------------------------------------------------------------------
# PRFPrepare
# ---------------------------------------------------------------------------


class PRFPrepare:
    """
    Prepare PRF/GLM mapping information for one subject / session.

    Parameters
    ----------
    lc_config : dict
        Parsed launchcontainers YAML configuration.
    """

    def __init__(self, lc_config: dict):
        self.lc_config = lc_config
        self._general = lc_config["general"]
        self._glm_cfg = lc_config.get("glm_specific", {})

    @property
    def basedir(self) -> str:
        return self._general["basedir"]

    @property
    def bidsdir(self) -> str:
        return op.join(self.basedir, self._general["bidsdir_name"])

    def parse_prf_mat(
        self,
        sub: str,
        ses: str,
        lc_glm: bool = False,
        output_dir: str | None = None,
    ) -> str:
        """
        Parse vistadisplog ``.mat`` files and write a mapping TSV.

        Columns always written:

        * ``log_file_path``  — full path to the ``.mat`` file
        * ``log_file_name``  — basename of the ``.mat`` file
        * ``stim_name``      — basename of the stimulus ``.mat`` (``params.loadMatrix``)
        * ``task_run``       — original task label + per-original-task run counter,
          e.g. ``task-fixRW_run-01``
        * ``acq_time``       — acquisition time parsed from the log filename
          (``HH:MM:SS``)

        Column written only when ``lc_glm=True``:

        * ``glm_task_run``   — normalized task label (``WCnonestop`` / ``WCblock``) +
          per-normalized-task run counter, e.g. ``task-WCnonestop_run-01``

        Output file is named ``sub-{sub}_ses-{ses}_desc-mapping_PRF_acqtime.tsv``
        and written to ``output_dir`` if provided, otherwise to the
        vistadisplog directory
        ``<bidsdir>/sourcedata/vistadisplog/sub-{sub}/ses-{ses}/``.

        Parameters
        ----------
        sub : str
            Subject identifier without the ``sub-`` prefix.
        ses : str
            Session identifier without the ``ses-`` prefix.
        lc_glm : bool
            When ``True``, add the ``glm_task_run`` column.
        output_dir : str or None
            Directory to write the TSV into.

        Returns
        -------
        str
            Path to the written TSV file.

        Raises
        ------
        FileNotFoundError
            If no ``20*.mat`` files are found in the vistadisplog directory.
        """
        from scipy.io import loadmat  # lazy import

        vistadisplog_dir = op.join(
            self.bidsdir,
            "sourcedata",
            "vistadisplog",
            f"sub-{sub}",
            f"ses-{ses}",
        )
        console.print(
            f"\n### Searching vistadisplog dir: {vistadisplog_dir}", style="cyan"
        )

        mat_files = sorted(glob.glob(op.join(vistadisplog_dir, "20*.mat")))
        if not mat_files:
            raise FileNotFoundError(f"No '20*.mat' files found in {vistadisplog_dir}")
        console.print(
            f"Found {len(mat_files)} vistadisplog .mat file(s).", style="cyan"
        )

        if output_dir is not None:
            out_dir = output_dir
        else:
            out_dir = vistadisplog_dir
        os.makedirs(out_dir, exist_ok=True)

        orig_counters: dict[str, int] = {}  # per-original-task run counter
        norm_counters: dict[str, int] = {}  # per-normalized-task run counter

        columns = [
            "log_file_path",
            "log_file_name",
            "stim_name",
            "task_run",
            "acq_time",
        ]
        if lc_glm:
            columns.append("glm_task_run")

        rows = []
        for mat_file in mat_files:
            log_file_name = op.basename(mat_file)
            # The .mat filename records the finish time; subtract 6 min to get start time

            raw_time = datetime.strptime(
                parse_hms(op.splitext(log_file_name)[0]), "%H:%M:%S"
            )
            acq_time = (raw_time - timedelta(minutes=6)).strftime("%H:%M:%S")

            params = loadmat(mat_file, simplify_cells=True)["params"]
            stim_path = params["loadMatrix"]
            stim_name = op.basename(stim_path)

            # Extract original task from stim name
            parts = stim_name.split("_")
            orig_task = parts[1] if len(parts) > 1 else "unknown"

            orig_counters[orig_task] = orig_counters.get(orig_task, 0) + 1
            task_run = f"task-{orig_task}_run-{orig_counters[orig_task]:02d}"

            row = {
                "log_file_path": mat_file,
                "log_file_name": log_file_name,
                "stim_name": stim_name,
                "task_run": task_run,
                "acq_time": acq_time,
            }

            if lc_glm:
                if "fix" not in orig_task:
                    console.print(
                        f"  [WARNING] 'fix' not found in task '{orig_task}' "
                        f"({stim_name}). glm_task_run set to N/A.",
                        style="yellow",
                    )
                    row["glm_task_run"] = "N/A"
                else:
                    norm_task = "WCblock" if "block" in orig_task else "WCnonestop"
                    norm_counters[norm_task] = norm_counters.get(norm_task, 0) + 1
                    row["glm_task_run"] = (
                        f"task-{norm_task}_run-{norm_counters[norm_task]:02d}"
                    )

            rows.append(row)
            glm_info = f"  glm_task_run={row['glm_task_run']}" if lc_glm else ""
            console.print(
                f"  {log_file_name}  stim={stim_name}  task_run={task_run}"
                f"  acq_time={acq_time}{glm_info}",
                style="cyan",
            )

        tsv_file = op.join(out_dir, f"sub-{sub}_ses-{ses}_desc-mapping_PRF_acqtime.tsv")
        with open(tsv_file, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

        console.print(f"\n### Mapping TSV written → {tsv_file}", style="bold red")
        return tsv_file
