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

import resource
import subprocess as sp
from concurrent.futures import ProcessPoolExecutor, as_completed

from launchcontainers.log_setup import console


def _parse_mem_bytes(mem_str: str) -> int:
    """Parse a human-readable memory string to bytes.

    Examples: ``'32g'``, ``'32G'``, ``'512m'``, ``'1t'``.
    """
    mem_str = mem_str.strip()
    unit = mem_str[-1].lower()
    value = float(mem_str[:-1])
    multipliers = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    return int(value * multipliers.get(unit, 1))


def _worker_init(mem_bytes: int | None) -> None:
    """Initialiser run once in each worker process.

    Sets the virtual-address-space limit so that each worker process (and the
    apptainer container it spawns as a child) cannot exceed ``mem_bytes`` of
    memory.  Has no effect when ``mem_bytes`` is ``None``.
    """
    if mem_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))


def _run_cmd(cmd: str) -> tuple[int, str]:
    """Run a single shell command via a bash login shell and return (returncode, cmd).

    Using ``bash -l`` ensures that the module system (e.g. Environment Modules
    or Lmod) is initialised, so ``module load apptainer/...`` works even inside
    a subprocess that would otherwise inherit a plain /bin/sh environment.
    """
    rc = sp.run(["bash", "-l", "-c", cmd]).returncode
    return rc, cmd


def launch_serial(cmds: list[str]) -> list[int]:
    """
    Execute a list of shell commands one by one in order.

    Parameters
    ----------
    cmds : list[str]
        Shell commands to run sequentially.

    Returns
    -------
    list[int]
        Return codes in submission order.
    """
    console.print(
        f"Launching {len(cmds)} jobs locally in serial mode",
        style="cyan",
    )
    results = []
    for cmd in cmds:
        rc, _ = _run_cmd(cmd)
        results.append(rc)
        console.print(f"Finished rc={rc} | {cmd[:100]}", style="cyan")
    console.print("All local jobs finished.", style="bold red")
    return results


def launch_parallel(
    cmds: list[str],
    max_workers: int | None = None,
    mem_per_job: str | None = None,
) -> list[int]:
    """
    Execute a list of shell commands in parallel using ProcessPoolExecutor.

    Each worker process is initialised with an optional memory ceiling via
    ``resource.setrlimit``, so the apptainer container it spawns cannot exceed
    that limit.  CPU concurrency is controlled by ``max_workers``; set it to
    ``floor(total_cores / cpus_per_job)`` in the yaml to keep total CPU usage
    within budget.

    Parameters
    ----------
    cmds : list[str]
        Shell commands to run in parallel.
    max_workers : int or None
        Maximum number of parallel workers. ``None`` uses the number of CPUs.
    mem_per_job : str or None
        Memory limit per worker, e.g. ``'32g'``. ``None`` means no limit.

    Returns
    -------
    list[int]
        Return codes in completion order.
    """
    mem_bytes = _parse_mem_bytes(mem_per_job) if mem_per_job else None
    console.print(
        f"Launching {len(cmds)} jobs locally in parallel mode "
        f"with max_workers={max_workers}, mem_per_job={mem_per_job}",
        style="cyan",
    )
    results = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(mem_bytes,),
    ) as executor:
        futures = {executor.submit(_run_cmd, cmd): cmd for cmd in cmds}
        for future in as_completed(futures):
            rc, cmd = future.result()
            results.append(rc)
            console.print(f"Finished rc={rc} | {cmd[:100]}", style="cyan")
    console.print("All local jobs finished.", style="bold red")
    return results
