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
"""
Shared Rich console, color-scheme constants, and verbosity control for launchcontainers.

All modules import ``console`` from here so that the single instance can be
redirected to a log file once (in ``cli.py``) and every ``console.print()``
call across the package ends up in that file automatically.

Verbosity levels
----------------
Call ``setup_verbosity()`` once at CLI startup to configure what gets printed:

- **default**  : INFO, WARNING, ERROR, CRITICAL are shown
- ``--quiet``  : only ERROR and CRITICAL are shown
- ``--verbose``: same as default (INFO and above)
- ``--debug``  : everything including DEBUG messages

Use the helper functions instead of ``console.print()`` directly so that the
verbosity level is respected automatically:

.. code-block:: python

    from launchcontainers.log_setup import log_debug, log_info, log_warning, log_error, log_critical

    log_info("Starting prepare phase")
    log_warning("Config key missing, using default")
    log_error("File not found")
    log_critical("Launching now")
    log_debug("Internal state: x = 42")

Color scheme (mirrors color_codes.txt)
---------------------------------------
- INFO / DEBUG  ``"cyan"``       General information messages
- WARNING       ``"yellow"``     Warning messages
- ERROR         ``"red"``        Error messages
- CRITICAL      ``"bold red"``   Critical messages and important notifications
                 (plain console.print with "bold red" is logged as INFO)
- BLUE_INFO     ``"blue"``       Specific info (e.g. container layout)
"""

from __future__ import annotations

import logging

from rich.console import Console

# ---------------------------------------------------------------------------
# Single shared console — import this everywhere instead of Console()
# ---------------------------------------------------------------------------


class _LoggingConsole(Console):
    """Console that also forwards every print() call to the Python logger."""

    def print(self, *objects, style=None, **kwargs):
        super().print(*objects, style=style, **kwargs)
        if not objects:
            return
        msg = " ".join(str(o) for o in objects)
        # "bold red" is used package-wide as emphasis for headers, not errors
        if style == "red":
            _logger.error(msg)
        elif style in ("yellow",):
            _logger.warning(msg)
        else:
            _logger.info(msg)


console = _LoggingConsole()


def _echo(msg: str, style: str) -> None:
    """Print to the terminal only; the log_* helpers log to file themselves."""
    Console.print(console, msg, style=style)


# ---------------------------------------------------------------------------
# Color-scheme constants (mirrors color_codes.txt)
# ---------------------------------------------------------------------------
DEBUG = "green"
INFO = "cyan"
WARNING = "yellow"
ERROR = "red"
CRITICAL = "bold red"
BLUE_INFO = "blue"

# ---------------------------------------------------------------------------
# Verbosity state — set once at startup via setup_verbosity()
# ---------------------------------------------------------------------------
_quiet: bool = False
_debug: bool = False


def setup_verbosity(
    quiet: bool = False, verbose: bool = False, debug: bool = False
) -> None:
    """
    Configure the verbosity level for all console output.

    Call this once at the start of each CLI command (``prepare``, ``run``,
    ``qc``) before any other output is produced.  All subsequent calls to
    ``log_info``, ``log_debug``, etc. will respect the level set here.

    Parameters
    ----------
    quiet : bool
        Suppress INFO and WARNING output; only ERROR and CRITICAL are shown.
    verbose : bool
        Alias for the default level (INFO and above). Provided for CLI
        symmetry; no additional output is added beyond the default.
    debug : bool
        Show DEBUG messages in addition to all other levels.
    """
    global _quiet, _debug
    _quiet = quiet
    _debug = debug


# ---------------------------------------------------------------------------
# File logging — attach handlers once per CLI run via set_log_files()
# ---------------------------------------------------------------------------
_logger = logging.getLogger("launchcontainers")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(logging.NullHandler())  # silent until set_log_files() is called

_LOG_FMT = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")


def set_log_files(log_fpath: str, err_fpath: str) -> None:
    """
    Attach two FileHandlers to the launchcontainers logger.

    Parameters
    ----------
    log_fpath : str
        ``.log`` file — captures everything (DEBUG and above).
    err_fpath : str
        ``.err`` file — captures WARNING and above only.
    """
    for h in list(_logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            _logger.removeHandler(h)

    log_h = logging.FileHandler(log_fpath)
    log_h.setLevel(logging.DEBUG)
    log_h.setFormatter(_LOG_FMT)
    _logger.addHandler(log_h)

    err_h = logging.FileHandler(err_fpath)
    err_h.setLevel(logging.WARNING)
    err_h.setFormatter(_LOG_FMT)
    _logger.addHandler(err_h)


# ---------------------------------------------------------------------------
# Helper print functions — use these instead of console.print() directly
# ---------------------------------------------------------------------------


def log_debug(msg: str) -> None:
    """Print a debug message (green). Only shown when --debug is set."""
    if _debug:
        _echo(msg, DEBUG)
    _logger.debug(msg)


def log_info(msg: str) -> None:
    """Print an info message (cyan). Suppressed in --quiet mode."""
    if not _quiet:
        _echo(msg, INFO)
    _logger.info(msg)


def log_warning(msg: str) -> None:
    """Print a warning message (yellow). Always logged to file."""
    if not _quiet:
        _echo(msg, WARNING)
    _logger.warning(msg)


def log_error(msg: str) -> None:
    """Print an error message (red). Always shown and logged."""
    _echo(msg, ERROR)
    _logger.error(msg)


def log_critical(msg: str) -> None:
    """Print a critical message (bold red). Always shown and logged."""
    _echo(msg, CRITICAL)
    _logger.critical(msg)
