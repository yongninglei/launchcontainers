# from __future__ import annotations

# from .local import run_local
# from .sge import run_sge
# from .slurm import run_slurm

# _DRIVERS = {
#     'local': run_local,
#     'slurm': run_slurm,
#     'sge': run_sge,
# }


# def get_runner(backend: str):
#     try:
#         return _DRIVERS[backend]
#     except KeyError:
#         raise ValueError(f'Unknown backend {backend!r}, choose from {_DRIVERS.keys()}')
