import logging
import os
from collections.abc import Callable

import dask
import dask.delayed
from tqdm import tqdm

log = logging.getLogger(__name__)


class Settings:
    ### General
    auto_save_on_disk: bool = True

    ### Parallelization
    _parallelization_backend: str | None = None
    available_parallelization_backends = [None, "dask"]
    dask_client_kwargs: dict = {}

    ### Backend settings
    native_read_region: bool = True

    ### Segmentation or aggregation
    gene_exclude_pattern: str | None = (
        "nan|<NA>|.*control.*|blank.*|antisense.*|unassigned.*|deprecated.*|intergenic.*|false.*|neg.*"
    )
    simplification_tolerance: float | None = None

    def __init__(self):
        self.parallelization_backend = os.environ.get("SOPA_PARALLELIZATION_BACKEND", None)

    @property
    def parallelization_backend(self):
        return self._parallelization_backend

    @parallelization_backend.setter
    def parallelization_backend(self, value):
        assert value in self.available_parallelization_backends, (
            f"Invalid parallelization backend. Available options are: {self.available_parallelization_backends}"
        )
        self._parallelization_backend = value

    def _run_with_backend(self, functions: list[Callable]):
        assert len(functions) > 0, "No function to run"

        if len(functions) == 1 or self.parallelization_backend is None:
            if len(functions) > 1:
                log.warning(
                    "Running without parallelization backend can be slow. "
                    "Consider using a backend, e.g. via `sopa.settings.parallelization_backend = 'dask'`, "
                    "or `export SOPA_PARALLELIZATION_BACKEND=dask`."
                )
            return [f() for f in tqdm(functions)]

        log.info(f"Using {self.parallelization_backend} backend")
        return getattr(self, f"_run_{self.parallelization_backend}_backend")(functions)

    ### Dask backend
    def _run_dask_backend(self, functions: list[Callable]):
        import logging
        import os

        from dask.distributed import Client, progress
        from distributed.scheduler import logger as _logger

        _logger.setLevel(logging.ERROR)

        n_cpus: int = os.cpu_count()
        dask_client_kwargs = self.dask_client_kwargs.copy()
        if "n_workers" not in dask_client_kwargs and len(functions) < n_cpus:
            dask_client_kwargs["n_workers"] = len(functions)

        with Client(**dask_client_kwargs) as client:
            ram_per_worker = client.cluster.workers[0].memory_manager.memory_limit

            if ram_per_worker < 4 * 1024**3:
                log.warning(
                    f"Each worker has less than 4GB of RAM ({ram_per_worker / 1024**3:.2f}GB), which may not be enough. "
                    f"Consider setting `sopa.settings.dask_client_kwargs['n_workers']` to use less workers ({len(client.cluster.workers)} currently)."
                )

            functions = [dask.delayed(function)() for function in functions]
            futures = dask.persist(*functions)
            progress(futures, notebook=False)
            return client.gather(client.compute(futures))


settings = Settings()
