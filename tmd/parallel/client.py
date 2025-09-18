# Copyright 2019-2025, Relay Therapeutics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import multiprocessing
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from concurrent import futures
from pathlib import Path
from subprocess import check_output
from typing import Any, Optional
from uuid import uuid4

from tmd.parallel.utils import get_gpu_count, get_visible_gpus

# (ytz): The classes in this file are designed to help provide a consistent API between
# multiprocessing (typically for local cluster use) and gRPC (distributed and multi-node).


class BaseFuture(ABC):
    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def result(self) -> Any: ...

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class AbstractClient:
    def __init__(self):
        self.max_workers = 1

    def submit(self, task_fn, *args, **kwargs) -> BaseFuture:
        """
        Submit is an asynchronous method that will launch task_fn whose
        results will be collected at a later point in time. The input task_fn
        and its arguments should be picklable. See Python documentation for
        pickle rules.

        Parameters
        ----------
        task_fn: callable
            A python function to be called

        args: list
            list of arguments for task_fn


        Returns
        -------
        Future
            A deferred object with a .result() method.

        Usage:

        client = ConcreteClient()

        futures = []
        for arg in args:
            fut = client.submit(task_fn, arg)
            futures.append(fut)

        res = []
        for fut in futures:
            res.append(fut.result())

        """
        raise NotImplementedError()

    def verify(self):
        """Verify performs any necessary checks to verify the client is ready to
        handle calls to submit.

        Raises
        ------
        Exception
            If verification fails
        """
        raise NotImplementedError()


class _MockFuture(BaseFuture):
    __slots__ = ("_id", "val")

    def __init__(self, val):
        self.val = val
        self._id = str(uuid4())

    def result(self) -> Any:
        return self.val

    def done(self) -> bool:
        return True

    @property
    def id(self) -> str:
        """
        Return the id as a str for this subjob
        """
        return self._id

    @property
    def name(self) -> str:
        """
        Return the name as a str for this subjob
        """
        return self._id


class WrappedFuture(BaseFuture):
    def __init__(self, future, job_id: str):
        self._future = future
        self._id = job_id

    def result(self) -> Any:
        return self._future.result()

    def done(self) -> bool:
        return self._future.done()

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._id


class SerialClient(AbstractClient):
    def submit(self, task_fn, *args, **kwargs) -> BaseFuture:
        return _MockFuture(task_fn(*args, **kwargs))

    def verify(self):
        return


class ProcessPoolClient(AbstractClient):
    def __init__(self, max_workers):
        """
        Generic wrapper around ProcessPoolExecutor. Each call to submit()
        will be run on a different worker.  If the number of jobs submitted
        is larger than the number of workers, the jobs will be batched. Each
        worker will run at most one job.

        Parameters
        ----------
        max_workers: int
            Number of workers to launch via the ProcessPoolExecutor

        """
        self.max_workers = max_workers
        self._idx = 0
        self._total_idx = 0
        ctxt = multiprocessing.get_context("spawn")
        self.executor = futures.ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctxt)

    def submit(self, task_fn, *args, **kwargs) -> BaseFuture:
        """
        See abstract class for documentation.
        """
        future = self.executor.submit(task_fn, *args, **kwargs)
        job_id = str(self._total_idx)
        self._total_idx += 1
        self._idx = (self._idx + 1) % self.max_workers
        return WrappedFuture(future, job_id)

    def verify(self):
        """
        See abstract class for documentation.
        """
        return

    def __getstate__(self):
        # Only store the max workers in the pickle
        return (self.max_workers,)

    def __setstate__(self, state):
        max_workers = state[0]
        self.__init__(max_workers)


class CUDAPoolClient(ProcessPoolClient):
    """
    Specialized wrapper for CUDA-dependent processes. Each call to submit()
    will run on a different GPU modulo num workers, which should be set to
    the number of GPUs.
    """

    def __init__(self, max_workers: int):
        super().__init__(max_workers)
        self._gpu_list = get_visible_gpus(max_workers)

    @staticmethod
    def wrapper(max_workers, idx, fn, *args, **kwargs):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        return fn(*args, **kwargs)

    def submit(self, task_fn, *args, **kwargs) -> BaseFuture:
        """
        See abstract class for documentation.
        """
        future = self.executor.submit(
            self.wrapper, self.max_workers, self._gpu_list[self._idx], task_fn, *args, **kwargs
        )
        job_id = str(self._total_idx)
        self._total_idx += 1
        self._idx = (self._idx + 1) % self.max_workers
        return WrappedFuture(future, job_id)

    def verify(self):
        """
        See abstract class for documentation.
        """
        gpus = get_gpu_count()
        assert self.max_workers <= gpus, f"More workers '{self.max_workers}' requested than GPUs '{gpus}'"
        assert len(self._gpu_list) >= self.max_workers, (
            "Fewer available GPUs than max workers expects, check CUDA_VISIBLE_DEVICES"
        )
        assert len(self._gpu_list) <= gpus, "More GPUs requested than the machine has, check CUDA_VISIBLE_DEVICES"


class CUDAMPSPoolClient(ProcessPoolClient):
    """Specialized wrapper for CUDA-dependent processes, when running MPS (https://docs.nvidia.com/deploy/mps/index.html). If MPS is not running,
    this will perform worse than CUDAPoolClient.
    """

    def __init__(self, num_gpus: int, workers_per_gpu: int = 4, active_thread_usage_per_worker: float | None = None):
        """
        Parameters
        ----------

        num_gpus: int
            The number of GPUs to use

        workers_per_gpu: int
            The number of processes allocated to each GPU

        active_thread_usage_per_worker: float or None
            Sets the environment variable `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` for each process. Defaults to 200 / workers_per_gpu,
            typically best left to the default.
        """
        super().__init__(num_gpus * workers_per_gpu)
        assert workers_per_gpu > 0
        assert active_thread_usage_per_worker is None or active_thread_usage_per_worker > 0.0
        self.workers_per_gpu = workers_per_gpu
        self.num_gpus = num_gpus
        self._gpu_list = get_visible_gpus(num_gpus)
        if active_thread_usage_per_worker is None:
            # Heuristic taken from https://developer.nvidia.com/blog/maximizing-openmm-molecular-dynamics-throughput-with-nvidia-multi-process-service/#more_throughput_with_cuda_mps_active_thread_percentage%C2%A0
            self._active_thread_usage_per_worker = 200.0 / self.workers_per_gpu
        else:
            self._active_thread_usage_per_worker = active_thread_usage_per_worker

    @staticmethod
    def wrapper(max_workers, idx, thread_percentage, fn, *args, **kwargs):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(thread_percentage)
        return fn(*args, **kwargs)

    def submit(self, task_fn, *args, **kwargs) -> BaseFuture:
        """
        See abstract class for documentation.
        """
        future = self.executor.submit(
            self.wrapper,
            self.max_workers,
            self._gpu_list[self._idx % len(self._gpu_list)],
            self._active_thread_usage_per_worker,
            task_fn,
            *args,
            **kwargs,
        )
        job_id = str(self._total_idx)
        self._total_idx += 1
        self._idx = (self._idx + 1) % self.max_workers
        return WrappedFuture(future, job_id)

    def verify(self):
        """
        See abstract class for documentation.
        """
        gpus = get_gpu_count()
        assert self.num_gpus <= gpus, f"More GPUs '{self.num_gpus}' requested than GPUs '{gpus}' available"
        assert len(self._gpu_list) >= self.num_gpus, (
            "Fewer available GPUs than GPUs requested, check CUDA_VISIBLE_DEVICES"
        )
        assert len(self._gpu_list) <= gpus, "More GPUs requested than the machine has, check CUDA_VISIBLE_DEVICES"
        try:
            # Timeout necessary, since sometimes cuda MPS will hang
            output = check_output(["nvidia-cuda-mps-control"], input=b"get_server_list", timeout=10.0)
        except FileNotFoundError:
            output = ""
        if len(output.strip()) == 0:
            warnings.warn("Cuda MPS doesn't appear to be running, GPU performance may be slower than expected")


class BinaryFutureWrapper:
    def __init__(self, future, job_id):
        """
        Utility class to help unwrap pickle'd Future objects.
        """
        self._future = future
        self._id = job_id

    def result(self) -> Any:
        return pickle.loads(self._future.result().binary)

    def done(self) -> bool:
        return self._future.done()

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return str(self._id)


class AbstractFileClient:
    def store_stream(self, path: str | Path, stream: io.IOBase):
        """
        Store a stream of binary data to a given path.

        Parameters
        ----------
        path:
            Relative path to store the data. The client may interpret
            this path as appropriate (i.e. file path, s3 path, etc).

        stream:
            Stream containing binary contents.
        """
        raise NotImplementedError()

    def store(self, path: str | Path, data: bytes):
        """
        Store the results to the given path.

        Parameters
        ----------
        path:
            Relative path to store the data. The client may interpret
            this path as appropriate (i.e. file path, s3 path, etc).

        data:
            Binary contents to store.
        """
        raise NotImplementedError()

    def load(self, path: str | Path) -> bytes:
        """
        Load the results from the given path.

        Parameters
        ----------
        path:
            Path to load from, the value returned by the `store` method.

        Returns
        -------
        bytes
            Binary contents of the file.
        """
        raise NotImplementedError()

    def exists(self, path: str | Path) -> bool:
        """
        Parameters
        ----------
        path:
            Path to load from, the value returned by the `store` method.

        Returns
        -------
        bool
            True if the results exist at this path.
        """
        raise NotImplementedError()

    def full_path(self, path: str | Path) -> str:
        """
        Parameters
        ----------
        path:
            Relative path to use.

        Returns
        -------
        str:
            The full path, the meaning of which depends on the
            subclass.
        """
        raise NotImplementedError()

    def delete(self, path: str | Path):
        """
        Parameters
        ----------
        path:
            Relative path to delete.
        """
        raise NotImplementedError()


class FileClient(AbstractFileClient):
    def __init__(self, base: Optional[Path | str] = None):
        self.base = base or Path().cwd()

    def store_stream(self, path: str | Path, stream: io.IOBase):
        full_path = Path(self.full_path(path))
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "wb") as ofs:
            chunk = stream.read(io.DEFAULT_BUFFER_SIZE)
            while chunk:
                ofs.write(chunk)
                chunk = stream.read(io.DEFAULT_BUFFER_SIZE)

    def store(self, path: str | Path, data: bytes):
        full_path = Path(self.full_path(path))
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)

    def load(self, path: str | Path) -> bytes:
        full_path = Path(self.full_path(path))
        return full_path.read_bytes()

    def exists(self, path: str | Path) -> bool:
        return Path(self.full_path(path)).exists()

    def full_path(self, path: str | Path) -> str:
        return str(Path(self.base, path).absolute())

    def delete(self, path: str | Path):
        Path(self.full_path(path)).unlink()


def save_results(result_paths: list[str], local_file_client: FileClient, remote_file_client: AbstractFileClient):
    """
    Load the results from `remote_file_client`, which may be remote and
    store them on the local file system using `local_file_client`.
    """
    for result_path in result_paths:
        if not local_file_client.exists(result_path):
            local_file_client.store(result_path, remote_file_client.load(result_path))


def iterate_completed_futures(futures: Sequence[BaseFuture]) -> Iterator[BaseFuture]:
    """Given a set of futures, return an iterator of futures whose `done()` function returns True.

    Useful for when the results of the futures take different amounts of time
    """
    while len(futures) > 0:
        leftover = []
        for fut in futures:
            if fut.done():
                yield fut
            else:
                leftover.append(fut)
        futures = leftover
