"""
Background ingest job tracking and parallel execution queue.
"""

import logging
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_WORKERS = 3


class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class IngestJob:
    id: str
    notebook_id: int
    source_name: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    message: str = ""
    chunks_total: int = 0
    chunks_done: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "notebook_id": self.notebook_id,
            "source_name": self.source_name,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "chunks_total": self.chunks_total,
            "chunks_done": self.chunks_done,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
        }


class IngestJobManager:
    """Thread-safe ingest jobs with a parallel worker pool."""

    def __init__(self, max_workers: int = MAX_WORKERS):
        self._jobs: Dict[str, IngestJob] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue()
        self._cancelled_notebooks: Set[int] = set()
        self._cancelled_jobs: Set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest")
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True, name="ingest-dispatch")
        self._dispatcher.start()

    def create_job(self, notebook_id: int, source_name: str) -> IngestJob:
        job = IngestJob(
            id=uuid.uuid4().hex[:12],
            notebook_id=notebook_id,
            source_name=source_name,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Optional[IngestJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, notebook_id: Optional[int] = None, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        if notebook_id is not None:
            jobs = [j for j in jobs if j.notebook_id == notebook_id]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    def list_active_jobs(self, notebook_id: Optional[int] = None) -> List[Dict[str, Any]]:
        active = {
            JobStatus.PENDING,
            JobStatus.EXTRACTING,
            JobStatus.EMBEDDING,
            JobStatus.INDEXING,
        }
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.status in active]
        if notebook_id is not None:
            jobs = [j for j in jobs if j.notebook_id == notebook_id]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]

    def is_cancelled(self, job_id: str, notebook_id: int) -> bool:
        with self._lock:
            if job_id in self._cancelled_jobs:
                return True
            if notebook_id in self._cancelled_notebooks:
                return True
        return False

    def cancel_notebook_jobs(self, notebook_id: int) -> None:
        with self._lock:
            self._cancelled_notebooks.add(notebook_id)
            for job in self._jobs.values():
                if job.notebook_id == notebook_id and job.status in {
                    JobStatus.PENDING,
                    JobStatus.EXTRACTING,
                    JobStatus.EMBEDDING,
                    JobStatus.INDEXING,
                }:
                    self._cancelled_jobs.add(job.id)

    def cancel_jobs_for_source(self, notebook_id: int, source_name: str) -> None:
        with self._lock:
            for job in self._jobs.values():
                if (
                    job.notebook_id == notebook_id
                    and job.source_name == source_name
                    and job.status in {
                        JobStatus.PENDING,
                        JobStatus.EXTRACTING,
                        JobStatus.EMBEDDING,
                        JobStatus.INDEXING,
                    }
                ):
                    self._cancelled_jobs.add(job.id)

    def _update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)

    def enqueue(self, job_id: str, worker: Callable[[], Any]) -> None:
        self._update(job_id, status=JobStatus.PENDING, message="Queued...", progress=0)
        self._queue.put((job_id, worker))

    def run_in_background(self, job_id: str, worker: Callable[[], Any]) -> None:
        self.enqueue(job_id, worker)

    def _dispatch_loop(self) -> None:
        while True:
            job_id, worker = self._queue.get()
            self._executor.submit(self._run_job, job_id, worker)
            self._queue.task_done()

    def _run_job(self, job_id: str, worker: Callable[[], Any]) -> None:
        job = self.get_job(job_id)
        if job and self.is_cancelled(job_id, job.notebook_id):
            self._update(
                job_id,
                status=JobStatus.CANCELLED,
                message="Cancelled",
                progress=0,
            )
            return

        try:
            result = worker()

            if job and self.is_cancelled(job_id, job.notebook_id):
                self._update(
                    job_id,
                    status=JobStatus.CANCELLED,
                    message="Cancelled",
                    progress=0,
                )
                return

            if isinstance(result, dict) and result.get("error"):
                err = str(result["error"])
                self._update(
                    job_id,
                    status=JobStatus.FAILED,
                    error=err,
                    message=err,
                    progress=0,
                    result=result,
                )
                return

            self._update(
                job_id,
                status=JobStatus.COMPLETED,
                progress=100,
                message="Complete",
                result=result if isinstance(result, dict) else {"status": "ok"},
                chunks_total=result.get("chunks", 0) if isinstance(result, dict) else 0,
                chunks_done=result.get("chunks", 0) if isinstance(result, dict) else 0,
            )
        except Exception as e:
            logger.error(f"Ingest job {job_id} failed: {e}")
            self._update(
                job_id,
                status=JobStatus.FAILED,
                error=str(e),
                message=str(e),
                progress=0,
            )
