"""

Background ingest job tracking and serialized execution queue.

"""



import logging

import queue

import threading

import uuid

from dataclasses import dataclass, field

from datetime import datetime

from enum import Enum

from typing import Any, Callable, Dict, List, Optional



logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)





class JobStatus(str, Enum):

    PENDING = "pending"

    EXTRACTING = "extracting"

    EMBEDDING = "embedding"

    INDEXING = "indexing"

    COMPLETED = "completed"

    FAILED = "failed"





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

    """Thread-safe ingest jobs with a single FIFO worker."""



    def __init__(self):

        self._jobs: Dict[str, IngestJob] = {}

        self._lock = threading.RLock()

        self._queue: queue.Queue = queue.Queue()

        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="ingest-worker")

        self._worker.start()



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



    def _update(self, job_id: str, **kwargs: Any) -> None:

        with self._lock:

            job = self._jobs.get(job_id)

            if job:

                for k, v in kwargs.items():

                    setattr(job, k, v)



    def enqueue(self, job_id: str, worker: Callable[[], Any]) -> None:

        """Add job to FIFO queue."""

        self._update(job_id, status=JobStatus.PENDING, message="Queued...", progress=0)

        self._queue.put((job_id, worker))



    def run_in_background(self, job_id: str, worker: Callable[[], Any]) -> None:

        """Alias for enqueue — one worker processes jobs sequentially."""

        self.enqueue(job_id, worker)



    def _worker_loop(self) -> None:

        while True:

            job_id, worker = self._queue.get()

            try:

                result = worker()

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

            finally:

                self._queue.task_done()


