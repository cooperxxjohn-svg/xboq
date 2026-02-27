"""
JobQueue — abstract job queue + LocalThreadQueue implementation.

Provides async pipeline execution: submit a function, poll for status,
get results when done. UI polls get_status() to show progress.

Sprint 16: Hosted pilot readiness.
"""

import enum
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from abc import ABC, abstractmethod


class JobStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a background job with progress tracking."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    error: str = ""
    result: Any = None
    _cancelled: bool = field(default=False, repr=False)

    def to_dict(self) -> dict:
        """Serialize to dict (excludes result for safe transport)."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error[:500] if self.error else "",
        }


class JobQueue(ABC):
    """Abstract job queue interface."""

    @abstractmethod
    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> str:
        """Submit a function for async execution. Returns job_id."""
        ...

    @abstractmethod
    def get_status(self, job_id: str) -> Optional[Job]:
        """Get current job status. Returns None if not found."""
        ...

    @abstractmethod
    def list_jobs(self, limit: int = 20) -> List[Job]:
        """List recent jobs, newest first."""
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Request cancellation. Returns True if job was found."""
        ...


class LocalThreadQueue(JobQueue):
    """Thread-pool based job queue for local/dev deployment.

    Uses concurrent.futures.ThreadPoolExecutor for background execution.
    Job progress is tracked via an injected progress_callback.
    """

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> str:
        """Submit a function for background execution.

        If the target function accepts a `progress_callback` kwarg, it will
        be injected with a callback that updates the Job's progress fields.

        Args:
            fn: Callable to execute (e.g., run_analysis_pipeline).
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            job_id string for polling status.
        """
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job = Job(
            job_id=job_id,
            created_at=datetime.now().isoformat(),
        )

        with self._lock:
            self._jobs[job_id] = job

        def _wrapper():
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now().isoformat()
            try:
                # Inject progress callback that updates the Job object
                def _progress_cb(stage: str, message: str, progress: float):
                    job.stage = stage
                    job.message = message
                    job.progress = min(max(progress, 0.0), 1.0)

                kwargs["progress_callback"] = _progress_cb

                result = fn(*args, **kwargs)
                job.result = result
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
            except Exception:
                job.error = traceback.format_exc()
                job.status = JobStatus.FAILED
            finally:
                job.completed_at = datetime.now().isoformat()

        self._executor.submit(_wrapper)
        return job_id

    def get_status(self, job_id: str) -> Optional[Job]:
        """Get current status of a job."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> List[Job]:
        """List recent jobs, newest first by created_at."""
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of a job. Best-effort for running jobs.

        Returns True if the job was found, False otherwise.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now().isoformat()
            elif job.status == JobStatus.RUNNING:
                job._cancelled = True
            return True
