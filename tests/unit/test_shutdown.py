"""Closing Materia stops everything Materia started.

A GPAW calculation runs in its own session so that cancelling it does not take
the application down with it.  The same isolation means it can outlive the
application unless something deliberately stops it, which is what these cases
check.  They use controlled subprocesses rather than real GPAW runs, including
one that ignores the first termination request, so the force-kill path is
exercised without leaving a first-principles calculation behind.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from materia.desktop_ui.jobs import JobQueue
from materia.solvers.gpaw_driver import runner as gpaw_runner

PATIENT = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
    "time.sleep(120)\n"
)

OBEDIENT = (
    "import sys, time\n"
    "sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
    "time.sleep(120)\n"
)


def spawn(script: str):
    """A registered stand-in for a GPAW worker, in its own session."""
    workdir = tempfile.mkdtemp(prefix="materia-shutdown-test-")
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, cwd=workdir, start_new_session=True)
    assert process.stdout.readline().strip() == "ready"
    gpaw_runner._register(process, workdir)
    return process, workdir


def alive(process) -> bool:
    if process.poll() is not None:
        return False
    try:
        os.killpg(os.getpgid(process.pid), 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


@pytest.fixture(autouse=True)
def no_leftovers():
    yield
    gpaw_runner.terminate_all(grace_s=2.0)


class TestTerminatingSubprocesses:

    def test_a_cooperative_process_is_stopped(self):
        process, workdir = spawn(OBEDIENT)
        assert gpaw_runner.active_runs() == 1
        assert gpaw_runner.terminate_all(grace_s=5.0) == 1
        assert not alive(process)
        assert gpaw_runner.active_runs() == 0
        assert not os.path.exists(workdir)

    def test_a_process_that_ignores_the_first_signal_is_killed(self):
        process, workdir = spawn(PATIENT)
        started = time.perf_counter()
        assert gpaw_runner.terminate_all(grace_s=1.0) == 1
        assert not alive(process)
        assert time.perf_counter() - started < 20.0
        assert not os.path.exists(workdir)

    def test_diagnostic_retention_keeps_the_directory(self):
        process, workdir = spawn(OBEDIENT)
        gpaw_runner.terminate_all(grace_s=5.0, clean=False)
        assert not alive(process)
        assert os.path.exists(workdir)
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    def test_terminating_nothing_is_harmless(self):
        assert gpaw_runner.terminate_all() == 0
        assert gpaw_runner.terminate_all() == 0


class TestJobQueueShutdown:

    def test_a_finished_job_stays_done(self):
        queue = JobQueue()
        job = queue.submit("t", "quick", lambda j: "answer")
        for _ in range(100):
            if job.status == "done":
                break
            time.sleep(0.02)
        report = queue.shutdown(grace_s=5.0)
        assert job.status == "done"
        assert job.result == "answer"
        assert report["first_call"] is True

    def test_a_running_job_finishes_cancelled_with_the_reason(self):
        queue = JobQueue()
        started = threading.Event()

        def work(job):
            started.set()
            while not job.cancel_flag.is_set():
                time.sleep(0.02)
            return "stopped"

        job = queue.submit("t", "slow", work)
        assert started.wait(5.0)
        queue.shutdown(grace_s=5.0)
        assert job.status == "cancelled"
        assert "shutting down" in job.cancel_reason
        assert job.status != "done"

    def test_a_job_that_ignores_cancellation_is_still_not_reported_as_done(self):
        queue = JobQueue()
        started = threading.Event()
        release = threading.Event()

        def stubborn(job):
            started.set()
            release.wait(30)
            return "late"

        job = queue.submit("t", "stubborn", stubborn)
        assert started.wait(5.0)
        report = queue.shutdown(grace_s=0.5)
        assert job.status == "cancelled"
        assert job.id in report["stranded"]
        release.set()

    def test_shutdown_terminates_registered_subprocesses(self):
        queue = JobQueue()
        process, _ = spawn(OBEDIENT)
        report = queue.shutdown(grace_s=5.0)
        assert report["subprocesses_terminated"] >= 1
        assert not alive(process)

    def test_repeated_shutdown_is_harmless(self):
        queue = JobQueue()
        first = queue.shutdown(grace_s=1.0)
        second = queue.shutdown(grace_s=1.0)
        third = queue.shutdown(grace_s=1.0)
        assert first["first_call"] is True
        assert second["first_call"] is False
        assert third["first_call"] is False

    def test_submission_after_shutdown_is_refused_without_running(self):
        queue = JobQueue()
        queue.shutdown(grace_s=1.0)
        ran = threading.Event()
        job = queue.submit("t", "late", lambda j: ran.set())
        time.sleep(0.3)
        assert job.status == "cancelled"
        assert "shutting down" in job.cancel_reason
        assert not ran.is_set()

    def test_queued_jobs_are_cancelled(self):
        queue = JobQueue()
        gate = threading.Event()

        def blocked(job):
            gate.wait(10)
            return "x"

        jobs = [queue.submit("t", f"j{i}", blocked) for i in range(3)]
        time.sleep(0.2)
        queue.shutdown(grace_s=0.5)
        assert all(j.status == "cancelled" for j in jobs)
        gate.set()


class TestServiceShutdown:

    def test_the_service_exposes_an_idempotent_shutdown(self):
        from materia.desktop_ui.service import Service

        service = Service()
        assert service.jobs.is_shut_down is False
        first = service.shutdown(grace_s=2.0)
        assert first["first_call"] is True
        assert service.jobs.is_shut_down is True
        assert service.shutdown(grace_s=2.0)["first_call"] is False

    def test_the_native_close_handler_shuts_the_service_down(self):
        source = (
            __import__("pathlib").Path("materia/desktop_ui/desktop.py").read_text())
        handler = source[source.index("def on_closed():"):]
        handler = handler[:handler.index("window.events.closed")]
        assert "service.shutdown()" in handler
        assert "server.shutdown()" in handler

    def test_the_headless_server_shuts_the_service_down(self):
        source = (
            __import__("pathlib").Path("materia/desktop_ui/server.py").read_text())
        assert "service.shutdown()" in source
