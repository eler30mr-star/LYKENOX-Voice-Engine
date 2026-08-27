"""Regenerate boundary-safe LYKENOX duration caches from the validated best aligner.

This command does not train or modify the aligner checkpoint. It reuses ``best.pt`` and
resumes ``alignment-v2`` generation from already-written per-utterance records. A default
wall-clock budget returns control before short external executor timeouts; rerunning the
same command continues only missing rows. A PID sentinel prevents duplicate concurrent
regeneration processes from competing for the same cache.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.training.speech_duration_cache import generate_duration_cache
from lykenox_voice_engine.training.speech_duration_outlier_review import review_duration_outliers


DEFAULT_TIME_BUDGET_SECONDS = 85.0


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(ok and exit_code.value == still_active)
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


@contextmanager
def _exclusive_run_lock(lock_path: Path):
    """Reject duplicate live runs while automatically clearing stale PID sentinels."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    own_pid = os.getpid()
    for _ in range(2):
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            try:
                existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                existing_pid = -1
            if existing_pid > 0 and _pid_is_alive(existing_pid):
                raise RuntimeError(
                    "Another LYKENOX boundary regeneration process is already running "
                    f"(PID {existing_pid})."
                )
            lock_path.unlink(missing_ok=True)
            continue
        else:
            try:
                os.write(descriptor, str(own_pid).encode("ascii"))
            finally:
                os.close(descriptor)
            break
    else:
        raise RuntimeError(f"Could not acquire regeneration lock: {lock_path}")

    try:
        yield
    finally:
        try:
            if lock_path.exists():
                stored_pid = int(lock_path.read_text(encoding="utf-8").strip())
                if stored_pid == own_pid:
                    lock_path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def _write_progress_report(
    duration_root: Path,
    report: dict[str, object],
) -> None:
    duration_root.mkdir(parents=True, exist_ok=True)
    (duration_root / "boundary_fix_regeneration_progress.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def regenerate_boundary_safe_durations(
    root: Path,
    *,
    threshold_frames: int = 100,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
) -> dict[str, object]:
    root = Path(root).resolve()
    frontend = SpanishTextFrontend()
    artifact_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "speech_aligner"
        / frontend.version
    )
    checkpoint = artifact_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Validated LYKENOX best aligner not found: {checkpoint}")

    lock_path = artifact_dir / "boundary_fix_regenerate.lock"
    with _exclusive_run_lock(lock_path):
        duration_report = generate_duration_cache(
            root,
            checkpoint,
            nonpause_warn_frames=threshold_frames,
            time_budget_seconds=time_budget_seconds,
            progress_every=5,
            resume=True,
        )
        duration_root = Path(str(duration_report["duration_cache_root"]))

        if duration_report["status"] == "incomplete":
            report = {
                "status": "incomplete",
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": duration_report.get("checkpoint_epoch"),
                "duration_cache_version": duration_report.get("duration_cache_version"),
                "boundary_blank_policy": duration_report.get("boundary_blank_policy"),
                "duration_cache_root": str(duration_root),
                "elapsed_seconds": duration_report.get("elapsed_seconds"),
                "time_budget_seconds": duration_report.get("time_budget_seconds"),
                "reused_records": duration_report.get("reused_records"),
                "new_records_generated": duration_report.get("new_records_generated"),
                "pending_item_count": duration_report.get("pending_item_count"),
                "train_generated": duration_report["splits"]["train"]["generated"],
                "train_items": duration_report["splits"]["train"]["items"],
                "val_generated": duration_report["splits"]["val"]["generated"],
                "val_items": duration_report["splits"]["val"]["items"],
                "next_gate": "rerun_same_command_to_resume",
                "note": (
                    "No training is repeated. Existing alignment-v2 utterance records "
                    "are reused on the next run."
                ),
            }
            _write_progress_report(duration_root, report)
            return report

        if duration_report["status"] != "pass":
            report = {
                "status": "duration_generation_failed",
                "checkpoint": str(checkpoint),
                "duration_cache_root": str(duration_root),
                "failures": duration_report.get("failures", []),
                "next_gate": "review_failed_alignments",
            }
            _write_progress_report(duration_root, report)
            return report

        review = review_duration_outliers(
            root,
            duration_root=duration_root,
            threshold_frames=threshold_frames,
        )
        if review["status"] == "pass":
            status = "pass"
            next_gate = "aligned_acoustic_smoke"
        else:
            status = "review_required"
            next_gate = str(review["next_gate"])

        report = {
            "status": status,
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": duration_report.get("checkpoint_epoch"),
            "duration_cache_version": duration_report.get("duration_cache_version"),
            "boundary_blank_policy": duration_report.get("boundary_blank_policy"),
            "duration_cache_root": str(duration_root),
            "elapsed_seconds": duration_report.get("elapsed_seconds"),
            "reused_records": duration_report.get("reused_records"),
            "new_records_generated": duration_report.get("new_records_generated"),
            "train_generated": duration_report["splits"]["train"]["generated"],
            "train_items": duration_report["splits"]["train"]["items"],
            "val_generated": duration_report["splits"]["val"]["generated"],
            "val_items": duration_report["splits"]["val"]["items"],
            "content_duration_frames": duration_report["content_duration_frames"],
            "nonpause_duration_frames": duration_report["nonpause_duration_frames"],
            "leading_boundary_frames": duration_report["leading_boundary_frames"],
            "trailing_boundary_frames": duration_report["trailing_boundary_frames"],
            "outlier_token_count": review["outlier_token_count"],
            "outlier_utterance_count": review["outlier_utterance_count"],
            "boundary_outlier_token_count": review["boundary_outlier_token_count"],
            "interior_outlier_token_count": review["interior_outlier_token_count"],
            "boundary_fraction": review["boundary_fraction"],
            "diagnosis": review["diagnosis"],
            "review_report": review["report_path"],
            "next_gate": next_gate,
        }
        report_path = duration_root / "boundary_fix_regeneration_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report["report_path"] = str(report_path)
        _write_progress_report(duration_root, report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--threshold-frames", type=int, default=100)
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=DEFAULT_TIME_BUDGET_SECONDS,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            regenerate_boundary_safe_durations(
                args.root,
                threshold_frames=args.threshold_frames,
                time_budget_seconds=args.time_budget_seconds,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
