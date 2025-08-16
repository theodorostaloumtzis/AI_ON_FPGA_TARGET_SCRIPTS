#!/usr/bin/env python3
"""
Run all rfmodels bitstreams end-to-end:
  • on_target.py  (inference + power)
  • validate_results.py (metrics, plots)

Bitfiles are discovered under: bitfiles/rfmodels/*.bit
Metrics are saved under:       metrics/rfmodels/<bit-stem>/

TTY-friendly logging preserves tqdm animations and tees to logs/run_*.log.
"""
from __future__ import annotations
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Literal, Optional, Sequence, List
from datetime import datetime

# --- Project paths ---
USER = "ubuntu"
PRJ = "AI_ON_FPGA_TARGET_SCRIPTS"
PWD = Path(f"/home/{USER}/{PRJ}").resolve()

RF_DIR = PWD / "bitfiles" / "rfmodels"
OUT_DIR = PWD / "metrics" / "rfmodels"

# --- Logging setup ---
LOG_DIR = PWD / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = LOG_DIR / f"run_rf_{timestamp}.log"

# open in text for simple prints + in bytes for PTY tee in run()
_log_text = open(LOG_FILE, "w", buffering=1)

def log_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=_log_text)

# --- Utility: run a command via PTY, tee to terminal + log ---
def run(cmd: Sequence[str], *, use_pty: bool = True) -> int:
    """
    Run a subprocess and tee its output to console + log.
    - When use_pty=True, attach a PTY to preserve tqdm progress bars.
    - When use_pty=False, use pipes (safer for simple scripts).
    Handles EIO on PTY reads gracefully when the child exits.
    """
    log_print(f"[cmd] {' '.join(cmd)}")
    if use_pty:
        import pty, select, errno
        master_fd, slave_fd = pty.openpty()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=None,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                cwd=PWD,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        finally:
            os.close(slave_fd)

        with open(LOG_FILE, "ab", buffering=0) as lf:
            try:
                while True:
                    if proc.poll() is not None:
                        r, _, _ = select.select([master_fd], [], [], 0)
                        if master_fd not in r:
                            break
                    r, _, _ = select.select([master_fd], [], [], 0.1)
                    if master_fd in r:
                        try:
                            data = os.read(master_fd, 4096)
                        except OSError as e:
                            if e.errno in (errno.EIO, 5):  # child exited
                                break
                            raise
                        if not data:
                            break
                        os.write(sys.stdout.fileno(), data)
                        lf.write(data)
            finally:
                os.close(master_fd)
        return proc.wait()
    else:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PWD,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        with open(LOG_FILE, "a", buffering=1) as lf:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
        return proc.wait()

# --- Helpers ---
def cd_project() -> None:
    try:
        os.chdir(PWD)
        log_print(f"[cwd] {Path.cwd()}")
    except FileNotFoundError:
        log_print(f"[error] Project path not found: {PWD}")
        sys.exit(1)

_num_re = re.compile(r"rfmodel(?P<num>[0-9_]+)", re.IGNORECASE)

def _extract_numeric_key(stem: str) -> float:
    """
    rfmodel6_25_cnn -> 6.25
    rfmodel12_5_cnn -> 12.5
    rfmodel25_cnn   -> 25.0
    """
    m = _num_re.search(stem)
    if not m:
        return float("inf")
    raw = m.group("num").replace("_", ".")
    try:
        return float(raw)
    except ValueError:
        return float("inf")

def find_rf_bitfiles(rf_dir: Path = RF_DIR) -> List[Path]:
    if not rf_dir.exists():
        raise FileNotFoundError(f"RF models dir not found: {rf_dir}")
    bits = sorted(rf_dir.glob("*.bit"), key=lambda p: _extract_numeric_key(p.stem))
    if not bits:
        raise FileNotFoundError(f"No .bit files found in {rf_dir}")
    return bits

# --- Script wrappers ---
def clear_bitfile() -> None:
    # Keep simple, no PTY needed
    run([sys.executable, "clear_global_state.py"], use_pty=False)

def run_inference(
    bitfile: str,
    metrics_dir: str,
    *,
    pack: bool = False,
    cycles: Literal["auto", "on", "off"] = "auto",
    cycle_type: Literal["core", "e2e", "both"] = "core",
    # Power options
    power_off: bool = False,
    power_poll: float = 0.07,
    power_frames: int = 128,
    idle_seconds: float = 1.0,
    no_progress: bool = False,
    power_rail: Optional[list[str]] = None,
) -> None:
    cmd = [
        sys.executable, "on_target.py",
        "-b", bitfile,
        "-m", str(metrics_dir),
        "--cycles", cycles,
        "--cycle-type", cycle_type,
        "--power-poll", str(power_poll),
        "--power-frames", str(power_frames),
        "--idle-seconds", str(idle_seconds),
    ]
    if pack:
        cmd.append("-pd")
    if power_off:
        cmd.append("--power-off")
    if no_progress:
        cmd.append("--no-progress")
    if power_rail:
        for pr in power_rail:
            cmd += ["--power-rail", pr]

    rc = run(cmd, use_pty=True)
    if rc != 0:
        raise RuntimeError(f"on_target.py failed for {bitfile} (rc={rc})")

def validate_results(
    metrics_dir: str,
    *,
    clk_mhz: Optional[float] = 150.0,
    exclude_pct: float = 1.0,
    show: bool = False,
    cycles_core_file: Optional[str] = "cycles_core.npy",
    cycles_e2e_file: Optional[str] = "cycles_e2e.npy",
    power_abs: Optional[str] = "power_abs.npy",
    power_dyn: Optional[str] = "power_dyn.npy",
) -> None:
    cmd = [sys.executable, "validate_results.py", "-m", str(metrics_dir)]
    if clk_mhz is not None:
        cmd += ["--clk-mhz", str(clk_mhz)]
    if exclude_pct and exclude_pct > 0:
        cmd += ["--exclude-pct", str(exclude_pct)]
    if not show:
        cmd.append("--no-show")
    if cycles_core_file:
        cmd += ["--cycles-core", cycles_core_file]
    if cycles_e2e_file:
        cmd += ["--cycles-e2e", cycles_e2e_file]
    if power_abs:
        cmd += ["--power-abs", power_abs]
    if power_dyn:
        cmd += ["--power-dyn", power_dyn]

    rc = run(cmd, use_pty=True)
    if rc != 0:
        raise RuntimeError(f"validate_results.py failed for {metrics_dir} (rc={rc})")

# --- Main orchestration ---
def main() -> None:
    cd_project()
    bits = find_rf_bitfiles(RF_DIR)
    log_print(f"Discovered {len(bits)} RF models:")
    for b in bits:
        log_print(f"  - {b.name}")

    for bit in bits:
        stem = bit.stem  # e.g., "rfmodel6_25_cnn"
        out_dir = OUT_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        log_print("\n────────────────────────────────────────────────────────")
        log_print(f"[RUN] {bit.name}")
        log_print("Clearing PL/driver state...")
        clear_bitfile()

        log_print("Running inference (on_target.py)…")
        # Most RF variants should use the same stream format; toggle pack if needed.
        run_inference(
            str(bit),
            metrics_dir=str(out_dir),
            cycles="auto",
            cycle_type="core",
            # power_off=True,   # uncomment to skip power sampling for speed
            no_progress=False,  # let on_target show its tqdm
        )

        log_print("Validating results (validate_results.py)…")
        validate_results(
            metrics_dir=str(out_dir),
            clk_mhz=150.0,
            exclude_pct=1.0,
            show=False,
        )

    log_print("\nAll RF models processed. Logs at:", LOG_FILE)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_print("\n[interrupt] Exiting due to user interrupt.")
        sys.exit(130)
    except Exception as e:
        log_print(f"[fatal] {e}")
        sys.exit(1)
    finally:
        try:
            _log_text.close()
        except Exception:
            pass
