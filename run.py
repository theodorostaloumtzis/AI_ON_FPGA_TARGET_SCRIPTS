#!/usr/bin/env python3
"""
Run the full workflow as a single script, with PTY-safe logging that
preserves tqdm progress bars while teeing output to a timestamped log file.
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Literal, Optional, Sequence
from datetime import datetime

# --- Project paths ---
USER = "ubuntu"
PRJ = "AI_ON_FPGA_TARGET_SCRIPTS"
PWD = Path(f"/home/{USER}/{PRJ}").resolve()

# --- Logging setup ---
LOG_DIR = PWD / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = LOG_DIR / f"run_{timestamp}.log"

# open in text for simple prints + in bytes for PTY tee in run()
_log_text = open(LOG_FILE, "w", buffering=1)

def log_print(*args, **kwargs):
    """Print to console AND to the log file."""
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

# --- Env helpers ---
def cd_project() -> None:
    try:
        os.chdir(PWD)
        log_print(f"[cwd] {Path.cwd()}")
    except FileNotFoundError:
        log_print(f"[error] Project path not found: {PWD}")
        sys.exit(1)

# --- Notebook functions, adapted ---
def clear_bitfile() -> None:
    # Simple script, no need for PTY
    run([sys.executable, "clear_global_state.py"], use_pty=False)

def run_inference(
    bitfile: str,
    metrics_dir: str = "metrics/",
    *,
    pack: bool = False,
    cycles: Literal["auto", "on", "off"] = "auto",
    cycle_type: Literal["core", "e2e", "both"] = "core",
    # Power options (mirrors on_target.py defaults)
    power_off: bool = False,
    power_poll: float = 0.07,
    power_frames: int = 128,
    power_rail: Optional[list[str]] = None,
    idle_seconds: float = 1.0,
    no_progress: bool = False,
) -> None:
    """
    Launch on_target.py with the new CLI. PTY is enabled to preserve tqdm.
    """
    if not bitfile:
        raise ValueError("You must supply a bitfile path.")
    if cycles not in {"auto", "on", "off"}:
        raise ValueError("cycles must be 'auto', 'on', or 'off'.")

    mdir = Path(metrics_dir).expanduser().resolve()
    mdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "on_target.py",
        "-b", bitfile,
        "-m", str(mdir),
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
        raise RuntimeError(f"on_target.py exited with code {rc}")

def validate_results(
    metrics_dir: str = "metrics/",
    *,
    clk_mhz: Optional[float] = None,
    exclude_pct: float = 0.0,
    show: bool = True,
    # Legacy core cycles name; can be overridden by explicit files
    cycles_file: Optional[str] = "cycles_core.npy",
    cycles_core_file: Optional[str] = None,
    cycles_e2e_file: Optional[str] = None,
    # Optional power traces
    power_abs: Optional[str] = "power_abs.npy",
    power_dyn: Optional[str] = "power_dyn.npy",
) -> None:
    """
    Launch validate_results.py, compatible with the refactor.
    """
    mdir = Path(metrics_dir).expanduser().resolve()
    if not mdir.exists():
        raise FileNotFoundError(f"{mdir} does not exist")

    core_path = cycles_core_file if cycles_core_file is not None else cycles_file
    e2e_path  = cycles_e2e_file

    cmd = [
        sys.executable, "validate_results.py",
        "-m", str(mdir),
    ]
    if core_path:
        cmd += ["--cycles-core", core_path]
    if e2e_path:
        cmd += ["--cycles-e2e", e2e_path]
    if power_abs:
        cmd += ["--power-abs", power_abs]
    if power_dyn:
        cmd += ["--power-dyn", power_dyn]
    if clk_mhz is not None:
        cmd += ["--clk-mhz", str(clk_mhz)]
    if exclude_pct > 0.0:
        cmd += ["--exclude-pct", str(exclude_pct)]
    if not show:
        cmd.append("--no-show")

    rc = run(cmd, use_pty=True)  # PTY so matplotlib/tqdm output is clean
    if rc != 0:
        raise RuntimeError(f"validate_results.py exited with code {rc}")

def assemble_paths():
    bitfiles = PWD / "bitfiles"
    return {
        "baseline_bit":    str(bitfiles / "baseline" / "baseline_cnn.bit"),
        "quant_bit":       str(bitfiles / "quantized" / "quant_cnn.bit"),
        "optim_bit":       str(bitfiles / "optim" / "optim_cnn.bit"),
        "optim64_bit":     str(bitfiles / "optim64" / "optim64_cnn.bit"),
        "metrics_baseline": str(PWD / "metrics" / "baseline"),
        "metrics_quant":    str(PWD / "metrics" / "quant"),
        "metrics_optim":    str(PWD / "metrics" / "optim"),
        "metrics_optim64":  str(PWD / "metrics" / "optim64"),
    }

def main() -> None:
    cd_project()
    p = assemble_paths()

    # Baseline
    log_print("Running inference for baseline configuration...")
    clear_bitfile()
    run_inference(
        p["baseline_bit"],
        metrics_dir=p["metrics_baseline"],
        cycles="auto",
        cycle_type="core",
        # Uncomment if you want to disable power for speed:
        # power_off=True,
    )
    validate_results(metrics_dir=p["metrics_baseline"], clk_mhz=150, exclude_pct=1.0)

    # Quantized
    log_print("Running inference for quantized configuration...")
    clear_bitfile()
    run_inference(
        p["quant_bit"],
        metrics_dir=p["metrics_quant"],
        cycles="auto",
        cycle_type="core",
    )
    validate_results(metrics_dir=p["metrics_quant"], clk_mhz=150, exclude_pct=1.0)

    # Optimized (uint16 stream)
    log_print("Running inference for optimized configuration (optim)...")
    clear_bitfile()
    run_inference(
        p["optim_bit"],
        metrics_dir=p["metrics_optim"],
        cycles="auto",
        cycle_type="core",
    )
    validate_results(metrics_dir=p["metrics_optim"], clk_mhz=200, exclude_pct=1.0)

    # Optimized (packed uint64 stream)
    log_print("Running inference for optimized configuration (optim64)...")
    clear_bitfile()
    run_inference(
        p["optim64_bit"],
        metrics_dir=p["metrics_optim64"],
        pack=True,
        cycles="auto",
        cycle_type="core",
    )
    validate_results(metrics_dir=p["metrics_optim64"], clk_mhz=200, exclude_pct=1.0)

    

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
