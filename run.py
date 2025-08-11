
#!/usr/bin/env python3
"""
Run the notebook workflow as a single script, with PTY-safe logging that
preserves tqdm progress bars while teeing output to a timestamped log file.
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Literal, Optional
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

def run(cmd: list[str], *, use_pty: bool = True) -> int:
    """
    Run a subprocess and tee its output to console + log.
    - When use_pty=True, attach a PTY to preserve tqdm progress bars.
    - When use_pty=False, use pipes (safer for simple scripts).
    Handles EIO on PTY reads gracefully when the child exits.
    """
    log_print(f"[cmd] {' '.join(cmd)}")
    if use_pty:
        import pty, os, select, sys, errno
        master_fd, slave_fd = pty.openpty()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=None,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
            )
        finally:
            os.close(slave_fd)

        with open(LOG_FILE, "ab", buffering=0) as lf:
            try:
                while True:
                    # If process is done AND no data available, exit
                    if proc.poll() is not None:
                        # Drain any remaining readable data without blocking
                        r, _, _ = select.select([master_fd], [], [], 0)
                        if master_fd not in r:
                            break
                    r, _, _ = select.select([master_fd], [], [], 0.1)
                    if master_fd in r:
                        try:
                            data = os.read(master_fd, 4096)
                        except OSError as e:
                            if e.errno in (errno.EIO, 5):  # Input/output error after child exits
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
        # Non-PTY path (no tqdm animations, but robust)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with open(LOG_FILE, "a", buffering=1) as lf:
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
    # No need for PTY here; avoid EIO cases on very short scripts
    run([sys.executable, "clear_global_state.py"], use_pty=False)

def run_inference(
    bitfile: str,
    metrics_dir: str = "metrics/",
    *,
    pack: bool = False,
    cycles: Literal["auto", "on", "off"] = "auto",
) -> None:
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
    ]
    if pack:
        cmd.append("-pd")
    run(cmd)

def validate_results(
    metrics_dir: str = "metrics/",
    *,
    clk_mhz: Optional[float] = None,
    exclude_pct: float = 0.0,
    show: bool = True,
    cycles_file: str = "cycles_raw.npy",
) -> None:
    mdir = Path(metrics_dir).expanduser().resolve()
    if not mdir.exists():
        raise FileNotFoundError(f"{mdir} does not exist")

    cmd = [
        sys.executable, "validate_results.py",
        "-m", str(mdir),
        "--cycles", cycles_file,
    ]
    if clk_mhz is not None:
        cmd += ["--clk-mhz", str(clk_mhz)]
    if exclude_pct > 0.0:
        cmd += ["--exclude-pct", str(exclude_pct)]
    if not show:
        cmd.append("--no-show")

    run(cmd)

def assemble_paths():
    bitfiles = PWD / "bitfiles"
    return {
        "baseline_bit": str(bitfiles / "baseline" / "baseline_cnn.bit"),
        "quant_bit": str(bitfiles / "quantized" / "quantized_cnn.bit"),
        "optim_bit": str(bitfiles / "optim" / "optim_cnn.bit"),
        "optim64_bit": str(bitfiles / "optim64" / "optim64_cnn.bit"),
        "metrics_baseline": str(PWD / "metrics" / "baseline"),
        "metrics_quant": str(PWD / "metrics" / "quant"),
        "metrics_optim": str(PWD / "metrics" / "optim"),
        "metrics_optim64": str(PWD / "metrics" / "optim64"),
    }

def main() -> None:
    cd_project()
    p = assemble_paths()

    log_print("Running inference for baseline configuration...")
    log_print("Clearing bitfile...")
    clear_bitfile()
    log_print("Running inference...")
    run_inference(p["baseline_bit"], metrics_dir=p["metrics_baseline"])

    log_print("Running inference for quantized configuration...")
    log_print("Clearing bitfile...")
    clear_bitfile()
    log_print("Running inference...")
    run_inference(p["quant_bit"], metrics_dir=p["metrics_quant"])

    validate_results(metrics_dir=p["metrics_baseline"], clk_mhz=150, exclude_pct=1.0)
    validate_results(metrics_dir=p["metrics_quant"], clk_mhz=150, exclude_pct=1.0)

    log_print("Running inference for optimized configuration (optim)...")
    log_print("Clearing bitfile...")
    clear_bitfile()
    log_print("Running inference...")
    run_inference(p["optim_bit"], metrics_dir=p["metrics_optim"])
    validate_results(metrics_dir=p["metrics_optim"], clk_mhz=150, exclude_pct=1.0)

    log_print("Running inference for optimized configuration (optim64)...")
    log_print("Clearing bitfile...")
    clear_bitfile()
    log_print("Running inference...")
    run_inference(p["optim64_bit"], metrics_dir=p["metrics_optim64"], pack=True)
    validate_results(metrics_dir=p["metrics_optim64"], clk_mhz=150, exclude_pct=1.0)

    validate_results(metrics_dir=p["metrics_optim"], clk_mhz=150, exclude_pct=0.0)

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
