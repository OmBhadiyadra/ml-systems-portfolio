"""
Benchmark: single-GPU baseline vs. multi-GPU DDP vs. multi-GPU FSDP.

Runs each configuration as a subprocess, captures its final summary line,
and prints a side-by-side comparison table.

Usage (on a machine with 2 GPUs, e.g. Kaggle's GPU T4 x2 notebook):
    python benchmark_distributed.py --config config.yaml --num_gpus 2
"""

import argparse
import re
import subprocess
import sys

SUMMARY_RE = re.compile(
    r"\[(?P<strategy>\w+)\] DONE world_size=(?P<world_size>\d+) "
    r"total_steps=(?P<steps>\d+) total_time=(?P<time>[\d.]+)s "
    r"throughput=(?P<throughput>[\d.]+) samples/s "
    r"peak_mem_per_gpu=(?P<mem>[\d.]+)MB"
)


def run_and_parse(cmd, label):
    print(f"\n{'=' * 60}\nRunning: {label}\n{'=' * 60}")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-3000:])

    if result.returncode != 0:
        print(f"!!! {label} FAILED (exit code {result.returncode}) !!!")
        print(result.stderr[-3000:])
        return None

    match = SUMMARY_RE.search(result.stdout)
    if not match:
        print(f"!!! Could not parse summary line from {label} output !!!")
        return None
    return match.groupdict()


def main(config_path, num_gpus):
    results = {}

    results["single-gpu"] = run_and_parse(
        [sys.executable, "train_distributed.py", "--config", config_path, "--strategy", "ddp"],
        "Single-GPU baseline (world_size=1)",
    )

    if num_gpus > 1:
        results["ddp"] = run_and_parse(
            [
                "torchrun", f"--nproc_per_node={num_gpus}", "train_distributed.py",
                "--config", config_path, "--strategy", "ddp",
            ],
            f"DDP across {num_gpus} GPUs",
        )
        results["fsdp"] = run_and_parse(
            [
                "torchrun", f"--nproc_per_node={num_gpus}", "train_distributed.py",
                "--config", config_path, "--strategy", "fsdp",
            ],
            f"FSDP across {num_gpus} GPUs",
        )

    print(f"\n{'=' * 60}\nCOMPARISON\n{'=' * 60}")
    print(f"{'Configuration':<16}{'Throughput (samples/s)':<26}{'Peak mem/GPU (MB)':<20}")
    for label, r in results.items():
        if r is None:
            print(f"{label:<16}{'FAILED':<26}{'-':<20}")
        else:
            print(f"{label:<16}{r['throughput']:<26}{r['mem']:<20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num_gpus", type=int, default=2)
    args = parser.parse_args()
    main(args.config, args.num_gpus)
