"""Benchmarks single-clip inference latency on whatever device is available
(MPS locally, CUDA on Kaggle, CPU as fallback). Same script runs unmodified
in both environments -- it just reports what it finds.

Run:  python benchmark_inference.py [checkpoint_path] [--batch-size N] [--runs N]
"""

import argparse
import statistics
import time

import torch

from config import CLIP_SAMPLES
from model import DeepfakeDetector

DEFAULT_CHECKPOINT = "best_detector_fixed_v5 (1).pth"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_model(checkpoint_path, device):
    model = DeepfakeDetector().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def benchmark(model, device, batch_size, n_warmup, n_runs):
    x = torch.randn(batch_size, CLIP_SAMPLES, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        sync(device)

        times_ms = []
        for _ in range(n_runs):
            sync(device)
            t0 = time.perf_counter()
            model(x)
            sync(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    return times_ms


def report(times_ms, batch_size):
    times_ms.sort()
    mean = statistics.mean(times_ms)
    median = statistics.median(times_ms)
    p95 = times_ms[int(0.95 * len(times_ms)) - 1]
    best, worst = times_ms[0], times_ms[-1]
    per_clip = mean / batch_size

    print(f"  runs: {len(times_ms)}  batch_size: {batch_size}")
    print(f"  mean:   {mean:8.2f} ms/batch   ({per_clip:6.2f} ms/clip)")
    print(f"  median: {median:8.2f} ms")
    print(f"  p95:    {p95:8.2f} ms")
    print(f"  min/max:{best:8.2f} / {worst:.2f} ms")
    print(f"  throughput: {1000.0 * batch_size / mean:.2f} clips/sec")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device.type}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} total parameters")

    print(f"\nBenchmarking on {device.type} "
          f"(warmup={args.warmup}, runs={args.runs}) ...")
    times_ms = benchmark(model, device, args.batch_size, args.warmup, args.runs)
    report(times_ms, args.batch_size)


if __name__ == "__main__":
    main()
