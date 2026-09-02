"""Builds a small held-out eval manifest from Chatterbox audio not used by build_dataset.py.
Not speaker-disjoint from train -- overlap is measured and printed at build time.
Run: python build_eval_dataset.py (after build_dataset.py)
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import build_dataset as base

OUT_ROOT = Path("v5_eval_dataset")
MANIFEST_DIR = OUT_ROOT / "manifests"
REPORT_DIR = OUT_ROOT / "reports"

DO_EXPORT = True
EXPORT_ROOT = Path("v5_eval_kaggle_export")  # kept separate from v5_kaggle_export

RANDOM_SEED = 20240601          
TOTAL_TARGET = 6_000            
REAL_FRACTION = 0.50
SPOOF_CLONE_SHARE = 0.65
SPOOF_TTS_SHARE = 0.25
SPOOF_OTHER_SHARE = 0.10


def log(msg: str = "") -> None:
    base.log(msg)


def already_used_paths() -> set:
    """Every file path build_dataset.py put in train_manifest or dev_manifest."""
    used = set()
    for name in ("train_manifest.csv", "dev_manifest.csv"):
        p = base.MANIFEST_DIR / name
        if not p.exists():
            continue
        df = pd.read_csv(p, keep_default_na=False, usecols=["path"])
        used.update(df["path"])
    return used


def export_flat(frames: dict) -> None:
    """Hardlink referenced files into this script's own export dir (kept separate from build_dataset.py's)."""
    audio_dir = EXPORT_ROOT / "audio"
    man_dir = EXPORT_ROOT / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    all_paths = set()
    for df in frames.values():
        all_paths.update(df["path"].tolist())

    flat_map = {p: base.flatten_name(p) for p in all_paths}
    if len(set(flat_map.values())) != len(flat_map):
        log("  [error] flat-name collision; export aborted.")
        return

    n_new = n_missing = 0
    total_bytes = 0
    for src, flat in flat_map.items():
        dst = audio_dir / flat
        if not os.path.exists(src):
            n_missing += 1
            continue
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            n_new += 1
        total_bytes += dst.stat().st_size

    for name, df in frames.items():
        out = df.copy()
        out["path"] = out["path"].map(lambda p: f"audio/{flat_map[p]}")
        out.to_csv(man_dir / name, index=False)

    log(f"  linked {n_new:,} new files ({len(flat_map):,} total, "
        f"{n_missing:,} missing) -> {total_bytes / 1e9:.2f} GB")


def main() -> int:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    for d in (MANIFEST_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_manifest_path = base.MANIFEST_DIR / "train_manifest.csv"
    if not train_manifest_path.exists():
        log(f"  [error] {train_manifest_path} not found -- run build_dataset.py first.")
        return 1

    used_paths = already_used_paths()
    log(f"[1/4] {len(used_paths):,} paths already used by v5_dataset train+dev")

    log("\n[2/4] Indexing the SAME pools build_dataset.py draws from, minus what it already took ...")
    cb = base.index_chatterbox_spoof()
    cb = cb[~cb["path"].isin(used_paths)]
    clone_pool = cb[cb["generator_id"] == "chatterbox:clone"]
    tts_pool = cb[cb["generator_id"] == "chatterbox:tts"]
    clone_speakers = set(clone_pool["speaker_id"])

    real_pool = base.index_real(clone_speakers)
    real_pool = real_pool[~real_pool["path"].isin(used_paths)]

    other_pool = base.index_other_spoof()
    other_pool = other_pool[~other_pool["path"].isin(used_paths)]

    log(f"  unused: clone {len(clone_pool):,} | tts {len(tts_pool):,} | "
        f"real {len(real_pool):,} | other {len(other_pool):,}")

    spoof_budget = int(TOTAL_TARGET * (1.0 - REAL_FRACTION))
    real_budget = TOTAL_TARGET - spoof_budget
    want_clone = int(spoof_budget * SPOOF_CLONE_SHARE)
    want_tts = int(spoof_budget * SPOOF_TTS_SHARE)
    want_other = spoof_budget - want_clone - want_tts

    log("\n[3/4] Sampling (decode-verifying every file, same as build_dataset.py) ...")
    clone_idx = base.sample_verified(clone_pool, want_clone, RANDOM_SEED + 1, "clone")
    real_idx = base.sample_verified(real_pool, real_budget, RANDOM_SEED + 2, "real")
    tts_idx = base.sample_verified(tts_pool, want_tts, RANDOM_SEED + 3, "tts")
    other_idx = (base.sample_verified(other_pool, want_other, RANDOM_SEED + 4, "other")
                 if not other_pool.empty else [])

    # Same speaker-match invariant as build_dataset.py, re-applied to this set.
    real_selected_speakers = set(real_pool.loc[real_idx, "speaker_id"])
    unpaired = set(clone_pool.loc[clone_idx, "speaker_id"]) - real_selected_speakers
    if unpaired:
        keep = [i for i in clone_idx if clone_pool.at[i, "speaker_id"] not in unpaired]
        log(f"    [clone] dropped {len(clone_idx) - len(keep):,} rows from "
            f"{len(unpaired)} speaker(s) left with no matching real audio")
        clone_idx = keep

    for label, got, want in [("real", len(real_idx), real_budget),
                             ("clone", len(clone_idx), want_clone),
                             ("tts", len(tts_idx), want_tts),
                             ("other", len(other_idx), want_other)]:
        flag = "" if got >= want else "   [warn] pool exhausted"
        log(f"    {label:<6} {got:>6,} / {want:,}{flag}")

    parts = [real_pool.loc[real_idx], clone_pool.loc[clone_idx], tts_pool.loc[tts_idx]]
    if other_idx:
        parts.append(other_pool.loc[other_idx])
    eval_df = pd.concat(parts, ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    log("  probing durations (header reads only) ...")
    eval_df = base.probe_durations(eval_df)

    # Hard check: not one file may be shared with train/dev.
    overlap_paths = set(eval_df["path"]) & used_paths
    if overlap_paths:
        log(f"  [error] {len(overlap_paths)} eval rows share a file with train/dev; aborting.")
        return 1
    log(f"\n  file-level check: 0 of {len(eval_df):,} rows overlap train/dev  (verified)")

    train_df = pd.read_csv(train_manifest_path, keep_default_na=False, usecols=["speaker_id"])
    train_speakers = set(train_df["speaker_id"])
    eval_speakers = set(eval_df["speaker_id"])
    speaker_overlap = eval_speakers & train_speakers
    overlap_pct = 100 * len(speaker_overlap) / max(len(eval_speakers), 1)
    log(f"  speaker overlap with train: {len(speaker_overlap)} / "
        f"{len(eval_speakers)} eval speakers ({overlap_pct:.1f}%)")

    eval_df.to_csv(MANIFEST_DIR / "eval_manifest.csv", index=False)

    def by(df, col):
        return {str(k): int(v) for k, v in df[col].value_counts().items()}

    stats = {
        "note": "Held-out from build_dataset.py's pool (never used in train or dev). "
                "Speakers are NOT disjoint from train -- overlap is measured below.",
        "rows": len(eval_df),
        "real": int((eval_df["label"] == 0).sum()),
        "spoof": int((eval_df["label"] == 1).sum()),
        "by_generator": by(eval_df[eval_df["label"] == 1], "generator_id"),
        "speaker_overlap_with_train": {
            "eval_speakers": len(eval_speakers),
            "overlapping_speakers": len(speaker_overlap),
            "overlap_pct": round(overlap_pct, 1),
        },
        "file_overlap_with_train_or_dev": 0,
    }
    (REPORT_DIR / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))

    log(f"\n  eval_manifest.csv: {len(eval_df):,} rows  (real {stats['real']:,} / spoof {stats['spoof']:,})")
    for gen, n in eval_df[eval_df["label"] == 1]["generator_id"].value_counts().items():
        log(f"      {gen:<24} {n:>6,}")

    if DO_EXPORT:
        log(f"\n[4/4] Exporting flat copy -> {EXPORT_ROOT}/ ...")
        export_flat({"eval_manifest.csv": eval_df})
    else:
        log("\n[4/4] Skipping export (DO_EXPORT=False)")

    log(f"\n[+] V5 eval set complete -> {OUT_ROOT}/")
    log(f"    {len(eval_df):,} rows, 0 files shared with train/dev, "
        f"{overlap_pct:.1f}% speaker overlap with train (expected; see docstring)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
