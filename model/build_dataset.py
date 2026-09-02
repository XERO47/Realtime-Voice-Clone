"""Builds the V5 train/dev manifests: speaker-matched, speaker-shared-with-dev, clone-majority
spoof mix drawn from Chatterbox + real audio + a second spoof corpus. Run: python build_dataset.py
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_ROOT, CHATTERBOX_ROOT, MINIMAL_EXPORT_ROOT, MINIMAL_INDEX,
    RANDOM_SEED, TOTAL_TARGET, REAL_FRACTION, SPOOF_CLONE_SHARE,
    SPOOF_TTS_SHARE, SPOOF_OTHER_SHARE, DEV_FRACTION, TTS, CLONE, UNLABELED,
    MANIFEST_COLUMNS,
)

OUT_ROOT = Path("v5_dataset")
MANIFEST_DIR = OUT_ROOT / "manifests"
METADATA_DIR = OUT_ROOT / "metadata"
REPORT_DIR = OUT_ROOT / "reports"

DO_EXPORT = True
EXPORT_ROOT = Path("v5_kaggle_export")  # flat, hardlinked copy for Kaggle upload


def log(msg: str = "") -> None:
    print(msg, flush=True)


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def make_rows(records: list) -> pd.DataFrame:
    if not records:
        return empty_frame()
    df = pd.DataFrame(records)
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[MANIFEST_COLUMNS]


def speaker_stratified_take(rows: pd.DataFrame, n: int, seed: int) -> list:
    """Draw up to n row indices, round-robin over speakers."""
    if n <= 0 or rows.empty:
        return []
    if n >= len(rows):
        return rows.index.tolist()

    rng = random.Random(seed)
    queues = {}
    for spk, g in rows.groupby("speaker_id", sort=True):
        idx = g.index.tolist()
        rng.shuffle(idx)
        queues[spk] = idx

    speakers = sorted(queues)
    rng.shuffle(speakers)

    picked = []
    while len(picked) < n:
        progressed = False
        for spk in speakers:
            q = queues[spk]
            if not q:
                continue
            picked.append(q.pop())
            progressed = True
            if len(picked) >= n:
                break
        if not progressed:
            break
    return picked


def index_chatterbox_spoof() -> pd.DataFrame:
    """Chatterbox clone + tts spoof rows that actually exist on disk."""
    if not CHATTERBOX_ROOT.exists():
        log(f"  [error] {CHATTERBOX_ROOT} not found.")
        return empty_frame()

    manifests = sorted(CHATTERBOX_ROOT.rglob("*_manifest.csv"))
    if not manifests:
        log("  [error] no Chatterbox manifests found.")
        return empty_frame()

    frames = []
    for mf in manifests:
        try:
            frames.append(pd.read_csv(mf, keep_default_na=False))
        except Exception as e:
            log(f"  [warn] could not read {mf}: {e}")
    if not frames:
        return empty_frame()

    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"] == 1]

    type_map = {"clone": (CLONE, "chatterbox:clone"), "tts": (TTS, "chatterbox:tts")}

    records, n_missing = [], 0
    for _, r in df.iterrows():
        rel = str(r["path"]).replace("\\", "/")
        path = CHATTERBOX_ROOT / rel
        if not os.path.exists(str(path)):
            n_missing += 1
            continue

        stype = str(r.get("synthetic_type", "")).strip().lower()
        if stype not in type_map:
            continue
        family, generator = type_map[stype]
        speaker = str(r.get("speaker_id", "")).strip() or "chatterbox:tts-nospeaker"

        records.append({
            "path": str(path), "label": 1, "attack_type": family,
            "source_dataset": "chatterbox", "speaker_id": speaker,
            "generator_id": generator, "attack_id": stype,
            "duration_sec": r.get("duration_sec", ""), "sample_rate": 16000, "split": "",
        })

    out = make_rows(records)
    if n_missing:
        log(f"  [note] {n_missing:,} Chatterbox manifest rows have no audio on disk; dropped.")
    for gen, n in out["generator_id"].value_counts().items():
        log(f"    {gen}: {n:,} rows")
    return out


def index_real(keep_speakers: set) -> pd.DataFrame:
    """Real audio restricted to speakers that also have a Chatterbox clone."""
    if not MINIMAL_INDEX.exists():
        log(f"  [error] {MINIMAL_INDEX} not found.")
        return empty_frame()

    df = pd.read_csv(MINIMAL_INDEX, keep_default_na=False)
    before = len(df)
    df = df[df["speaker_id"].isin(keep_speakers)]
    log(f"    {len(df):,} of {before:,} real clips belong to a clone speaker")

    records = []
    for _, r in df.iterrows():
        path = MINIMAL_EXPORT_ROOT / str(r["path"])
        if not os.path.exists(str(path)):
            continue
        records.append({
            "path": str(path), "label": 0, "attack_type": UNLABELED,
            "source_dataset": f"real:{r.get('source_corpus', '')}",
            "speaker_id": str(r["speaker_id"]), "generator_id": "", "attack_id": "",
            "duration_sec": r.get("duration_sec", ""),
            "sample_rate": r.get("sample_rate", 16000), "split": "",
        })

    out = make_rows(records)
    log(f"    real rows on disk: {len(out):,} ({out['speaker_id'].nunique()} speakers)")
    return out


def index_other_spoof() -> pd.DataFrame:
    """A second spoof corpus (ASVspoof5, or WaveFake as fallback) so "fake" != "Chatterbox"."""
    root = DATA_ROOT / "ASVspoof_5"
    proto = root / "protocols" / "ASVspoof5.train.tsv"
    audio_dir = root / "train" / "flac_T"

    if proto.exists() and audio_dir.exists():
        records = []
        with open(proto) as fh:
            for line in fh:
                p = line.split()
                if len(p) < 9:
                    continue
                speaker, utt, _gender, _codec, _q, _src, _ac, attack, verdict = p[:9]
                if verdict != "spoof":
                    continue
                path = audio_dir / f"{utt}.flac"
                if not path.exists():
                    continue
                records.append({
                    "path": str(path), "label": 1, "attack_type": CLONE,
                    "source_dataset": "asvspoof5", "speaker_id": f"asvspoof5_other:{speaker}",
                    "generator_id": f"asvspoof5:{attack}", "attack_id": attack,
                    "duration_sec": "", "sample_rate": 16000, "split": "",
                })
        out = make_rows(records)
        if len(out):
            log(f"    ASVspoof5 train spoof: {len(out):,} rows "
                f"({out['generator_id'].nunique()} attacks)")
            return out

    wf = DATA_ROOT / "WaveFake" / "extracted" / "generated_audio"
    if wf.exists():
        records = []
        for sub in sorted(p for p in wf.iterdir() if p.is_dir()):
            vocoder = sub.name
            corpus = "jsut" if vocoder.startswith("jsut") else "ljspeech"
            for f in sub.rglob("*.wav"):
                records.append({
                    "path": str(f), "label": 1, "attack_type": UNLABELED,
                    "source_dataset": "wavefake", "speaker_id": f"wavefake:{corpus}",
                    "generator_id": f"wavefake:{vocoder}", "attack_id": vocoder,
                    "duration_sec": "", "sample_rate": 22050, "split": "",
                })
        out = make_rows(records)
        log(f"    WaveFake (ASVspoof5 unavailable): {len(out):,} rows")
        return out

    log("  [warn] no second spoof corpus found; continuing with Chatterbox only.")
    return empty_frame()


def decode_ok(paths: list, workers: int = 24) -> list:
    """Per path: does the audio actually decode, not merely have a valid header?"""
    import concurrent.futures as cf
    import soundfile as sf

    def _ok(p):
        try:
            sf.read(p, dtype="float32", always_2d=False)
            return True
        except Exception:
            return False

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_ok, paths))


def sample_verified(pool: pd.DataFrame, n: int, seed: int, label: str) -> list:
    """speaker_stratified_take, but only keep rows that really decode."""
    if pool.empty or n <= 0:
        return []

    remaining = pool
    chosen, dropped, rounds = [], 0, 0
    while len(chosen) < n and not remaining.empty and rounds < 5:
        idx = speaker_stratified_take(remaining, n - len(chosen), seed + rounds)
        if not idx:
            break
        ok = decode_ok(remaining.loc[idx, "path"].tolist())
        chosen.extend(i for i, good in zip(idx, ok) if good)
        dropped += sum(1 for good in ok if not good)
        remaining = remaining.drop(index=idx)
        rounds += 1

    if dropped:
        log(f"    [{label}] dropped {dropped:,} undecodable file(s); replaced from the remaining pool")
    return chosen


def split_by_utterance(df: pd.DataFrame, dev_fraction: float, seed: int) -> pd.DataFrame:
    """Hold out a fraction of each speaker's clips for dev -- not speaker-disjoint."""
    df = df.copy()
    df["split"] = "train"
    rng = random.Random(seed)

    dev_idx = []
    for _, g in df.groupby("speaker_id", sort=True):
        idx = g.index.tolist()
        rng.shuffle(idx)
        n_dev = int(round(len(idx) * dev_fraction))
        if n_dev == 0 and len(idx) >= 2:
            n_dev = 1
        dev_idx.extend(idx[:n_dev])

    df.loc[dev_idx, "split"] = "dev"
    return df


def probe_durations(df: pd.DataFrame, workers: int = 16) -> pd.DataFrame:
    """Fill duration_sec so the trainer can tile long dev clips into windows."""
    import concurrent.futures as cf
    import soundfile as sf

    if df.empty:
        return df

    def _probe(path):
        try:
            info = sf.info(path)
            return info.frames / info.samplerate if info.samplerate else 0.0
        except Exception:
            return 0.0

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        durations = list(ex.map(_probe, df["path"].tolist()))

    out = df.copy()
    out["duration_sec"] = durations
    return out


def flatten_name(path: str) -> str:
    """Original path -> unique flat filename."""
    return path.replace("/", "__").replace("\\", "__")


def export_flat(frames: dict) -> None:
    """Hardlink every referenced file into one flat folder for Kaggle upload."""
    audio_dir = EXPORT_ROOT / "audio"
    man_dir = EXPORT_ROOT / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    all_paths = set()
    for df in frames.values():
        all_paths.update(df["path"].tolist())

    flat_map = {p: flatten_name(p) for p in all_paths}
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
        f"{n_missing:,} missing) -> {total_bytes / 1e9:.1f} GB")


def main() -> int:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    for d in (MANIFEST_DIR, METADATA_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    spoof_budget = int(TOTAL_TARGET * (1.0 - REAL_FRACTION))
    real_budget = TOTAL_TARGET - spoof_budget
    want_clone = int(spoof_budget * SPOOF_CLONE_SHARE)
    want_tts = int(spoof_budget * SPOOF_TTS_SHARE)
    want_other = spoof_budget - want_clone - want_tts

    log("[1/5] Indexing Chatterbox spoof ...")
    cb = index_chatterbox_spoof()
    if cb.empty:
        log("ERROR: no Chatterbox spoof audio found.")
        return 1

    clone_pool = cb[cb["generator_id"] == "chatterbox:clone"]
    tts_pool = cb[cb["generator_id"] == "chatterbox:tts"]
    clone_speakers = set(clone_pool["speaker_id"])
    log(f"  clone speakers: {len(clone_speakers)}")

    log("\n[2/5] Indexing real audio for those exact speakers ...")
    real_pool = index_real(clone_speakers)
    if real_pool.empty:
        log("ERROR: no real audio found.")
        return 1

    missing_real = clone_speakers - set(real_pool["speaker_id"])
    if missing_real:
        log(f"  [note] {len(missing_real)} clone speakers have no real audio; "
            "their clone rows are dropped to keep the pairing exact.")
        clone_pool = clone_pool[~clone_pool["speaker_id"].isin(missing_real)]
        clone_speakers -= missing_real

    log("\n[3/5] Indexing second spoof corpus ...")
    other_pool = index_other_spoof()

    log("\n[4/5] Sampling to budget (decode-verifying every file) ...")
    clone_idx = sample_verified(clone_pool, want_clone, RANDOM_SEED + 1, "clone")
    real_idx = sample_verified(real_pool, real_budget, RANDOM_SEED + 2, "real")
    tts_idx = sample_verified(tts_pool, want_tts, RANDOM_SEED + 3, "tts")
    other_idx = (sample_verified(other_pool, want_other, RANDOM_SEED + 4, "other")
                 if not other_pool.empty else [])

    # Re-check the speaker match after verification: a speaker whose real
    # clips all failed to decode would otherwise leave its clones unpaired.
    real_selected_speakers = set(real_pool.loc[real_idx, "speaker_id"])
    unpaired = set(clone_pool.loc[clone_idx, "speaker_id"]) - real_selected_speakers
    if unpaired:
        keep = [i for i in clone_idx if clone_pool.at[i, "speaker_id"] not in unpaired]
        log(f"    [clone] dropped {len(clone_idx) - len(keep):,} rows from "
            f"{len(unpaired)} speaker(s) left with no decodable real audio")
        clone_idx = keep

    parts = [real_pool.loc[real_idx], clone_pool.loc[clone_idx], tts_pool.loc[tts_idx]]
    if other_idx:
        parts.append(other_pool.loc[other_idx])
    master = pd.concat(parts, ignore_index=True)

    for label, got, want in [("real", len(real_idx), real_budget),
                             ("clone", len(clone_idx), want_clone),
                             ("tts", len(tts_idx), want_tts),
                             ("other", len(other_idx), want_other)]:
        flag = "" if got >= want else "   [warn] pool exhausted"
        log(f"    {label:<6} {got:>7,} / {want:,}{flag}")

    log("\n[5/5] Splitting by utterance (speakers SHARED across train/dev) ...")
    master = split_by_utterance(master, DEV_FRACTION, RANDOM_SEED + 5)

    train = master[master["split"] == "train"].sample(
        frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    dev = master[master["split"] == "dev"].reset_index(drop=True)

    log("  probing dev durations (header reads only) ...")
    dev = probe_durations(dev)

    tr_real_spk = set(train[train["label"] == 0]["speaker_id"])
    tr_clone_spk = set(train[train["generator_id"] == "chatterbox:clone"]["speaker_id"])
    only_clone = tr_clone_spk - tr_real_spk
    log(f"\n  train real speakers : {len(tr_real_spk)}")
    log(f"  train clone speakers: {len(tr_clone_spk)}")
    log(f"  clone speakers WITHOUT matching real audio in train: {len(only_clone)}")
    if only_clone:
        log(f"  [warn] e.g. {sorted(only_clone)[:5]}")

    train.to_csv(MANIFEST_DIR / "train_manifest.csv", index=False)
    dev.to_csv(MANIFEST_DIR / "dev_manifest.csv", index=False)

    def by(df, col):
        return {str(k): int(v) for k, v in df[col].value_counts().items()}

    stats = {
        "note": "Speaker-matched and NOT speaker-disjoint. Train and dev share "
                "speakers by design; dev EER is an overfit-check, not a "
                "generalization metric.",
        "totals": {"train": len(train), "dev": len(dev)},
        "train": {
            "real": int((train["label"] == 0).sum()),
            "spoof": int((train["label"] == 1).sum()),
            "by_generator": by(train[train["label"] == 1], "generator_id"),
            "by_family": by(train, "attack_type"),
            "speakers": int(train["speaker_id"].nunique()),
        },
        "dev": {
            "real": int((dev["label"] == 0).sum()),
            "spoof": int((dev["label"] == 1).sum()),
            "by_generator": by(dev[dev["label"] == 1], "generator_id"),
        },
        "speaker_match": {
            "train_real_speakers": len(tr_real_spk),
            "train_clone_speakers": len(tr_clone_spk),
            "clone_speakers_without_real": len(only_clone),
        },
    }
    (REPORT_DIR / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))
    (METADATA_DIR / "build_config.json").write_text(json.dumps({
        "version": "v5-overfit-chatterbox", "random_seed": RANDOM_SEED,
        "total_target": TOTAL_TARGET, "real_fraction": REAL_FRACTION,
        "spoof_shares": {"clone": SPOOF_CLONE_SHARE, "tts": SPOOF_TTS_SHARE,
                          "other": SPOOF_OTHER_SHARE},
        "dev_fraction": DEV_FRACTION,
        "split_policy": "by utterance; speakers intentionally shared train/dev",
        "generalization": "not a goal",
    }, indent=2))

    log(f"\n  train: {len(train):,} rows  "
        f"(real {int((train['label'] == 0).sum()):,} / spoof {int((train['label'] == 1).sum()):,})")
    for gen, n in train[train["label"] == 1]["generator_id"].value_counts().items():
        log(f"      {gen:<24} {n:>7,}  ({100*n/int((train['label']==1).sum()):.1f}% of spoof)")
    log(f"  dev:   {len(dev):,} rows")

    if DO_EXPORT:
        log(f"\n[+] Exporting flat copy -> {EXPORT_ROOT}/ ...")
        export_flat({"train_manifest.csv": train, "dev_manifest.csv": dev})
        shutil.copy2(METADATA_DIR / "build_config.json",
                     EXPORT_ROOT / "manifests" / "build_config.json")

    log(f"\n[+] V5 build complete -> {OUT_ROOT}/")
    log("    Reminder: dev shares speakers with train ON PURPOSE. A low dev EER here "
        "means\n    the model learned Chatterbox's fingerprint, NOT that it generalizes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
