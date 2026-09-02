"""Dataset, path resolution, and sampler for the V5 manifest format."""

import os
import random

import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torchaudio.functional as aF
from torch.utils.data import Dataset, WeightedRandomSampler

from config import (
    SAMPLE_RATE, CLIP_SAMPLES, ATTACK_TO_IDX, IGNORE_INDEX,
    AUG_NOISE_P, AUG_NOISE_SNR_RANGE, AUG_GAIN_P, AUG_GAIN_DB_RANGE,
    EVAL_CAP_DEFAULT,
)


def resolve_path_column(paths, manifest_parent, base_dir=None):
    """Resolve every path in a manifest's `path` column to an on-disk path."""
    sample = [p for p in paths.head(20) if p]

    if sample and all(os.path.isabs(p) for p in sample):
        return paths
    if base_dir is not None:
        return paths.map(lambda p: p if os.path.isabs(p) else os.path.join(base_dir, p))

    candidates = [
        lambda p: p,
        lambda p: os.path.join(manifest_parent, p),
        lambda p: os.path.join(os.path.dirname(manifest_parent), p),
    ]
    for fn in candidates:
        if sample and all(os.path.exists(fn(p)) for p in sample):
            return paths.map(lambda p: p if os.path.isabs(p) else fn(p))

    def _resolve_path(p):
        if os.path.isabs(p) or os.path.exists(p):
            return p
        for cand in (os.path.join(manifest_parent, p),
                     os.path.join(os.path.dirname(manifest_parent), p)):
            if os.path.exists(cand):
                return cand
        return p
    return paths.map(_resolve_path)


def stratified_eval_subset(df, family_caps, default_cap=EVAL_CAP_DEFAULT, seed=42):
    """Cap each attack family's rows for cost-effective, deterministic evaluation."""
    if not family_caps:
        return df.reset_index(drop=True)

    parts = []
    real = df[df["label"] == 0]
    real_cap = family_caps.get("real", default_cap)
    parts.append(real.sample(min(real_cap, len(real)), random_state=seed)
                 if len(real) > real_cap else real)

    spoof = df[df["label"] != 0]
    fam_col = (spoof["attack_type"].astype(str).str.strip()
               if "attack_type" in spoof.columns
               else pd.Series([""] * len(spoof), index=spoof.index))

    for fam, fam_rows in spoof.groupby(fam_col, sort=True):
        cap = family_caps.get(fam, default_cap)
        if len(fam_rows) <= cap:
            parts.append(fam_rows)
            continue

        groups = ([g for _, g in fam_rows.groupby(fam_rows["attack_id"].astype(str), sort=True)]
                  if "attack_id" in fam_rows.columns else [fam_rows])

        take = {i: 0 for i in range(len(groups))}
        remaining, active = cap, list(range(len(groups)))
        while remaining > 0 and active:
            share = max(remaining // len(active), 1)
            progressed = False
            for i in list(active):
                room = len(groups[i]) - take[i]
                grant = min(share, room, remaining)
                if grant <= 0:
                    active.remove(i)
                    continue
                take[i] += grant
                remaining -= grant
                progressed = True
                if take[i] >= len(groups[i]):
                    active.remove(i)
                if remaining <= 0:
                    break
            if not progressed:
                break

        for i, g in enumerate(groups):
            if take[i] >= len(g):
                parts.append(g)
            elif take[i] > 0:
                parts.append(g.sample(take[i], random_state=seed))

    out = pd.concat(parts) if parts else df.iloc[0:0]
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_sampler(df, epoch_size=None):
    """50% real / 50% spoof, keeping the spoof mix exactly as built by the dataset builder."""
    is_real = (df["label"] == 0).values
    n_real = int(is_real.sum())
    n_spoof = len(df) - n_real

    weights = np.zeros(len(df), dtype=np.float64)
    if n_real > 0:
        weights[is_real] = 0.5 / n_real
    if n_spoof > 0:
        weights[~is_real] = 0.5 / n_spoof

    n = epoch_size or len(df)
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=n, replacement=True)


def _tile_to_length(wav, target_len, fade_ms=10.0, sr=SAMPLE_RATE):
    """Loop a short clip to target_len with a crossfaded seam, instead of zero-padding it."""
    n = wav.shape[0]
    if n <= 0:
        return torch.zeros(target_len, dtype=torch.float32)

    fade = min(int(sr * fade_ms / 1000.0), n // 2)
    if fade > 1:
        t = torch.linspace(0.0, 1.0, fade, dtype=wav.dtype)
        unit = wav.clone()
        unit[-fade:] = wav[-fade:] * (1.0 - t) + wav[:fade] * t
    else:
        unit = wav

    reps = -(-target_len // n)  # ceil division
    return unit.repeat(reps)[:target_len]


class AudioDataset(Dataset):
    """Manifest-driven dataset. Returns (waveform, label, attack_idx, row_idx)."""
    def __init__(self, manifest_csv, base_dir=None, fraction=1.0,
                 train=True, multi_window=None, max_windows=4,
                 family_caps=None, augment=True):
        super().__init__()
        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

        full_df = pd.read_csv(manifest_csv, keep_default_na=False)
        manifest_parent = os.path.dirname(os.path.abspath(manifest_csv))
        full_df['path'] = resolve_path_column(full_df['path'], manifest_parent, base_dir)
        if 'attack_type' not in full_df.columns:
            full_df['attack_type'] = ""

        if fraction >= 1.0:
            self.df = full_df.reset_index(drop=True)
        else:
            real_df = full_df[full_df['label'] == 0]
            fake_df = full_df[full_df['label'] == 1]
            n_samples = max(int(min(len(real_df), len(fake_df)) * fraction), 1)
            self.df = pd.concat([
                real_df.sample(min(n_samples, len(real_df)), random_state=42),
                fake_df.sample(min(n_samples, len(fake_df)), random_state=42),
            ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

        if family_caps:
            self.df = stratified_eval_subset(self.df, family_caps)

        self.train = train
        self.augment = augment and train
        self.max_windows = max_windows
        self.multi_window = (not train) if multi_window is None else multi_window

        # --- Expand rows into (row_index, window_index) items ---
        self.items = []
        if self.multi_window and "duration_sec" in self.df.columns:
            durations = pd.to_numeric(self.df["duration_sec"], errors="coerce")
            for i, dur in enumerate(durations):
                n_win = 1
                if pd.notna(dur) and dur > 2 * (CLIP_SAMPLES / SAMPLE_RATE):
                    n_win = min(int(dur // (CLIP_SAMPLES / SAMPLE_RATE)), self.max_windows)
                self.items.extend((i, w) for w in range(max(n_win, 1)))
        else:
            self.items = [(i, 0) for i in range(len(self.df))]

    def __len__(self):
        return len(self.items)

    def _standardize(self, data, sr, train, window=0):
        """-> float32 [64000] mono @ 16 kHz."""
        wav = torch.from_numpy(np.asarray(data, dtype=np.float32))
        if wav.dim() > 1:
            wav = wav.mean(dim=-1)
        if sr != SAMPLE_RATE:
            wav = aF.resample(wav, sr, SAMPLE_RATE)

        n = wav.shape[0]
        if n < CLIP_SAMPLES:
            return _tile_to_length(wav, CLIP_SAMPLES)
        if n == CLIP_SAMPLES:
            return wav

        if train:
            start = torch.randint(0, n - CLIP_SAMPLES + 1, (1,)).item()
        elif n <= 2 * CLIP_SAMPLES:
            start = (n - CLIP_SAMPLES) // 2
        else:
            start = min(window * CLIP_SAMPLES, n - CLIP_SAMPLES)
        return wav[start:start + CLIP_SAMPLES]

    def _augment_waveform(self, wav):
        if not self.augment:
            return wav

        if random.random() < AUG_NOISE_P:
            snr_db = random.uniform(*AUG_NOISE_SNR_RANGE)
            noise = torch.randn_like(wav)
            noise_power = noise.norm()
            if noise_power > 1e-8:
                scale = wav.norm() / (noise_power * (10 ** (snr_db / 20)))
                wav = wav + noise * scale

        if random.random() < AUG_GAIN_P:
            gain_db = random.uniform(*AUG_GAIN_DB_RANGE)
            wav = wav * (10 ** (gain_db / 20))

        return wav

    def __getitem__(self, idx):
        row_idx, window = self.items[idx]
        row = self.df.iloc[row_idx]
        label = int(row['label'])

        attack_idx = ATTACK_TO_IDX.get(str(row.get('attack_type', "")).strip(), IGNORE_INDEX)
        if label == 0:
            attack_idx = IGNORE_INDEX

        try:
            data, sr = sf.read(row['path'], dtype='float32', always_2d=False)
            wav = self._standardize(data, sr, self.train, window)
        except Exception as e:
            print(f"Warning: unreadable file ({row['path']}): {e}")
            wav = torch.zeros(CLIP_SAMPLES, dtype=torch.float32)

        wav = self._augment_waveform(wav)
        return wav, label, attack_idx, row_idx

    def attack_ids(self):
        """Per-item attack_id string, for per-attack EER breakdown."""
        if "attack_id" not in self.df.columns:
            return ["" for _ in self.items]
        col = self.df["attack_id"].astype(str).to_numpy()
        return [col[i] for i, _ in self.items]

    def generator_ids(self):
        """Per-item generator_id string, for per-generator EER breakdown."""
        if "generator_id" not in self.df.columns:
            return ["" for _ in self.items]
        col = self.df["generator_id"].astype(str).to_numpy()
        return [col[i] for i, _ in self.items]
