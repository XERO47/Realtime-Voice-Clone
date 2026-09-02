"""Evaluates a trained V5 checkpoint: real/TTS/clone accuracy at its own threshold.
Run: python evaluate.py [checkpoint_path] [manifest_path]
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import OUTPUT_CHECKPOINT, VAL_MANIFEST, EVAL_BATCH_SIZE, NUM_WORKERS, USE_AMP, ATTACK_TO_IDX
from model import DeepfakeDetector
from dataset import AudioDataset
from metrics import aggregate_by_utterance


def evaluate_model(checkpoint_path=None, manifest_path=None, threshold=None,
                   device=None, batch_size=None):
    """Load a checkpoint and report real/TTS/clone accuracy on a manifest."""
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = checkpoint_path or OUTPUT_CHECKPOINT
    manifest_path = manifest_path or VAL_MANIFEST
    batch_size = batch_size or EVAL_BATCH_SIZE

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = DeepfakeDetector().to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys in checkpoint: {unexpected[:5]}")
    model.eval()

    thr = threshold if threshold is not None else ckpt.get("ce_optimal_threshold", 0.5)
    print(f"Checkpoint epoch {ckpt.get('epoch', '?')}  |  threshold {thr:.4f} "
          f"({'model-chosen' if threshold is None else 'override'})")

    print(f"\nLoading manifest: {manifest_path}")
    ds = AudioDataset(manifest_path, train=False, augment=False)
    gens = ds.generator_ids()
    ids = ds.attack_ids()
    print(f"  {len(ds):,} windows")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    all_labels, all_scores, all_attacks, all_utts = [], [], [], []
    with torch.no_grad():
        for batch_idx, (audio, labels, attacks, utts) in enumerate(loader):
            audio = audio.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = model(audio)
            probs = F.softmax(out["logits"].float(), dim=-1)
            all_labels.extend(labels.numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
            all_attacks.extend(attacks.numpy())
            all_utts.extend(utts.numpy())
            if batch_idx % 20 == 0:
                print(f"  batch [{batch_idx}/{len(loader)}]")

    agg_labels, agg_scores, agg_attacks, _ = aggregate_by_utterance(
        all_utts, all_labels, all_scores, all_attacks, ids)
    _, _, _, agg_gens = aggregate_by_utterance(
        all_utts, all_labels, all_scores, all_attacks, gens)

    labels_np = np.asarray(agg_labels)
    scores_np = np.asarray(agg_scores)
    attacks_np = np.asarray(agg_attacks)
    gens_np = np.asarray(agg_gens, dtype=object)
    preds_np = (scores_np >= thr).astype(int)

    def accuracy(mask, want_pred):
        n = int(mask.sum())
        if n == 0:
            return float("nan"), 0
        correct = int(((preds_np == want_pred) & mask).sum())
        return 100.0 * correct / n, n

    real_mask = labels_np == 0
    tts_mask = (labels_np == 1) & (attacks_np == ATTACK_TO_IDX["TTS"])
    clone_mask = (labels_np == 1) & (attacks_np == ATTACK_TO_IDX["CLONE"])

    real_acc, real_n = accuracy(real_mask, 0)
    tts_acc, tts_n = accuracy(tts_mask, 1)
    clone_acc, clone_n = accuracy(clone_mask, 1)
    overall_acc = 100.0 * float((preds_np == labels_np).mean())

    print(f"\n{'=' * 62}")
    print(f"V5 EVALUATION -- {len(agg_labels):,} utterances (from {len(all_labels):,} windows)")
    print(f"{'=' * 62}")
    print(f"Overall accuracy: {overall_acc:6.2f}%  (n={len(agg_labels):,})\n")
    print(f"  REAL   correctly identified as real: {real_acc:6.2f}%  (n={real_n:,})")
    print(f"  TTS    correctly identified as fake: {tts_acc:6.2f}%  (n={tts_n:,})")
    print(f"  CLONE  correctly identified as fake: {clone_acc:6.2f}%  (n={clone_n:,})")

    print("\n  Per-generator breakdown:")
    spoof_gens = sorted(set(gens_np[labels_np == 1]) - {""})
    for gen in spoof_gens:
        mask = (labels_np == 1) & (gens_np == gen)
        acc, n = accuracy(mask, 1)
        star = "  <- TARGET" if gen == "chatterbox:clone" else ""
        print(f"    {gen:<24} {acc:6.2f}%  (n={n:,}){star}")

    print(f"{'=' * 62}")
    print("NOTE: this manifest's train/dev speakers overlap by design. These numbers "
          "reflect\nrecognition of Chatterbox clones of TRAINED speakers, not "
          "generalization to new\nspeakers or generators.")

    return {
        "overall_accuracy": overall_acc,
        "real_accuracy": real_acc, "real_n": real_n,
        "tts_accuracy": tts_acc, "tts_n": tts_n,
        "clone_accuracy": clone_acc, "clone_n": clone_n,
        "threshold": thr,
        "per_generator": {gen: accuracy((labels_np == 1) & (gens_np == gen), 1)[0]
                         for gen in spoof_gens},
    }


if __name__ == "__main__":
    ckpt_arg = sys.argv[1] if len(sys.argv) > 1 else None
    manifest_arg = sys.argv[2] if len(sys.argv) > 2 else None
    evaluate_model(checkpoint_path=ckpt_arg, manifest_path=manifest_arg)
