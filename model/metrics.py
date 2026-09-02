"""Equal Error Rate and per-attack / per-generator breakdowns."""

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_curve

from config import ATTACK_TO_IDX


def compute_eer(labels, scores):
    """Equal Error Rate + the threshold it occurs at."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    difference = fpr - fnr

    best_idx = np.argmin(np.abs(difference))
    optimal_threshold = float(thresholds[best_idx])
    crossing = np.where(np.diff(np.sign(difference)) != 0)[0]

    if len(crossing) == 0:
        return float((fpr[best_idx] + fnr[best_idx]) / 2.0), optimal_threshold

    i = crossing[0]
    x1, x2 = difference[i], difference[i + 1]
    y1, y2 = fpr[i], fpr[i + 1]
    if x2 == x1:
        return float((fpr[i] + fnr[i]) / 2.0), optimal_threshold

    eer = y1 + (0.0 - x1) * (y2 - y1) / (x2 - x1)
    return float(np.clip(eer, 0.0, 1.0)), optimal_threshold


def per_attack_eer(labels, scores, attacks, attack_name):
    """EER for one attack FAMILY (e.g. TTS, CLONE), scored against all real."""
    labels, scores, attacks = np.asarray(labels), np.asarray(scores), np.asarray(attacks)
    target_idx = ATTACK_TO_IDX[attack_name]
    keep = (labels == 0) | ((labels == 1) & (attacks == target_idx))
    if keep.sum() == 0 or len(np.unique(labels[keep])) < 2:
        return float("nan")
    eer, _ = compute_eer(labels[keep], scores[keep])
    return eer


def per_generator_eer(labels, scores, generator_ids, target_gen):
    """EER for one specific generator (e.g. chatterbox:clone), vs all real."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    generator_ids = np.asarray(generator_ids, dtype=object)

    keep = (labels == 0) | ((labels == 1) & (generator_ids == target_gen))
    n_spoof = int(((labels == 1) & (generator_ids == target_gen)).sum())
    if n_spoof == 0 or len(np.unique(labels[keep])) < 2:
        return float("nan"), 0
    eer, _ = compute_eer(labels[keep], scores[keep])
    return eer, n_spoof


def aggregate_by_utterance(utt_ids, labels, scores, attacks, extra_ids):
    """Collapse per-window scores to one score per utterance (mean)."""
    order = defaultdict(list)
    for i, u in enumerate(utt_ids):
        order[u].append(i)

    out_labels, out_scores, out_attacks, out_ids = [], [], [], []
    for u, idxs in order.items():
        out_scores.append(float(np.mean([scores[i] for i in idxs])))
        out_labels.append(labels[idxs[0]])
        out_attacks.append(attacks[idxs[0]])
        out_ids.append(extra_ids[idxs[0]] if extra_ids is not None else "")
    return out_labels, out_scores, out_attacks, out_ids
