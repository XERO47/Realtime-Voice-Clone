"""Trains the V5 Chatterbox specialist and checkpoints on overall CE-EER. Run: python train.py"""

import os
import time
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    class _WandbStub:
        def __getattr__(self, name):
            return lambda *a, **k: None
        class Artifact:
            def __init__(self, *a, **k): pass
            def add_file(self, *a, **k): pass
    wandb = _WandbStub()

from config import (
    TRAIN_MANIFEST, VAL_MANIFEST, OUTPUT_CHECKPOINT, NUM_EPOCHS, BATCH_SIZE,
    GRAD_ACCUM_STEPS, NUM_WORKERS, USE_AMP, EARLY_STOP_PATIENCE, SAMPLE_RATE,
    CLIP_SAMPLES, ATTACK_CLASSES, ATTACK_TO_IDX, EVAL_BATCH_SIZE,
    LR_LORA, LR_SSL_HEAD, LR_BRANCHES, LR_CLASSIFIER, WEIGHT_DECAY,
    WANDB_PROJECT, WANDB_RUN_NAME, WAVLM_PRETRAINED, LORA_R, LORA_ALPHA,
    LORA_TARGET_MODULES, LORA_LAYERS, CROSS_ATTN_HEADS, GAT_HIDDEN,
    EMBEDDING_DIM, RAW_DIM, MEL_DIM, AUG_NOISE_P, AUG_GAIN_P,
    AUG_SPEC_FREQ_MASK_PARAM, AUG_SPEC_TIME_MASK_PARAM,
)
from model import DeepfakeDetector
from dataset import AudioDataset, build_sampler
from metrics import compute_eer, per_attack_eer, per_generator_eer, aggregate_by_utterance


def collect_trainable_state(model):
    """Trainable params + their buffers only, so the frozen WavLM backbone isn't saved."""
    named_params = dict(model.named_parameters())
    state = {k: v for k, v in model.state_dict().items()
             if k in named_params and named_params[k].requires_grad}

    trainable_modules = {name for name, mod in model.named_modules()
                         if any(p.requires_grad for p in mod.parameters(recurse=False))}
    for name, buf in model.named_buffers():
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        if parent in trainable_modules:
            state[name] = buf
    return state


def grad_norm(params):
    total = sum(p.grad.data.norm(2).item() ** 2 for p in params if p.grad is not None)
    return total ** 0.5


def train_epoch(model, dataloader, optimizer, scaler, scheduler, ce_criterion,
                device, epoch, clip_params, param_groups_for_logging):
    model.train()
    running_loss, n_batches, n_skipped = 0.0, 0, 0

    accum_loss = torch.tensor(0.0, device=device)
    accum_count = 0

    for batch_idx, (audio, labels, _attacks, _row_idx) in enumerate(dataloader):
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            outputs = model(audio)
            loss = ce_criterion(outputs["logits"], labels) / GRAD_ACCUM_STEPS

        if not torch.isfinite(loss):
            n_skipped += 1
            print(f"  [warn] epoch {epoch} batch {batch_idx}: non-finite loss, skipped")
            continue

        scaler.scale(loss).backward()
        accum_loss += loss.detach()
        accum_count += 1

        if accum_count % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step_loss = accum_loss.item()
            running_loss += step_loss
            n_batches += 1

            log = {
                "train/batch_loss": step_loss,
                "train/lr_lora": optimizer.param_groups[0]['lr'],
                "epoch_progress": epoch + (batch_idx / max(len(dataloader), 1)),
            }
            if n_batches % 200 == 0:
                for name, params in param_groups_for_logging.items():
                    log[f"grad/{name}"] = grad_norm(params)
                emb = outputs["embedding"].float().detach()
                log["feat/embedding_norm"] = emb.norm(dim=-1).mean().item()
                log["feat/embedding_std"] = emb.std().item()
            wandb.log(log)

            if n_batches % 50 == 0:
                print(f"Epoch [{epoch}] Step [{n_batches}] Loss: {step_loss:.4f}")

            accum_loss = torch.tensor(0.0, device=device)

    if n_skipped:
        print(f"  [warn] epoch {epoch}: skipped {n_skipped} non-finite batch(es)")
    return running_loss / max(n_batches, 1)


def validate_epoch(model, dataloader, ce_criterion, device,
                   attack_ids=None, aggregate=False, label="val", generator_ids=None):
    """Returns overall, per-family and per-generator metrics."""
    model.eval()
    val_loss, n_batches = 0.0, 0
    all_labels, all_scores, all_attacks, all_utts = [], [], [], []

    with torch.no_grad():
        for batch_idx, (audio, labels, attacks, utts) in enumerate(dataloader):
            audio = audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(audio)

            ce_loss = ce_criterion(outputs["logits"], labels)
            if torch.isfinite(ce_loss):
                val_loss += ce_loss.item()
                n_batches += 1

            probs = F.softmax(outputs["logits"].float(), dim=-1)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
            all_attacks.extend(attacks.cpu().numpy())
            all_utts.extend(utts.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"  [{label}] batch [{batch_idx}/{len(dataloader)}]")

    avg_loss = val_loss / max(n_batches, 1)
    ids_per_item = attack_ids if attack_ids is not None else ["" for _ in all_labels]
    gens_per_item = generator_ids if generator_ids is not None else ["" for _ in all_labels]

    if aggregate:
        agg_labels, agg_scores, agg_attacks, agg_ids = aggregate_by_utterance(
            all_utts, all_labels, all_scores, all_attacks, ids_per_item)
        _, _, _, agg_gens = aggregate_by_utterance(
            all_utts, all_labels, all_scores, all_attacks, gens_per_item)
    else:
        agg_labels, agg_scores = all_labels, all_scores
        agg_attacks, agg_ids, agg_gens = all_attacks, ids_per_item, gens_per_item

    eer, thresh = compute_eer(agg_labels, agg_scores)
    metrics = {
        "val_loss": avg_loss, "ce_eer": eer, "ce_thresh": thresh,
        "n_windows": len(all_labels), "n_utterances": len(agg_labels),
    }

    labels_np, attacks_np = np.asarray(agg_labels), np.asarray(agg_attacks)
    for name in ATTACK_CLASSES:
        idx = ATTACK_TO_IDX[name]
        present = int(((labels_np == 1) & (attacks_np == idx)).sum())
        metrics[f"{name.lower()}_n"] = present
        metrics[f"{name.lower()}_eer"] = (
            per_attack_eer(agg_labels, agg_scores, agg_attacks, name)
            if present else float("nan"))

    # EER per generator -- chatterbox:clone is the target
    metrics["per_generator"] = {}
    present_gens = sorted({g for g, lb in zip(agg_gens, agg_labels) if lb == 1 and str(g)})
    for gen in present_gens:
        eer_val, n = per_generator_eer(agg_labels, agg_scores, agg_gens, gen)
        if n:
            metrics["per_generator"][gen] = {"eer": eer_val, "n": n}

    return metrics


def print_dashboard(epoch, train_loss, m):
    print(f"\n{'=' * 68}\nEpoch {epoch} Summary\n{'=' * 68}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {m['val_loss']:.4f}")
    print(f"Overall CE-EER: {m['ce_eer'] * 100:.2f}%  (thr {m['ce_thresh']:.4f})   "
          f"<- selection metric")
    if m.get("n_windows") != m.get("n_utterances"):
        print(f"  ({m['n_windows']:,} windows aggregated to {m['n_utterances']:,} utterances)")

    if m.get("per_generator"):
        print("\n  Per-generator EER (vs all real):")
        for gen, info in sorted(m["per_generator"].items()):
            star = "  <- TARGET" if gen == "chatterbox:clone" else ""
            print(f"    {gen:<22}: EER {info['eer'] * 100:6.2f}%   n={info['n']:<7}{star}")

    print("\n  Per-family EER:")
    for name in ATTACK_CLASSES:
        key = name.lower()
        n = m.get(f"{key}_n", 0)
        if n == 0:
            continue
        print(f"    {name:<12}: EER {m[f'{key}_eer'] * 100:6.2f}%   n={n:<7}")

    print("\n  NOTE: dev shares speakers with train by design. This is an overfit "
          "check,\n        not a generalization measurement.")
    print(f"{'=' * 68}\n")


def build_optimizer(model):
    """Per-branch learning rates (LoRA gets the lowest, classifier the highest)."""
    lora_params, ssl_head_params = [], []
    raw_params, mel_params, classifier_params, other_params = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "wavlm" in name and "lora" in name:
            lora_params.append(param)
        elif "wavlm" in name:
            ssl_head_params.append(param)
        elif "raw_branch" in name or "sinc" in name:
            raw_params.append(param)
        elif "mel_branch" in name:
            mel_params.append(param)
        elif "classifier" in name:
            classifier_params.append(param)
        else:
            other_params.append(param)

    def split_decay(params, lr, name):
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        groups = []
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": WEIGHT_DECAY, "name": name})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": f"{name}_nodecay"})
        return groups

    opt_groups = []
    opt_groups += split_decay(lora_params, LR_LORA, "lora")
    opt_groups += split_decay(ssl_head_params, LR_SSL_HEAD, "ssl_head")
    opt_groups += split_decay(raw_params, LR_BRANCHES, "raw")
    opt_groups += split_decay(mel_params, LR_BRANCHES, "mel")
    opt_groups += split_decay(classifier_params, LR_CLASSIFIER, "classifier")
    opt_groups += split_decay(other_params, LR_CLASSIFIER, "other")

    param_groups_for_logging = {
        "wavlm_lora": lora_params, "raw_branch": raw_params,
        "mel_branch": mel_params, "classifier": classifier_params,
    }
    return torch.optim.AdamW(opt_groups), param_groups_for_logging


def run_training_pipeline():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Initializing V5 Chatterbox specialist on {device} ...")
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(TRAIN_MANIFEST):
        raise FileNotFoundError(f"{TRAIN_MANIFEST} not found. Run build_dataset.py first.")
    train_full_df = pd.read_csv(TRAIN_MANIFEST, keep_default_na=False)

    model = DeepfakeDetector().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {trainable:,} trainable / {total:,} total parameters")

    wandb.init(
        project=WANDB_PROJECT, name=WANDB_RUN_NAME,
        config={
            "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM_STEPS, "use_amp": USE_AMP,
            "objective": "overfit Chatterbox clone detection (no generalization)",
            "architecture": "WavLM(LoRA)+LogMel+SincRaw -> BiCrossAttn -> HeteroGAT(160)",
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_layers": LORA_LAYERS,
            "cross_attn_heads": CROSS_ATTN_HEADS, "gat_hidden": GAT_HIDDEN,
            "embedding_dim": EMBEDDING_DIM, "sample_rate": SAMPLE_RATE,
            "augmentation": {"noise_p": AUG_NOISE_P, "gain_p": AUG_GAIN_P,
                             "spec_freq_mask": AUG_SPEC_FREQ_MASK_PARAM,
                             "spec_time_mask": AUG_SPEC_TIME_MASK_PARAM},
            "lr": {"lora": LR_LORA, "ssl_head": LR_SSL_HEAD,
                   "branches": LR_BRANCHES, "classifier": LR_CLASSIFIER},
            "trainable_params": trainable,
        },
    )

    ce_criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    optimizer, param_groups_for_logging = build_optimizer(model)
    clip_params = [p for p in model.parameters() if p.requires_grad]

    total_steps = NUM_EPOCHS * (len(train_full_df) // (BATCH_SIZE * GRAD_ACCUM_STEPS))
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-6 / LR_CLASSIFIER, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_eer = float('inf')
    patience_counter = 0

    val_dataset = AudioDataset(VAL_MANIFEST, train=False, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    val_ids = val_dataset.attack_ids()
    val_gens = val_dataset.generator_ids()
    print(f"Validation set: {len(val_dataset):,} windows")

    train_dataset = AudioDataset(TRAIN_MANIFEST, train=True, augment=True)
    sampler = build_sampler(train_dataset.df)
    print(f"Training set: {len(train_dataset):,} samples")
    spoof_mix = train_dataset.df[train_dataset.df["label"] == 1]["generator_id"].value_counts()
    for gen, n in spoof_mix.items():
        print(f"    {gen:<24} {n:>7,}  ({100 * n / spoof_mix.sum():.1f}% of spoof)")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch} | Training samples: {len(train_dataset):,}")
        t_epoch = time.perf_counter()

        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, scheduler,
                                 ce_criterion, device, epoch, clip_params,
                                 param_groups_for_logging)
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        m = validate_epoch(model, val_loader, ce_criterion, device,
                           attack_ids=val_ids, aggregate=True, label="dev",
                           generator_ids=val_gens)
        t_dev = time.perf_counter() - t0
        t_total = time.perf_counter() - t_epoch

        print_dashboard(epoch, train_loss, m)
        print(f"  Timing: train {t_train:.1f}s | dev {t_dev:.1f}s | epoch {t_total:.1f}s")

        selection_eer = m["ce_eer"]
        clone_info = m.get("per_generator", {}).get("chatterbox:clone")
        clone_eer = clone_info["eer"] if clone_info else float("nan")

        log = {
            "epoch": epoch, "train/epoch_loss": train_loss,
            "time/train_sec": t_train, "time/dev_sec": t_dev, "time/epoch_sec": t_total,
            "val/epoch_loss": m["val_loss"], "val/ce_eer": m["ce_eer"],
            "val/ce_threshold": m["ce_thresh"], "val/chatterbox_clone_eer": clone_eer,
        }
        for name in ATTACK_CLASSES:
            log[f"val/{name.lower()}_eer"] = m[f"{name.lower()}_eer"]
        for gen, info in m.get("per_generator", {}).items():
            log[f"val/gen_{gen.replace(':', '_')}_eer"] = info["eer"]
        wandb.log(log)

        if np.isnan(selection_eer):
            print("Warning: selection metric is NaN, skipping.")
        elif selection_eer < best_eer:
            best_eer = selection_eer
            patience_counter = 0
            print(f"New best CE-EER: {selection_eer * 100:.2f}%  "
                  f"(chatterbox:clone {clone_eer * 100:.2f}%). Saving checkpoint...")

            checkpoint = {
                'epoch': epoch,
                'model_config': {
                    'wavlm_pretrained': WAVLM_PRETRAINED, 'lora_r': LORA_R,
                    'lora_alpha': LORA_ALPHA, 'lora_target_modules': LORA_TARGET_MODULES,
                    'lora_layers': LORA_LAYERS, 'cross_attn_heads': CROSS_ATTN_HEADS,
                    'gat_hidden': GAT_HIDDEN, 'embedding_dim': EMBEDDING_DIM,
                    'raw_dim': RAW_DIM, 'mel_dim': MEL_DIM,
                    'sample_rate': SAMPLE_RATE, 'clip_samples': CLIP_SAMPLES,
                    'attack_classes': ATTACK_CLASSES,
                    'label_map': {0: "bonafide_real", 1: "spoofed_fake"},
                },
                'model_state_dict': collect_trainable_state(model),
                'ce_optimal_threshold': m["ce_thresh"],
                'val_metrics': {k: float(v) for k, v in m.items() if isinstance(v, (int, float))},
                'per_generator_eer': m.get("per_generator", {}),
                'selection_metric': 'overall CE-EER',
                'caveat': ('v5_dataset train/dev share speakers by design; these EERs '
                           'are an overfit check on the Chatterbox generator, NOT a '
                           'generalization measurement'),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            torch.save(checkpoint, OUTPUT_CHECKPOINT)

            artifact = wandb.Artifact(
                name="detector-best-v5", type="model",
                description=f"Best V5 Chatterbox specialist (Epoch {epoch})",
                metadata={"epoch": epoch, "ce_eer": m["ce_eer"], "chatterbox_clone_eer": clone_eer},
            )
            artifact.add_file(OUTPUT_CHECKPOINT)
            wandb.log_artifact(artifact)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: no improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    print("\nTraining complete.")
    print(f"Best dev CE-EER: {best_eer * 100:.2f}%  -> {OUTPUT_CHECKPOINT}")
    print("Reminder: train and dev share speakers by design. This measures how well "
          "the\nmodel memorized the Chatterbox fingerprint, not whether it generalizes.")
    wandb.finish()


if __name__ == "__main__":
    run_training_pipeline()
