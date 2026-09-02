"""Shared constants for the V5 Chatterbox-specialist detector."""

from pathlib import Path

# ── Paths ──
TRAIN_MANIFEST = "v5_kaggle_export/manifests/train_manifest.csv"
VAL_MANIFEST = "v5_kaggle_export/manifests/dev_manifest.csv"
OUTPUT_CHECKPOINT = "best_detector_v5.pth"

# ── Training schedule ──
NUM_EPOCHS = 8
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2            # effective batch = 32
NUM_WORKERS = 4
USE_AMP = True
EARLY_STOP_PATIENCE = 3

# ── Audio contract ──
SAMPLE_RATE = 16000
CLIP_SAMPLES = 64000            # 4 seconds
N_FRAMES = 200                  # 20 ms hop

# ── WavLM ──
WAVLM_PRETRAINED = "microsoft/wavlm-base-plus"
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_LAYERS = [6, 7, 8, 9, 10, 11]

# ── Architecture ──
CROSS_ATTN_HEADS = 8
GAT_HIDDEN = 128
EMBEDDING_DIM = 160
RAW_DIM = 64
MEL_DIM = 64

# ── Attack taxonomy (for evaluation, not training) ──
ATTACK_CLASSES = ["TTS", "CLONE", "VC", "CODEC", "ADVERSARIAL"]
ATTACK_TO_IDX = {name: i for i, name in enumerate(ATTACK_CLASSES)}
IGNORE_INDEX = -100

# ── Evaluation ──
EVAL_BATCH_SIZE = 128
EVAL_CAP_DEFAULT = 4_000
REPORT_GENERATORS = ["chatterbox:clone", "chatterbox:tts"]

# ── Augmentation ──
AUG_NOISE_P = 0.30
AUG_NOISE_SNR_RANGE = (10, 25)  # dB
AUG_GAIN_P = 0.20
AUG_GAIN_DB_RANGE = (-6, 6)
AUG_SPEC_FREQ_MASK_PARAM = 12
AUG_SPEC_TIME_MASK_PARAM = 30

# ── Learning rates (per group) ──
LR_LORA = 1e-5
LR_SSL_HEAD = 3e-5
LR_BRANCHES = 5e-5
LR_CLASSIFIER = 1e-4
WEIGHT_DECAY = 1e-4

# ── W&B ──
WANDB_PROJECT = "telephony-deepfake-detector"
WANDB_RUN_NAME = "v5-chatterbox-specialist"

# ── Dataset builder ──
DATA_ROOT = Path("audio_deepfake_datasets")
CHATTERBOX_ROOT = DATA_ROOT / "Chatterbox"
MINIMAL_EXPORT_ROOT = Path("minimal_export")
MINIMAL_INDEX = MINIMAL_EXPORT_ROOT / "minimal_audio_index.csv"

RANDOM_SEED = 1337
TOTAL_TARGET = 40_000
REAL_FRACTION = 0.50
SPOOF_CLONE_SHARE = 0.65
SPOOF_TTS_SHARE = 0.25
SPOOF_OTHER_SHARE = 0.10
DEV_FRACTION = 0.15

TTS, CLONE, UNLABELED = "TTS", "CLONE", ""

MANIFEST_COLUMNS = [
    "path", "label", "attack_type", "source_dataset", "speaker_id",
    "generator_id", "attack_id", "duration_sec", "sample_rate", "split",
]
