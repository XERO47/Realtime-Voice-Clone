# V5: Chatterbox-specialist deepfake detector

Deliberately overfits on one voice-cloning generator (Chatterbox) rather than
generalizing across generators. See `build_dataset.py`'s docstring for why.

## Files

| File | Purpose |
|---|---|
| `config.py` | All shared constants (paths, hyperparameters, audio contract). |
| `model.py` | `DeepfakeDetector` -- WavLM+LogMel+SincRaw branches, cross-attention fusion, AASIST backend, binary classifier. |
| `dataset.py` | `AudioDataset` (manifest -> waveform tensors) and the balanced sampler. |
| `metrics.py` | EER, per-attack and per-generator EER, utterance-level aggregation. |
| `build_dataset.py` | Builds `v5_dataset/` + `v5_kaggle_export/` (train/dev manifests) from raw Chatterbox + real-audio sources. |
| `build_eval_dataset.py` | Builds a small held-out eval set from Chatterbox audio *not* used by `build_dataset.py`. |
| `train.py` | Training loop + checkpointing (best `best_detector_v5.pth` by dev CE-EER). |
| `evaluate.py` | Loads a checkpoint and reports real/TTS/clone accuracy on a manifest. |

## Running it

```bash
python build_dataset.py            # builds v5_dataset/ and v5_kaggle_export/
python build_eval_dataset.py       # optional: small held-out spot-check set
python train.py                    # trains, saves best_detector_v5.pth
python evaluate.py                                     # evaluate on dev set
python evaluate.py best_detector_v5.pth v5_eval_kaggle_export/manifests/eval_manifest.csv
```
