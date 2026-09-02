from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torchaudio

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from transformers import WavLMModel


MODEL_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = Path(
    os.getenv("VOXGUARD_CHECKPOINT_PATH", str(MODEL_ROOT / "best_detector_fixed_v5.pth"))
)
BACKBONE_CHECKPOINT_PATH = Path(
    os.getenv("VOXGUARD_BACKBONE_CHECKPOINT_PATH", str(MODEL_ROOT / "best_telephony_detector.pth"))
)
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 4.0
N_FRAMES = 200

WAVLM_PRETRAINED = "microsoft/wavlm-base-plus"
LORA_LAYERS = [6, 7, 8, 9, 10, 11]
GAT_HIDDEN = 128
EMBEDDING_DIM = 160
RAW_DIM = 64
MEL_DIM = 64


# ---- Dependency-free LoRA, matching peft's get_peft_model key layout so the
#      checkpoint (trained with real peft) loads without pulling in peft itself. ----
class LoraLinear(nn.Module):
    def __init__(self, source: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base_layer = nn.Linear(source.in_features, source.out_features, bias=source.bias is not None)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(source.in_features, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, source.out_features, bias=False)})
        self.scaling = alpha / rank

    @property
    def weight(self) -> Tensor:
        update = self.lora_B["default"].weight @ self.lora_A["default"].weight
        return self.base_layer.weight + update * self.scaling

    @property
    def bias(self) -> Tensor | None:
        return self.base_layer.bias

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight, self.bias)


class LoraWavLM(nn.Module):
    def __init__(self, wavlm: WavLMModel) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = wavlm
        for index in LORA_LAYERS:
            attention = self.base_model.model.encoder.layers[index].attention
            attention.q_proj = LoraLinear(attention.q_proj)
            attention.v_proj = LoraLinear(attention.v_proj)

    def forward(self, waveform: Tensor, output_hidden_states: bool = True):
        return self.base_model.model(waveform, output_hidden_states=output_hidden_states)


# 1. LOG-MEL SPECTRAL FRONT-END
class LogMelFrontEnd(nn.Module):
    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=80, f_min=20.0,
                 f_max=8000.0, n_fft=512, win_length=400, hop_length=160, out_dim=MEL_DIM):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, f_min=f_min, f_max=f_max, n_mels=n_mels, power=2.0,
        )
        self.downsampler = nn.Sequential(
            nn.Conv1d(n_mels, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        mel = self.mel_transform(x)
        mel = torch.log(mel.clamp_min(1e-8))
        if mel.shape[-1] > 2 * N_FRAMES:
            mel = mel[:, :, :2 * N_FRAMES]
        elif mel.shape[-1] < 2 * N_FRAMES:
            mel = F.pad(mel, (0, 2 * N_FRAMES - mel.shape[-1]))
        aligned = self.downsampler(mel)
        return aligned.transpose(1, 2)


# 2. WavLM FRONT-END
class WavLMBranch(nn.Module):
    def __init__(self, pretrained_model_name=WAVLM_PRETRAINED) -> None:
        super().__init__()
        wavlm = WavLMModel.from_pretrained(pretrained_model_name)
        for param in wavlm.parameters():
            param.requires_grad = False
        self.wavlm = LoraWavLM(wavlm)
        self.layer_weights = nn.Parameter(torch.ones(13) / 13)
        self.compressor = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm(128), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        hidden_states = self.wavlm(x, output_hidden_states=True).hidden_states
        norm_weights = F.softmax(self.layer_weights, dim=0)
        weighted = 0
        for state, weight in zip(hidden_states, norm_weights):
            weighted = weighted + (state * weight)
        if weighted.shape[1] != N_FRAMES:
            weighted = weighted.transpose(1, 2)
            weighted = F.interpolate(weighted, size=N_FRAMES, mode="linear", align_corners=False).transpose(1, 2)
        return self.compressor(weighted)


# 3. SINC CONV
class SincConv(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0,
                 sample_rate=SAMPLE_RATE, min_low_hz=50.0, min_band_hz=50.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        mel_low = 2595.0 * math.log10(1.0 + min_low_hz / 700.0)
        mel_high = 2595.0 * math.log10(1.0 + high_hz / 700.0)
        mel_pts = torch.linspace(mel_low, mel_high, out_channels + 1)
        hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)

        self.low_hz_ = nn.Parameter(hz_pts[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter(torch.diff(hz_pts).unsqueeze(1))

        half_k = (kernel_size - 1) // 2
        n = torch.arange(1, half_k + 1, dtype=torch.float32)
        self.register_buffer("n_", n.unsqueeze(0))
        full_window = torch.hamming_window(kernel_size, periodic=False)
        self.register_buffer("window_right_", full_window[half_k + 1:].unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        max_low = self.sample_rate / 2.0 - self.min_band_hz
        low = torch.clamp(self.min_low_hz + torch.abs(self.low_hz_), min=self.min_low_hz, max=max_low)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           min=self.min_low_hz + self.min_band_hz, max=self.sample_rate / 2.0)
        f_low = low / self.sample_rate
        f_high = high / self.sample_rate
        n = self.n_

        sinc_high = torch.sin(2.0 * math.pi * f_high * n) / (math.pi * n)
        sinc_low = torch.sin(2.0 * math.pi * f_low * n) / (math.pi * n)

        bp_right = (sinc_high - sinc_low) * self.window_right_
        bp_center = 2.0 * (f_high - f_low)
        bp_left = torch.flip(bp_right, dims=[1])

        band_pass = torch.cat([bp_left, bp_center, bp_right], dim=1)
        band_pass = band_pass / (band_pass.abs().max(dim=1, keepdim=True).values + 1e-8)

        filters = band_pass.unsqueeze(1)
        return F.conv1d(x, filters, stride=self.stride, padding=self.padding)


# 4. RAW WAVEFORM ENCODER
class RawResidualBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(p=dropout)
        self.shortcut = (nn.Conv1d(channels, channels, kernel_size=1, stride=stride, bias=False)
                         if stride > 1 else nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class RawBranch(nn.Module):
    def __init__(self, out_dim: int = RAW_DIM, n_frames: int = N_FRAMES) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.sinc_stem = SincConv(out_channels=out_dim, kernel_size=251, stride=16, padding=125)
        self.stem_norm = nn.GroupNorm(8, out_dim)
        self.stem_act = nn.LeakyReLU(0.2)
        self.blocks = nn.Sequential(
            RawResidualBlock(out_dim, stride=2), RawResidualBlock(out_dim, stride=2), RawResidualBlock(out_dim, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x.unsqueeze(1)
        h = self.sinc_stem(h)
        h = self.stem_act(self.stem_norm(h))
        h = self.blocks(h)
        h = F.interpolate(h, size=self.n_frames, mode="linear", align_corners=False)
        return h.transpose(1, 2)


# 5. BIDIRECTIONAL CROSS-ATTENTION FUSION
class BidirectionalCrossAttentionFusion(nn.Module):
    def __init__(self, wavlm_dim: int = 128, mel_dim: int = MEL_DIM, num_heads: int = 8) -> None:
        super().__init__()
        self.w2m_k_proj = nn.Linear(mel_dim, wavlm_dim)
        self.w2m_v_proj = nn.Linear(mel_dim, wavlm_dim)
        self.w2m_attn = nn.MultiheadAttention(embed_dim=wavlm_dim, num_heads=num_heads, batch_first=True)
        self.m2w_k_proj = nn.Linear(wavlm_dim, mel_dim)
        self.m2w_v_proj = nn.Linear(wavlm_dim, mel_dim)
        self.m2w_attn = nn.MultiheadAttention(embed_dim=mel_dim, num_heads=4, batch_first=True)
        self.ln_wavlm = nn.LayerNorm(wavlm_dim)
        self.ln_mel = nn.LayerNorm(mel_dim)

    def forward(self, wavlm_tensor: Tensor, mel_tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        wavlm_normed = self.ln_wavlm(wavlm_tensor)
        mel_normed = self.ln_mel(mel_tensor)

        attn_out_w, _ = self.w2m_attn(query=wavlm_normed, key=self.w2m_k_proj(mel_normed), value=self.w2m_v_proj(mel_normed))
        wavlm_enriched = wavlm_tensor + attn_out_w

        attn_out_m, _ = self.m2w_attn(query=mel_normed, key=self.m2w_k_proj(wavlm_normed), value=self.m2w_v_proj(wavlm_normed))
        mel_enriched = mel_tensor + attn_out_m

        fused = torch.cat([wavlm_enriched, mel_enriched], dim=-1)
        return wavlm_enriched, mel_enriched, fused


# 6. AASIST-STYLE GRAPH BACKEND
class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: Tensor) -> Tensor:
        scores = self.sigmoid(self.proj(self.drop(h)))
        _, n_nodes, n_feat = h.size()
        n_keep = max(int(n_nodes * self.k), 1)
        _, idx = torch.topk(scores, n_keep, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        return torch.gather(h, 1, idx)


class HeterogeneousGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj_type_t = nn.Linear(in_dim, in_dim)
        self.proj_type_s = nn.Linear(in_dim, in_dim)
        self.attn_tt_left = nn.Linear(out_dim, 1, bias=False)
        self.attn_tt_right = nn.Linear(out_dim, 1, bias=False)
        self.attn_ss_left = nn.Linear(out_dim, 1, bias=False)
        self.attn_ss_right = nn.Linear(out_dim, 1, bias=False)
        self.attn_cross_left = nn.Linear(out_dim, 1, bias=False)
        self.attn_cross_right = nn.Linear(out_dim, 1, bias=False)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.master_attn_proj = nn.Linear(in_dim, out_dim)
        self.master_attn_weight = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_normal_(self.master_attn_weight)
        self.proj_master_with_att = nn.Linear(in_dim, out_dim)
        self.proj_master_without_att = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.input_drop = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, t_nodes: Tensor, s_nodes: Tensor, master: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        nt = t_nodes.size(1)
        t = self.proj_type_t(t_nodes)
        s = self.proj_type_s(s_nodes)
        x = self.input_drop(torch.cat([t, s], dim=1))
        t_part, s_part = x[:, :nt, :], x[:, nt:, :]

        att_tt = self.leaky_relu(self.attn_tt_left(t_part) + self.attn_tt_right(t_part).transpose(1, 2))
        att_ss = self.leaky_relu(self.attn_ss_left(s_part) + self.attn_ss_right(s_part).transpose(1, 2))
        att_ts = self.leaky_relu(self.attn_cross_left(t_part) + self.attn_cross_right(s_part).transpose(1, 2))
        att_st = self.leaky_relu(self.attn_cross_left(s_part) + self.attn_cross_right(t_part).transpose(1, 2))

        att_top = torch.cat([att_tt, att_ts], dim=2)
        att_bot = torch.cat([att_st, att_ss], dim=2)
        att_full = F.softmax(torch.cat([att_top, att_bot], dim=1), dim=-1)

        x_att = self.proj_with_att(torch.bmm(att_full, x))
        x_out = self.act(self.norm(x_att + self.proj_without_att(x)))

        master_att = torch.tanh(self.master_attn_proj(x * master.expand_as(x)))
        master_att = F.softmax(torch.matmul(master_att, self.master_attn_weight), dim=1)
        master_out = self.proj_master_with_att(torch.matmul(master_att.transpose(1, 2), x)) + \
            self.proj_master_without_att(master)

        return x_out[:, :nt, :], x_out[:, nt:, :], master_out


class AASISTBackend(nn.Module):
    def __init__(self, temporal_in_dim: int = 256, spectral_in_dim: int = MEL_DIM,
                 gat_dim: int = GAT_HIDDEN, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.temporal_proj = nn.Sequential(nn.Linear(temporal_in_dim, gat_dim), nn.LayerNorm(gat_dim), nn.GELU())
        self.spectral_proj = nn.Sequential(nn.Linear(spectral_in_dim, gat_dim), nn.LayerNorm(gat_dim), nn.GELU())
        self.master = nn.Parameter(torch.randn(1, 1, gat_dim))
        nn.init.xavier_normal_(self.master)
        self.hgat1 = HeterogeneousGraphAttentionLayer(gat_dim, gat_dim)
        self.hgat2 = HeterogeneousGraphAttentionLayer(gat_dim, gat_dim)
        self.pool_t1 = GraphPool(k=0.7, in_dim=gat_dim)
        self.pool_s1 = GraphPool(k=0.7, in_dim=gat_dim)
        self.pool_t2 = GraphPool(k=0.5, in_dim=gat_dim)
        self.pool_s2 = GraphPool(k=0.5, in_dim=gat_dim)
        self.readout = nn.Sequential(
            nn.Linear(gat_dim * 5, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU(),
        )

    def forward(self, temporal_input: Tensor, spectral_input: Tensor) -> Tensor:
        t_nodes = self.temporal_proj(temporal_input)
        s_nodes = self.spectral_proj(spectral_input)
        master = self.master.expand(temporal_input.size(0), -1, -1)

        t_nodes, s_nodes, master = self.hgat1(t_nodes, s_nodes, master)
        t_nodes, s_nodes = self.pool_t1(t_nodes), self.pool_s1(s_nodes)
        t_nodes, s_nodes, master = self.hgat2(t_nodes, s_nodes, master)
        t_nodes, s_nodes = self.pool_t2(t_nodes), self.pool_s2(s_nodes)

        t_max, t_mean = torch.max(t_nodes, dim=1)[0], torch.mean(t_nodes, dim=1)
        s_max, s_mean = torch.max(s_nodes, dim=1)[0], torch.mean(s_nodes, dim=1)
        concat = torch.cat([t_max, t_mean, s_max, s_mean, master.squeeze(1)], dim=-1)
        return self.readout(concat)


# 7. UNIFIED MODEL
class TelephonyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel_branch = LogMelFrontEnd()
        self.wavlm_branch = WavLMBranch()
        self.raw_branch = RawBranch()
        self.fusion = BidirectionalCrossAttentionFusion()
        self.aasist_backend = AASISTBackend(temporal_in_dim=128 + MEL_DIM + RAW_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2),
        )

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        mel_feat = self.mel_branch(waveform)
        wavlm_feat = self.wavlm_branch(waveform)
        raw_feat = self.raw_branch(waveform)
        mel_feat = mel_feat.to(wavlm_feat.dtype)

        wavlm_enriched, mel_enriched, _ = self.fusion(wavlm_feat, mel_feat)
        temporal_input = torch.cat([wavlm_enriched, mel_enriched, raw_feat], dim=-1)
        embedding = self.aasist_backend(temporal_input, mel_enriched)
        return self.classifier(embedding), embedding


@dataclass(frozen=True)
class DetectionResult:
    logits: list[float]
    probabilities: list[float]
    embedding_norm: float


class DetectorRuntime:
    def __init__(self) -> None:
        self.model: TelephonyDetector | None = None
        self.metadata: dict[str, object] = {}
        self.label_map: dict[int, str] = {0: "bonafide_real", 1: "spoofed_fake"}
        self.spoof_class_index = 1
        self.uses_external_backbone = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> TelephonyDetector:
        if self.model is not None:
            return self.model
        with self._lock:
            if self.model is not None:
                return self.model
            if not CHECKPOINT_PATH.is_file():
                raise FileNotFoundError(f"Detection checkpoint not found: {CHECKPOINT_PATH}")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
            model_config = checkpoint.get("model_config", {})
            model = TelephonyDetector()
            trained_state = checkpoint["model_state_dict"]

            # The checkpoint stores only the trainable delta (LoRA + custom heads);
            # the frozen WavLM backbone comes from the local mirror of the pretrained
            # weights, same source WavLMModel.from_pretrained would otherwise fetch.
            merged_state = dict(trained_state)
            if BACKBONE_CHECKPOINT_PATH.is_file():
                backbone_checkpoint = torch.load(BACKBONE_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
                backbone_state = {
                    key: value for key, value in backbone_checkpoint["model_state_dict"].items()
                    if key.startswith("wavlm_branch.wavlm.base_model.model.")
                }
                merged_state = {**backbone_state, **trained_state}
                self.uses_external_backbone = True

            missing, unexpected = model.load_state_dict(merged_state, strict=False)
            hard_missing = [key for key in missing if "mel_transform" not in key]
            if hard_missing:
                raise RuntimeError(f"Checkpoint is missing required weights: {hard_missing}")
            # `unexpected` may include an auxiliary domain-adversarial head (da_head.*)
            # present in some training runs; it's inference-time dead weight, ignored.

            model.eval()
            model.to(self.device)
            raw_label_map = model_config.get("label_map", self.label_map)
            self.label_map = {int(index): str(label) for index, label in raw_label_map.items()}
            self.spoof_class_index = next(
                (index for index, label in self.label_map.items() if "spoof" in label.lower()), 1
            )
            self.metadata = {
                key: checkpoint[key]
                for key in ("epoch", "ce_optimal_threshold", "selection_metric", "caveat", "per_generator_eer")
                if key in checkpoint
            }
            self.model = model
        return self.model

    @torch.inference_mode()
    def predict(self, waveform: Tensor) -> DetectionResult:
        model = self.load()
        waveform = waveform.to(self.device)
        with self._inference_lock:
            logits, embedding = model(waveform)
        probabilities = torch.softmax(logits, dim=-1)
        return DetectionResult(
            logits=logits[0].cpu().tolist(),
            probabilities=probabilities[0].cpu().tolist(),
            embedding_norm=float(embedding[0].norm().cpu()),
        )


runtime = DetectorRuntime()
