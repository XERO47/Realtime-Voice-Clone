"""V5 model: WavLM+LogMel+SincRaw -> cross-attention fusion -> AASIST -> binary classifier.
Input is 16 kHz mono, 4 seconds (64000 samples); output is 2-way logits (0=real, 1=fake).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from transformers import WavLMModel
from peft import LoraConfig, get_peft_model

from config import (
    SAMPLE_RATE, N_FRAMES, WAVLM_PRETRAINED, LORA_R, LORA_ALPHA,
    LORA_TARGET_MODULES, LORA_LAYERS, CROSS_ATTN_HEADS, GAT_HIDDEN,
    EMBEDDING_DIM, RAW_DIM, MEL_DIM, AUG_SPEC_FREQ_MASK_PARAM,
    AUG_SPEC_TIME_MASK_PARAM,
)


class LogMelFrontEnd(nn.Module):
    """80-bin log-Mel spectrogram, 20-8000 Hz, with SpecAugment during training."""
    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=80, f_min=20.0,
                 f_max=8000.0, n_fft=512, win_length=400, hop_length=160,
                 out_dim=MEL_DIM):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, f_min=f_min, f_max=f_max, n_mels=n_mels,
            power=2.0,
        )
        self.downsampler = nn.Sequential(
            nn.Conv1d(n_mels, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_dim),
            nn.GELU(),
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=AUG_SPEC_FREQ_MASK_PARAM)
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=AUG_SPEC_TIME_MASK_PARAM)

    def forward(self, x):
        mel = self.mel_transform(x)                  # [B, 80, F]
        mel = torch.log(mel.clamp_min(1e-8))

        if self.training:
            mel = self.freq_masking(mel)
            mel = self.time_masking(mel)

        # Force exactly 2*N_FRAMES frames so stride-2 conv yields N_FRAMES
        if mel.shape[-1] > 2 * N_FRAMES:
            mel = mel[:, :, :2 * N_FRAMES]
        elif mel.shape[-1] < 2 * N_FRAMES:
            mel = F.pad(mel, (0, 2 * N_FRAMES - mel.shape[-1]))

        aligned = self.downsampler(mel)              # [B, 64, 200]
        return aligned.transpose(1, 2)               # [B, 200, 64]


class WavLMFrontEnd(nn.Module):
    """WavLM-base-plus with a frozen backbone and LoRA adapters on the top layers."""
    def __init__(self, pretrained_model_name=WAVLM_PRETRAINED, use_lora=True):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(pretrained_model_name)
        self.wavlm.freeze_feature_encoder()
        for param in self.wavlm.parameters():
            param.requires_grad = False

        if use_lora:
            lora_config = LoraConfig(
                r=LORA_R, lora_alpha=LORA_ALPHA,
                target_modules=LORA_TARGET_MODULES,
                layers_to_transform=LORA_LAYERS,
                lora_dropout=0.1, bias="none",
            )
            self.wavlm = get_peft_model(self.wavlm, lora_config)

        self.layer_weights = nn.Parameter(torch.ones(13) / 13)
        self.compressor = nn.Sequential(
            nn.Linear(768, 128), nn.LayerNorm(128), nn.GELU(),
        )

    def forward(self, x):
        outputs = self.wavlm(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        norm_weights = F.softmax(self.layer_weights, dim=0)

        weighted_representation = 0
        for state, weight in zip(hidden_states, norm_weights):
            weighted_representation = weighted_representation + (state * weight)

        if weighted_representation.shape[1] != N_FRAMES:
            weighted_representation = weighted_representation.transpose(1, 2)
            weighted_representation = F.interpolate(
                weighted_representation, size=N_FRAMES,
                mode='linear', align_corners=False,
            ).transpose(1, 2)

        return self.compressor(weighted_representation)  # [B, 200, 128]


class SincConv(nn.Module):
    """Learnable bandpass filters parameterized by cutoff frequencies (Ravanelli & Bengio, 2018)."""
    def __init__(self, out_channels, kernel_size, stride=1, padding=0,
                 sample_rate=SAMPLE_RATE, min_low_hz=50.0, min_band_hz=50.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
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

    def forward(self, x):
        """x: [B, 1, T] -> [B, out_channels, T']"""
        max_low = self.sample_rate / 2.0 - self.min_band_hz
        low = torch.clamp(self.min_low_hz + torch.abs(self.low_hz_),
                          min=self.min_low_hz, max=max_low)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           min=self.min_low_hz + self.min_band_hz,
                           max=self.sample_rate / 2.0)

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


class RawResidualBlock(nn.Module):
    """Residual block using strided convolution (not MaxPool) for downsampling."""
    def __init__(self, channels, stride=2, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(p=dropout)
        self.shortcut = (nn.Conv1d(channels, channels, kernel_size=1, stride=stride, bias=False)
                         if stride > 1 else nn.Identity())

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class RawWaveformEncoder(nn.Module):
    """64000 -> SincConv(s16) 4000 -> 3x stride(2) 500 -> interp 200."""
    def __init__(self, out_dim=RAW_DIM, n_frames=N_FRAMES):
        super().__init__()
        self.n_frames = n_frames
        self.sinc_stem = SincConv(out_channels=out_dim, kernel_size=251, stride=16, padding=125)
        self.stem_norm = nn.GroupNorm(8, out_dim)
        self.stem_act = nn.LeakyReLU(0.2)
        self.blocks = nn.Sequential(
            RawResidualBlock(out_dim, stride=2),
            RawResidualBlock(out_dim, stride=2),
            RawResidualBlock(out_dim, stride=2),
        )

    def forward(self, x):
        h = x.unsqueeze(1)                          # [B, 1, 64000]
        h = self.sinc_stem(h)                        # [B, 64, 4000]
        h = self.stem_act(self.stem_norm(h))
        h = self.blocks(h)                           # [B, 64, 500]
        h = F.interpolate(h, size=self.n_frames, mode='linear', align_corners=False)
        return h.transpose(1, 2)                     # [B, 200, 64]


class BidirectionalCrossAttentionFusion(nn.Module):
    """Two-way cross-attention: WavLM queries Mel and Mel queries WavLM."""
    def __init__(self, wavlm_dim=128, mel_dim=MEL_DIM, num_heads=CROSS_ATTN_HEADS):
        super().__init__()
        self.w2m_k_proj = nn.Linear(mel_dim, wavlm_dim)
        self.w2m_v_proj = nn.Linear(mel_dim, wavlm_dim)
        self.w2m_attn = nn.MultiheadAttention(embed_dim=wavlm_dim, num_heads=num_heads,
                                              batch_first=True)

        self.m2w_k_proj = nn.Linear(wavlm_dim, mel_dim)
        self.m2w_v_proj = nn.Linear(wavlm_dim, mel_dim)
        self.m2w_attn = nn.MultiheadAttention(embed_dim=mel_dim, num_heads=4, batch_first=True)

        self.ln_wavlm = nn.LayerNorm(wavlm_dim)
        self.ln_mel = nn.LayerNorm(mel_dim)

    def forward(self, wavlm_tensor, mel_tensor):
        """Returns (wavlm_enriched [B,T,128], mel_enriched [B,T,64], fused [B,T,192])."""
        wavlm_normed = self.ln_wavlm(wavlm_tensor)
        mel_normed = self.ln_mel(mel_tensor)

        k1 = self.w2m_k_proj(mel_normed)
        v1 = self.w2m_v_proj(mel_normed)
        attn_out_w, _ = self.w2m_attn(query=wavlm_normed, key=k1, value=v1)
        wavlm_enriched = wavlm_tensor + attn_out_w

        k2 = self.m2w_k_proj(wavlm_normed)
        v2 = self.m2w_v_proj(wavlm_normed)
        attn_out_m, _ = self.m2w_attn(query=mel_normed, key=k2, value=v2)
        mel_enriched = mel_tensor + attn_out_m

        fused = torch.cat([wavlm_enriched, mel_enriched], dim=-1)
        return wavlm_enriched, mel_enriched, fused


class GraphPool(nn.Module):
    """Top-K attention-based graph pooling (from the official AASIST)."""
    def __init__(self, k: float, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, h):
        """h: [B, N, D] -> [B, N', D] where N' = max(int(N * k), 1)"""
        Z = self.drop(h)
        scores = self.sigmoid(self.proj(Z))

        _, n_nodes, n_feat = h.size()
        n_keep = max(int(n_nodes * self.k), 1)

        _, idx = torch.topk(scores, n_keep, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        return torch.gather(h, 1, idx)


class HeterogeneousGraphAttentionLayer(nn.Module):
    """Models T->T, S->S, T->S, S->T interactions plus a learnable master node."""
    def __init__(self, in_dim, out_dim, dropout=0.2):
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

    def forward(self, t_nodes, s_nodes, master):
        Nt = t_nodes.size(1)

        t = self.proj_type_t(t_nodes)
        s = self.proj_type_s(s_nodes)
        x = self.input_drop(torch.cat([t, s], dim=1))

        t_part, s_part = x[:, :Nt, :], x[:, Nt:, :]

        att_tt = self.leaky_relu(self.attn_tt_left(t_part) + self.attn_tt_right(t_part).transpose(1, 2))
        att_ss = self.leaky_relu(self.attn_ss_left(s_part) + self.attn_ss_right(s_part).transpose(1, 2))
        att_ts = self.leaky_relu(self.attn_cross_left(t_part) + self.attn_cross_right(s_part).transpose(1, 2))
        att_st = self.leaky_relu(self.attn_cross_left(s_part) + self.attn_cross_right(t_part).transpose(1, 2))

        att_top = torch.cat([att_tt, att_ts], dim=2)
        att_bot = torch.cat([att_st, att_ss], dim=2)
        att_full = F.softmax(torch.cat([att_top, att_bot], dim=1), dim=-1)

        x_att = self.proj_with_att(torch.bmm(att_full, x))
        x_res = self.proj_without_att(x)
        x_out = self.act(self.norm(x_att + x_res))

        master_expanded = master.expand_as(x)
        master_att = torch.tanh(self.master_attn_proj(x * master_expanded))
        master_att = F.softmax(torch.matmul(master_att, self.master_attn_weight), dim=1)

        master_with_att = self.proj_master_with_att(torch.matmul(master_att.transpose(1, 2), x))
        master_without_att = self.proj_master_without_att(master)
        master_out = master_with_att + master_without_att

        return x_out[:, :Nt, :], x_out[:, Nt:, :], master_out


class AASISTBackend(nn.Module):
    """AASIST-style heterogeneous graph backend over temporal and spectral nodes."""
    def __init__(self, temporal_in_dim=256, spectral_in_dim=MEL_DIM,
                 gat_dim=GAT_HIDDEN, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_in_dim, gat_dim), nn.LayerNorm(gat_dim), nn.GELU())
        self.spectral_proj = nn.Sequential(
            nn.Linear(spectral_in_dim, gat_dim), nn.LayerNorm(gat_dim), nn.GELU())

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

    def forward(self, temporal_input, spectral_input):
        t_nodes = self.temporal_proj(temporal_input)
        s_nodes = self.spectral_proj(spectral_input)

        B = temporal_input.size(0)
        master = self.master.expand(B, -1, -1)

        t_nodes, s_nodes, master = self.hgat1(t_nodes, s_nodes, master)
        t_nodes, s_nodes = self.pool_t1(t_nodes), self.pool_s1(s_nodes)

        t_nodes, s_nodes, master = self.hgat2(t_nodes, s_nodes, master)
        t_nodes, s_nodes = self.pool_t2(t_nodes), self.pool_s2(s_nodes)

        t_max, t_mean = torch.max(t_nodes, dim=1)[0], torch.mean(t_nodes, dim=1)
        s_max, s_mean = torch.max(s_nodes, dim=1)[0], torch.mean(s_nodes, dim=1)

        concat = torch.cat([t_max, t_mean, s_max, s_mean, master.squeeze(1)], dim=-1)
        return self.readout(concat)  # [B, 160]


class DeepfakeDetector(nn.Module):
    """Full V5 model: three feature branches, cross-attention fusion, AASIST, and a binary classifier."""
    def __init__(self):
        super().__init__()
        self.mel_branch = LogMelFrontEnd()
        self.wavlm_branch = WavLMFrontEnd()
        self.raw_branch = RawWaveformEncoder()

        self.fusion = BidirectionalCrossAttentionFusion()
        self.aasist_backend = AASISTBackend(temporal_in_dim=128 + MEL_DIM + RAW_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        """x: [B, 64000] -> dict of outputs."""
        x_audio = x.float()

        with torch.amp.autocast('cuda', enabled=False):
            mel_feat = self.mel_branch(x_audio)                # [B, 200, 64]

        wavlm_feat = self.wavlm_branch(x)                     # [B, 200, 128]
        raw_feat = self.raw_branch(x)                          # [B, 200, 64]
        mel_feat = mel_feat.to(wavlm_feat.dtype)

        wavlm_enriched, mel_enriched, _ = self.fusion(wavlm_feat, mel_feat)
        temporal_input = torch.cat([wavlm_enriched, mel_enriched, raw_feat], dim=-1)

        embedding = self.aasist_backend(temporal_input, mel_enriched)
        logits = self.classifier(embedding)

        return {"logits": logits, "embedding": embedding}
