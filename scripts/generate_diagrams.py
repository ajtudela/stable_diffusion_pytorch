"""Generate architecture diagrams for the Stable Diffusion implementation.

Creates publication-quality PNG diagrams for:
  1. Overall Stable Diffusion pipeline
  2. U-Net architecture (classic U-shape)
  3. VAE Encoder / Decoder
  4. Transformer / Attention mechanism
  5. UNET Attention Block (Self-Attn → Cross-Attn → FFN)
  6. Residual blocks (VAE and U-Net variants)
  7. CLIP Text Encoder

All images are saved to the ``images/`` directory.
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 10,
    "axes.facecolor": "#fdfdfd",
    "figure.facecolor": "#fdfdfd",
    "savefig.facecolor": "#fdfdfd",
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT_DIR, exist_ok=True)

# Colour palette
C_CLIP = "#4a90d9"
C_CLIP_DARK = "#2e6db4"
C_VAE_ENC = "#e67e22"
C_VAE_DEC = "#27ae60"
C_UNET = "#8e44ad"
C_UNET_ENC = "#9b59b6"
C_UNET_DEC = "#a569bd"
C_UNET_BN = "#6c3483"
C_ATTN = "#2c3e50"
C_CROSS = "#34495e"
C_RESID = "#c0392b"
C_TIME = "#f39c12"
C_SKIP = "#e74c3c"
C_TEXT = "#ffffff"
C_DARK_TEXT = "#2c3e50"
C_LIGHT_BG = "#ecf0f1"
C_ARROW = "#555555"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    color: str,
    text_color: str = C_TEXT,
    fontsize: int = 9,
    radius: float = 0.15,
    lw: float = 1.2,
    edge_color: str | None = None,
    alpha: float = 1.0,
    zorder: int = 3,
) -> FancyBboxPatch:
    """Draw a rounded rectangle with centred text."""
    ec = edge_color or color
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=ec, linewidth=lw,
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=fontsize, color=text_color,
        fontweight="bold", zorder=zorder + 1,
        linespacing=1.3,
    )
    return patch


def _arrow(
    ax: plt.Axes,
    x0: float, y0: float,
    x1: float, y1: float,
    color: str = C_ARROW,
    style: str = "-|>",
    lw: float = 1.5,
    ls: str = "-",
    zorder: int = 2,
    connectionstyle: str = "arc3,rad=0",
    mutation_scale: float = 14,
) -> None:
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=color,
        linewidth=lw, linestyle=ls,
        connectionstyle=connectionstyle,
        mutation_scale=mutation_scale,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _bracket_label(
    ax: plt.Axes,
    x: float,
    y0: float,
    y1: float,
    label: str,
    side: str = "left",
    color: str = C_DARK_TEXT,
    fontsize: int = 9,
) -> None:
    """Draw a vertical bracket with label on the side."""
    offset = -0.3 if side == "left" else 0.3
    ax.annotate(
        "", xy=(x + offset, y1), xytext=(x + offset, y0),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.2),
    )
    ax.text(
        x + offset - (0.15 if side == "left" else -0.15),
        (y0 + y1) / 2, label,
        ha="right" if side == "left" else "left",
        va="center", fontsize=fontsize, color=color,
        fontstyle="italic", rotation=90,
    )


# ===================================================================
# 1. Overall Stable Diffusion Pipeline
# ===================================================================

def draw_pipeline() -> None:
    """High-level pipeline: Text → CLIP → U-Net ← VAE ↔ Image."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 7.5)
    ax.axis("off")
    ax.set_aspect("equal")

    # Title
    ax.text(8, 7.1, "Stable Diffusion — Pipeline Overview",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C_DARK_TEXT)

    # --- CLIP ---
    _box(ax, 2.5, 5.5, 3.8, 1.2, "CLIP Text Encoder\n12× Transformer, 768-d",
         C_CLIP, fontsize=10)
    _box(ax, 2.5, 3.5, 2.0, 0.7, "Tokenizer\n(BPE)", C_CLIP_DARK, fontsize=8)

    # Text prompt
    _box(ax, 2.5, 1.8, 2.2, 0.7, "Text Prompt", C_LIGHT_BG,
         text_color=C_DARK_TEXT, fontsize=9, edge_color="#bbb")
    _arrow(ax, 2.5, 2.15, 2.5, 3.15, color=C_CLIP)
    _arrow(ax, 2.5, 3.85, 2.5, 4.9, color=C_CLIP)

    # Context output label
    ax.text(4.7, 5.0, "(B, 77, 768)", fontsize=7, color=C_CLIP,
            fontstyle="italic")

    # --- VAE Encoder ---
    _box(ax, 8, 1.8, 2.8, 1.0, "VAE Encoder\n3→128→256→512→4ch\n↓8× spatial",
         C_VAE_ENC, fontsize=8)
    _box(ax, 8, 0.3, 2.2, 0.6, "Input Image (optional)",
         C_LIGHT_BG, text_color=C_DARK_TEXT, fontsize=8, edge_color="#bbb")
    _arrow(ax, 8, 0.6, 8, 1.3, color=C_VAE_ENC)

    # --- Noise / Scheduler ---
    _box(ax, 12.5, 1.8, 2.4, 0.7, "Noise  z_T ~ N(0,I)\nor  Scheduler",
         C_LIGHT_BG, text_color=C_DARK_TEXT, fontsize=8, edge_color="#999")

    # --- Diffusion / U-Net (centre) ---
    _box(ax, 8, 4.2, 5.0, 1.6,
         "Diffusion U-Net\nEncoder → Bottleneck → Decoder\n"
         "conditioned on text + time-step",
         C_UNET, fontsize=10)

    # Time embedding
    _box(ax, 13.5, 5.5, 2.4, 0.8, "Time Embedding\nMLP  320→1280", C_TIME,
         text_color=C_DARK_TEXT, fontsize=8)
    _arrow(ax, 12.3, 5.3, 10.5, 4.8, color=C_TIME, ls="--")

    # CLIP → UNet
    _arrow(ax, 4.4, 5.3, 5.5, 4.8, color=C_CLIP, lw=2)

    # VAE Enc → UNet
    _arrow(ax, 8, 2.3, 8, 3.4, color=C_VAE_ENC)

    # Noise → UNet
    _arrow(ax, 12.5, 2.15, 10.2, 3.4, color="#999", ls="--")

    # --- VAE Decoder ---
    _box(ax, 8, 6.3, 2.8, 0.8,
         "VAE Decoder\n4→512→256→128→3ch  ↑8×",
         C_VAE_DEC, fontsize=8)
    _arrow(ax, 8, 5.0, 8, 5.9, color=C_UNET)

    # Output image
    _box(ax, 13.5, 6.3, 2.4, 0.7, "Generated Image",
         C_LIGHT_BG, text_color=C_DARK_TEXT, fontsize=9, edge_color="#bbb")
    _arrow(ax, 9.4, 6.3, 12.3, 6.3, color=C_VAE_DEC, lw=2)

    # Shape labels
    ax.text(8, 3.0, "(B,4,H/8,W/8)", fontsize=7, ha="center",
            color=C_VAE_ENC, fontstyle="italic")
    ax.text(8, 5.5, "(B,4,H/8,W/8)", fontsize=7, ha="center",
            color=C_UNET, fontstyle="italic")

    fig.savefig(os.path.join(OUT_DIR, "pipeline_overview.png"))
    plt.close(fig)
    print("  ✓ pipeline_overview.png")


# ===================================================================
# 2. U-Net Architecture (classic U-shape)
# ===================================================================

def draw_unet() -> None:
    """Classic U-shaped diagram with encoder, bottleneck, decoder."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 11)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(8, 10.5, "U-Net Architecture — Noise Prediction Network",
            ha="center", fontsize=15, fontweight="bold", color=C_DARK_TEXT)

    # Encoder stages (left side, going down)
    enc_stages = [
        ("Conv 4→320\n+ 2×(Res+Attn)\nH/8 · 320ch", 2.0, 8.5, 3.0, 1.2),
        ("2×(Res 320→640\n+ Attn)\nH/16 · 640ch",    2.0, 6.5, 3.0, 1.2),
        ("2×(Res 640→1280\n+ Attn)\nH/32 · 1280ch",  2.0, 4.5, 3.0, 1.2),
        ("2×ResBlock\nH/64 · 1280ch",                 2.0, 2.5, 3.0, 1.0),
    ]

    # Decoder stages (right side, going up)
    dec_stages = [
        ("2×Res + Upsample\nH/64→H/32 · 1280ch",           14.0, 2.5, 3.2, 1.0),
        ("2×(Res+Attn)\n+ Upsample\nH/32→H/16 · 1280ch",   14.0, 4.5, 3.2, 1.2),
        ("2×(Res+Attn)\n+ Upsample\nH/16→H/8 · 640ch",     14.0, 6.5, 3.2, 1.2),
        ("3×(Res+Attn)\nH/8 · 320ch",                       14.0, 8.5, 3.2, 1.0),
    ]

    # Draw encoder
    for i, (label, x, y, w, h) in enumerate(enc_stages):
        _box(ax, x, y, w, h, label, C_UNET_ENC, fontsize=8)
        if i < len(enc_stages) - 1:
            # Downsample arrow
            _arrow(ax, x, y - h / 2, x, enc_stages[i + 1][2] + enc_stages[i + 1][4] / 2,
                   color=C_UNET, lw=2)
            ax.text(x + 1.7, (y - h / 2 + enc_stages[i + 1][2] + enc_stages[i + 1][4] / 2) / 2,
                    "↓ stride=2", fontsize=7, color=C_UNET, ha="center")

    # Draw decoder
    for i, (label, x, y, w, h) in enumerate(dec_stages):
        _box(ax, x, y, w, h, label, C_UNET_DEC, fontsize=8)
        if i < len(dec_stages) - 1:
            _arrow(ax, x, y + h / 2, x, dec_stages[i + 1][2] - dec_stages[i + 1][4] / 2,
                   color=C_UNET, lw=2)

    # Bottleneck
    _box(ax, 8, 1.0, 4.5, 1.0,
         "Bottleneck: ResBlock → Attention(8h,160) → ResBlock\n1280ch · H/64",
         C_UNET_BN, fontsize=9)

    # Encoder → Bottleneck
    _arrow(ax, 2.0, 2.0, 5.75, 1.3, color=C_UNET, lw=2)
    # Bottleneck → Decoder
    _arrow(ax, 10.25, 1.3, 14.0, 2.0, color=C_UNET, lw=2)

    # Skip connections (horizontal dashed)
    skip_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
    skip_labels = ["skip 1280ch", "skip 1280ch", "skip 640ch", "skip 320ch"]
    for i in range(4):
        ey = enc_stages[i][2]
        dy = dec_stages[3 - i][2]
        ex = enc_stages[i][1] + enc_stages[i][3] / 2
        dx = dec_stages[3 - i][1] - dec_stages[3 - i][3] / 2
        y_mid = ey
        _arrow(ax, ex, y_mid, dx, dy,
               color=skip_colors[i], ls="--", lw=1.8,
               connectionstyle="arc3,rad=0.0")
        ax.text((ex + dx) / 2, (y_mid + dy) / 2 + 0.25,
                skip_labels[i], fontsize=7, ha="center",
                color=skip_colors[i], fontstyle="italic",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    # Input / Output
    _box(ax, 2.0, 9.8, 2.4, 0.5, "Noisy Latent (B,4,H/8,W/8)",
         C_LIGHT_BG, text_color=C_DARK_TEXT, fontsize=8, edge_color="#aaa")
    _arrow(ax, 2.0, 9.55, 2.0, 9.1, color=C_ARROW)

    _box(ax, 14.0, 9.8, 2.8, 0.5, "Predicted Noise (B,4,H/8,W/8)",
         C_LIGHT_BG, text_color=C_DARK_TEXT, fontsize=8, edge_color="#aaa")
    _arrow(ax, 14.0, 9.0, 14.0, 9.55, color=C_ARROW)

    # Output Layer
    _box(ax, 14.0, 9.35, 2.4, 0.35, "GN→SiLU→Conv 320→4",
         "#7f8c8d", fontsize=7)

    # Conditioning labels
    _box(ax, 8.0, 9.8, 2.8, 0.5, "CLIP Context (B,77,768)",
         C_CLIP, fontsize=8)
    _arrow(ax, 7.0, 9.55, 3.5, 9.1, color=C_CLIP, ls="--", lw=1.2)
    _arrow(ax, 9.0, 9.55, 12.4, 9.1, color=C_CLIP, ls="--", lw=1.2)
    ax.text(5.0, 9.5, "cross-attn", fontsize=7,
            color=C_CLIP, fontstyle="italic")
    ax.text(11.0, 9.5, "cross-attn", fontsize=7,
            color=C_CLIP, fontstyle="italic")

    _box(ax, 8.0, 0.0, 2.4, 0.45, "Time Emb (B,1280)",
         C_TIME, text_color=C_DARK_TEXT, fontsize=8)
    _arrow(ax, 6.8, 0.15, 3.5, 2.0, color=C_TIME, ls="--", lw=1.2)
    _arrow(ax, 9.2, 0.15, 12.4, 2.0, color=C_TIME, ls="--", lw=1.2)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=C_UNET_ENC, label="Encoder Stage"),
        mpatches.Patch(facecolor=C_UNET_DEC, label="Decoder Stage"),
        mpatches.Patch(facecolor=C_UNET_BN, label="Bottleneck"),
        mpatches.Patch(facecolor=C_CLIP, label="CLIP Conditioning"),
        mpatches.Patch(facecolor=C_TIME, label="Time Conditioning"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8,
              framealpha=0.9, edgecolor="#ccc")

    fig.savefig(os.path.join(OUT_DIR, "unet_architecture.png"))
    plt.close(fig)
    print("  ✓ unet_architecture.png")


# ===================================================================
# 3. VAE Encoder & Decoder
# ===================================================================

def draw_vae() -> None:
    """VAE encoder (left) and decoder (right) side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # --- Encoder ---
    ax = axes[0]
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 13)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("VAE Encoder  (image → latent)", fontsize=13,
                 fontweight="bold", color=C_VAE_ENC, pad=12)

    enc_blocks: list[tuple[str, str, float]] = [
        ("Input Image\n(B, 3, H, W)", C_LIGHT_BG, 12.0),
        ("Conv2d 3→128", C_VAE_ENC, 11.0),
        ("ResBlock(128,128) ×2", C_VAE_ENC, 10.0),
        ("↓ Conv stride=2", "#d35400", 9.2),
        ("ResBlock(128→256)\nResBlock(256,256)", C_VAE_ENC, 8.2),
        ("↓ Conv stride=2", "#d35400", 7.2),
        ("ResBlock(256→512)\nResBlock(512,512)", C_VAE_ENC, 6.2),
        ("↓ Conv stride=2", "#d35400", 5.2),
        ("ResBlock(512,512) ×3", C_VAE_ENC, 4.2),
        ("Self-Attention(512)", C_ATTN, 3.2),
        ("ResBlock(512,512)", C_VAE_ENC, 2.4),
        ("GN(32) → SiLU", "#7f8c8d", 1.6),
        ("Conv 512→8 → Conv 1×1", C_VAE_ENC, 0.8),
        ("Split μ + log σ²\nz = μ + σ·ε\n× 0.18215", C_RESID, -0.3),
    ]

    bw, bh = 3.5, 0.55
    cx = 3.0
    for i, (label, color, y) in enumerate(enc_blocks):
        tc = C_DARK_TEXT if color == C_LIGHT_BG else C_TEXT
        h = bh if "\n" not in label else bh + 0.25
        _box(ax, cx, y, bw, h, label, color, text_color=tc, fontsize=8)
        if i < len(enc_blocks) - 1:
            next_y = enc_blocks[i + 1][2]
            _arrow(ax, cx, y - h / 2, cx, next_y + (bh if "\n" not in enc_blocks[i + 1][0] else bh + 0.25) / 2,
                   color=C_ARROW)

    # Resolution annotations
    res_labels = [
        (5.2, 11.0, "H × W"),
        (5.2, 8.2, "H/2 × W/2"),
        (5.2, 6.2, "H/4 × W/4"),
        (5.2, 4.2, "H/8 × W/8"),
    ]
    for rx, ry, rl in res_labels:
        ax.text(rx, ry, rl, fontsize=7, color="#888", fontstyle="italic",
                ha="left", va="center")

    _box(ax, cx, -1.3, 2.8, 0.45, "Latent z  (B, 4, H/8, W/8)",
         C_LIGHT_BG, text_color=C_DARK_TEXT, fontsize=8, edge_color="#aaa")
    _arrow(ax, cx, -0.65, cx, -1.07, color=C_ARROW)

    # --- Decoder ---
    ax = axes[1]
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 13)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("VAE Decoder  (latent → image)", fontsize=13,
                 fontweight="bold", color=C_VAE_DEC, pad=12)

    dec_blocks: list[tuple[str, str, float]] = [
        ("Latent z / 0.18215\n(B, 4, H/8, W/8)", C_LIGHT_BG, 12.0),
        ("Conv 4→4 (1×1)\nConv 4→512", C_VAE_DEC, 11.0),
        ("ResBlock(512,512)", C_VAE_DEC, 10.1),
        ("Self-Attention(512)", C_ATTN, 9.3),
        ("ResBlock(512,512) ×4", C_VAE_DEC, 8.4),
        ("↑ Upsample ×2 + Conv", "#1e8449", 7.5),
        ("ResBlock(512,512) ×3", C_VAE_DEC, 6.6),
        ("↑ Upsample ×2 + Conv", "#1e8449", 5.7),
        ("ResBlock(512→256)\nResBlock(256,256) ×2", C_VAE_DEC, 4.7),
        ("↑ Upsample ×2 + Conv", "#1e8449", 3.7),
        ("ResBlock(256→128)\nResBlock(128,128) ×2", C_VAE_DEC, 2.7),
        ("GN(32) → SiLU", "#7f8c8d", 1.8),
        ("Conv 128→3", C_VAE_DEC, 1.0),
    ]

    for i, (label, color, y) in enumerate(dec_blocks):
        tc = C_DARK_TEXT if color == C_LIGHT_BG else C_TEXT
        h = bh if "\n" not in label else bh + 0.25
        _box(ax, cx, y, bw, h, label, color, text_color=tc, fontsize=8)
        if i < len(dec_blocks) - 1:
            next_y = dec_blocks[i + 1][2]
            _arrow(ax, cx, y - h / 2, cx, next_y + (bh if "\n" not in dec_blocks[i + 1][0] else bh + 0.25) / 2,
                   color=C_ARROW)

    res_labels2 = [
        (5.2, 10.1, "H/8 × W/8"),
        (5.2, 6.6, "H/4 × W/4"),
        (5.2, 4.7, "H/2 × W/2"),
        (5.2, 2.7, "H × W"),
    ]
    for rx, ry, rl in res_labels2:
        ax.text(rx, ry, rl, fontsize=7, color="#888", fontstyle="italic",
                ha="left", va="center")

    _box(ax, cx, 0.1, 2.8, 0.45, "RGB Image  (B, 3, H, W)",
         C_LIGHT_BG, text_color=C_DARK_TEXT, fontsize=8, edge_color="#aaa")
    _arrow(ax, cx, 0.73, cx, 0.33, color=C_ARROW)

    fig.tight_layout(pad=2)
    fig.savefig(os.path.join(OUT_DIR, "vae_encoder_decoder.png"))
    plt.close(fig)
    print("  ✓ vae_encoder_decoder.png")


# ===================================================================
# 4. Attention Mechanisms
# ===================================================================

def draw_attention() -> None:
    """Self-Attention & Cross-Attention diagrams side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    # --- Self-Attention ---
    ax = axes[0]
    ax.set_xlim(-1, 9)
    ax.set_ylim(-0.5, 10)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("Multi-Head Self-Attention", fontsize=13,
                 fontweight="bold", color=C_ATTN, pad=12)

    sa_blocks = [
        ("Input x\n(B, SeqLen, d_embed)", C_LIGHT_BG, 9.0),
        ("Linear  →  3·d_embed\n(fused Q, K, V projection)", C_ATTN, 7.8),
        ("Split → Q, K, V", C_ATTN, 6.8),
        ("Reshape → (B, n_heads, Seq, d_head)", "#5d6d7e", 5.9),
        ("Attention Scores\nQ · Kᵀ  /  √d_head", C_RESID, 4.8),
        ("Causal Mask (optional)\n-∞ for future positions", "#7f8c8d", 3.8),
        ("Softmax → Weights", C_ATTN, 2.9),
        ("Weighted Sum:  W · V", C_ATTN, 2.1),
        ("Merge Heads → (B, Seq, d_embed)", "#5d6d7e", 1.3),
        ("Output Projection\nLinear → d_embed", C_ATTN, 0.3),
    ]

    cx, bw, bh = 4.0, 4.3, 0.55
    for i, (label, color, y) in enumerate(sa_blocks):
        tc = C_DARK_TEXT if color == C_LIGHT_BG else C_TEXT
        h = bh if "\n" not in label else bh + 0.25
        _box(ax, cx, y, bw, h, label, color, text_color=tc, fontsize=8)
        if i < len(sa_blocks) - 1:
            next_y = sa_blocks[i + 1][2]
            nh = bh if "\n" not in sa_blocks[i + 1][0] else bh + 0.25
            _arrow(ax, cx, y - h / 2, cx, next_y + nh / 2, color=C_ARROW)

    # --- Cross-Attention ---
    ax = axes[1]
    ax.set_xlim(-1, 9)
    ax.set_ylim(-0.5, 10)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("Multi-Head Cross-Attention", fontsize=13,
                 fontweight="bold", color=C_CROSS, pad=12)

    # Two inputs
    _box(ax, 2.0, 9.0, 3.0, 0.65, "Queries x (image)\n(B, H·W, d_embed)",
         C_UNET, fontsize=8)
    _box(ax, 6.5, 9.0, 3.0, 0.65, "Context y (text)\n(B, 77, 768)",
         C_CLIP, fontsize=8)

    # Projections
    _box(ax, 2.0, 7.8, 2.0, 0.5, "Q = W_q · x", C_CROSS, fontsize=8)
    _box(ax, 5.0, 7.8, 2.0, 0.5, "K = W_k · y", C_CROSS, fontsize=8)
    _box(ax, 7.5, 7.8, 2.0, 0.5, "V = W_v · y", C_CROSS, fontsize=8)

    _arrow(ax, 2.0, 8.67, 2.0, 8.05, color=C_ARROW)
    _arrow(ax, 5.5, 8.67, 5.0, 8.05, color=C_ARROW)
    _arrow(ax, 7.2, 8.67, 7.5, 8.05, color=C_ARROW)

    # Reshape
    _box(ax, 4.5, 6.7, 5.5, 0.5,
         "Reshape → (B, n_heads, Seq, d_head)", "#5d6d7e", fontsize=8)
    _arrow(ax, 2.0, 7.55, 3.0, 6.95, color=C_ARROW)
    _arrow(ax, 5.0, 7.55, 4.5, 6.95, color=C_ARROW)
    _arrow(ax, 7.5, 7.55, 6.0, 6.95, color=C_ARROW)

    ca_rest = [
        ("Attention Scores\nQ · Kᵀ  /  √d_head", C_RESID, 5.5),
        ("Softmax → Weights", C_CROSS, 4.5),
        ("Weighted Sum:  W · V", C_CROSS, 3.6),
        ("Merge Heads", "#5d6d7e", 2.8),
        ("Output Projection\nLinear → d_embed", C_CROSS, 1.8),
        ("Output\n(B, SeqLen_Q, d_embed)", C_LIGHT_BG, 0.6),
    ]
    for i, (label, color, y) in enumerate(ca_rest):
        tc = C_DARK_TEXT if color == C_LIGHT_BG else C_TEXT
        h = bh if "\n" not in label else bh + 0.25
        _box(ax, 4.5, y, 4.3, h, label, color, text_color=tc, fontsize=8)
        if i == 0:
            _arrow(ax, 4.5, 6.45, 4.5, y + h / 2, color=C_ARROW)
        if i < len(ca_rest) - 1:
            ny = ca_rest[i + 1][2]
            nh = bh if "\n" not in ca_rest[i + 1][0] else bh + 0.25
            _arrow(ax, 4.5, y - h / 2, 4.5, ny + nh / 2, color=C_ARROW)

    # Label: "no causal mask"
    ax.text(7.5, 4.9, "no causal mask\n(all tokens visible)",
            fontsize=7, color="#888", fontstyle="italic", ha="center")

    fig.tight_layout(pad=2)
    fig.savefig(os.path.join(OUT_DIR, "attention_mechanisms.png"))
    plt.close(fig)
    print("  ✓ attention_mechanisms.png")


# ===================================================================
# 5. UNET Attention Block
# ===================================================================

def draw_unet_attention_block() -> None:
    """Self-Attn → Cross-Attn → GeGLU FFN with residual connections."""
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 15)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.set_title("UNET Attention Block  (unet_utils.py)", fontsize=14,
                 fontweight="bold", color=C_UNET, pad=10)

    cx = 5.0
    bw = 4.0

    blocks = [
        ("Feature Map  (B, C, H, W)", C_LIGHT_BG, 14.0, C_DARK_TEXT),
        ("GroupNorm(32, C)", "#7f8c8d", 13.1, C_TEXT),
        ("Conv 1×1  (channel proj)", C_UNET, 12.3, C_TEXT),
        ("Flatten → (B, H·W, C)", "#5d6d7e", 11.5, C_TEXT),
    ]

    for i, (lbl, col, y, tc) in enumerate(blocks):
        _box(ax, cx, y, bw, 0.5, lbl, col, text_color=tc, fontsize=9)
        if i < len(blocks) - 1:
            _arrow(ax, cx, y - 0.25, cx,
                   blocks[i + 1][2] + 0.25, color=C_ARROW)

    # --- Self-Attention sub-block ---
    y_sa_top = 10.5
    rect = FancyBboxPatch((1.0, 7.7), 8.0, 3.2,
                          boxstyle="round,pad=0.1,rounding_size=0.2",
                          facecolor=C_ATTN, alpha=0.08, edgecolor=C_ATTN,
                          linewidth=1.3, linestyle="--", zorder=1)
    ax.add_patch(rect)
    ax.text(1.3, 10.7, "Self-Attention", fontsize=9, color=C_ATTN,
            fontweight="bold")

    sa_items = [
        ("LayerNorm", "#5d6d7e", y_sa_top),
        ("Multi-Head Self-Attention  (8 heads)", C_ATTN, 9.6),
        ("(+) Residual", C_RESID, 8.8),
    ]
    _arrow(ax, cx, 11.25, cx, y_sa_top + 0.25, color=C_ARROW)
    for i, (lbl, col, y) in enumerate(sa_items):
        _box(ax, cx, y, bw - 0.4, 0.5, lbl, col, fontsize=8)
        if i < len(sa_items) - 1:
            _arrow(ax, cx, y - 0.25, cx,
                   sa_items[i + 1][2] + 0.25, color=C_ARROW)
    # Residual skip
    _arrow(ax, 1.5, 11.25, 1.5, 8.8, color=C_RESID, ls="--", lw=1.2)
    _arrow(ax, 1.5, 8.8, cx - bw / 2 + 0.2,
           8.8, color=C_RESID, ls="--", lw=1.2)

    # --- Cross-Attention sub-block ---
    rect2 = FancyBboxPatch((1.0, 5.0), 8.0, 3.2,
                           boxstyle="round,pad=0.1,rounding_size=0.2",
                           facecolor=C_CLIP, alpha=0.08, edgecolor=C_CLIP,
                           linewidth=1.3, linestyle="--", zorder=1)
    ax.add_patch(rect2)
    ax.text(1.3, 8.0, "Cross-Attention", fontsize=9, color=C_CLIP,
            fontweight="bold")

    ca_items = [
        ("LayerNorm", "#5d6d7e", 7.6),
        ("Multi-Head Cross-Attention  (8 heads)", C_CLIP_DARK, 6.7),
        ("(+) Residual", C_RESID, 5.9),
    ]
    _arrow(ax, cx, 8.55, cx, 7.85, color=C_ARROW)
    for i, (lbl, col, y) in enumerate(ca_items):
        _box(ax, cx, y, bw - 0.4, 0.5, lbl, col, fontsize=8)
        if i < len(ca_items) - 1:
            _arrow(ax, cx, y - 0.25, cx,
                   ca_items[i + 1][2] + 0.25, color=C_ARROW)

    # Context input
    _box(ax, 9.5, 6.7, 1.5, 0.5, "CLIP\nContext", C_CLIP, fontsize=7)
    _arrow(ax, 8.75, 6.7, cx + bw / 2 - 0.2, 6.7, color=C_CLIP, ls="--")

    # Residual skip
    _arrow(ax, 1.5, 8.55, 1.5, 5.9, color=C_RESID, ls="--", lw=1.2)
    _arrow(ax, 1.5, 5.9, cx - bw / 2 + 0.2,
           5.9, color=C_RESID, ls="--", lw=1.2)

    # --- GeGLU FFN sub-block ---
    rect3 = FancyBboxPatch((1.0, 2.0), 8.0, 3.4,
                           boxstyle="round,pad=0.1,rounding_size=0.2",
                           facecolor=C_TIME, alpha=0.08, edgecolor=C_TIME,
                           linewidth=1.3, linestyle="--", zorder=1)
    ax.add_patch(rect3)
    ax.text(1.3, 5.2, "GeGLU Feed-Forward", fontsize=9, color="#b7950b",
            fontweight="bold")

    ff_items = [
        ("LayerNorm", "#5d6d7e", 4.8),
        ("Linear → 8·C  (split: value + gate)", "#b7950b", 3.9),
        ("value × GELU(gate)", "#b7950b", 3.1),
        ("Linear → C", "#b7950b", 2.3),
    ]
    _arrow(ax, cx, 5.65, cx, 5.05, color=C_ARROW)
    for i, (lbl, col, y) in enumerate(ff_items):
        _box(ax, cx, y, bw - 0.4, 0.5, lbl, col, fontsize=8)
        if i < len(ff_items) - 1:
            _arrow(ax, cx, y - 0.25, cx,
                   ff_items[i + 1][2] + 0.25, color=C_ARROW)

    # FFN residual
    _box(ax, cx, 1.5, bw - 0.4, 0.5, "(+) Residual", C_RESID, fontsize=8)
    _arrow(ax, cx, 2.05, cx, 1.75, color=C_ARROW)
    _arrow(ax, 1.5, 5.65, 1.5, 1.5, color=C_RESID, ls="--", lw=1.2)
    _arrow(ax, 1.5, 1.5, cx - bw / 2 + 0.2,
           1.5, color=C_RESID, ls="--", lw=1.2)

    # --- Output ---
    _box(ax, cx, 0.5, bw, 0.5, "Reshape → (B, C, H, W)", "#5d6d7e", fontsize=9)
    _arrow(ax, cx, 1.25, cx, 0.75, color=C_ARROW)

    _box(ax, cx, -0.3, bw, 0.5, "Conv 1×1 + Long Residual", C_UNET, fontsize=9)
    _arrow(ax, cx, 0.25, cx, -0.05, color=C_ARROW)

    # Long residual
    ax.annotate("", xy=(0.5, -0.3), xytext=(0.5, 14.0),
                arrowprops=dict(arrowstyle="-|>", color=C_SKIP,
                                linewidth=1.5, linestyle="--"))
    ax.text(0.2, 7.0, "long\nresidual", fontsize=7, color=C_SKIP,
            ha="center", rotation=90, fontstyle="italic")

    fig.savefig(os.path.join(OUT_DIR, "unet_attention_block.png"))
    plt.close(fig)
    print("  ✓ unet_attention_block.png")


# ===================================================================
# 6. Residual Blocks
# ===================================================================

def draw_residual_blocks() -> None:
    """VAE ResBlock and UNET ResBlock (with time conditioning)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # --- VAE Residual Block ---
    ax = axes[0]
    ax.set_xlim(-1, 9)
    ax.set_ylim(-0.5, 9)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("VAE Residual Block  (vae_utils.py)", fontsize=12,
                 fontweight="bold", color=C_VAE_ENC, pad=10)

    cx, bw = 4.0, 3.5
    vae_rb = [
        ("Input  (B, in_ch, H, W)", C_LIGHT_BG, 8.2, C_DARK_TEXT),
        ("GroupNorm(32)", "#7f8c8d", 7.3, C_TEXT),
        ("SiLU", "#16a085", 6.5, C_TEXT),
        ("Conv 3×3  in→out", C_VAE_ENC, 5.7, C_TEXT),
        ("GroupNorm(32)", "#7f8c8d", 4.9, C_TEXT),
        ("SiLU", "#16a085", 4.1, C_TEXT),
        ("Conv 3×3  out→out", C_VAE_ENC, 3.3, C_TEXT),
        ("(+) Add", C_RESID, 2.3, C_TEXT),
        ("Output  (B, out_ch, H, W)", C_LIGHT_BG, 1.3, C_DARK_TEXT),
    ]

    for i, (lbl, col, y, tc) in enumerate(vae_rb):
        _box(ax, cx, y, bw, 0.5, lbl, col, text_color=tc, fontsize=8)
        if i < len(vae_rb) - 1:
            _arrow(ax, cx, y - 0.25, cx,
                   vae_rb[i + 1][2] + 0.25, color=C_ARROW)

    # Skip connection line
    ax.annotate("", xy=(1.0, 2.3), xytext=(1.0, 8.2),
                arrowprops=dict(arrowstyle="-|>", color=C_SKIP,
                                linewidth=1.5, linestyle="--"))
    _arrow(ax, 1.0, 2.3, cx - bw / 2, 2.3, color=C_SKIP, ls="--")
    ax.text(0.35, 5.2, "Identity\nor\nConv 1×1", fontsize=7, color=C_SKIP,
            ha="center", rotation=90, fontstyle="italic")

    # --- UNET Residual Block ---
    ax = axes[1]
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 9)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("U-Net Residual Block  (unet_utils.py)", fontsize=12,
                 fontweight="bold", color=C_UNET, pad=10)

    cx = 4.5
    unet_rb = [
        ("Feature  (B, in_ch, H, W)", C_LIGHT_BG, 8.2, C_DARK_TEXT),
        ("GroupNorm(32)", "#7f8c8d", 7.3, C_TEXT),
        ("SiLU", "#16a085", 6.5, C_TEXT),
        ("Conv 3×3  in→out", C_UNET, 5.7, C_TEXT),
        ("(+) Add  (feature + time)", C_TIME, 4.7, C_DARK_TEXT),
        ("GroupNorm(32)", "#7f8c8d", 3.9, C_TEXT),
        ("SiLU", "#16a085", 3.1, C_TEXT),
        ("Conv 3×3  out→out", C_UNET, 2.3, C_TEXT),
        ("(+) Add  (+ residual)", C_RESID, 1.3, C_TEXT),
        ("Output  (B, out_ch, H, W)", C_LIGHT_BG, 0.3, C_DARK_TEXT),
    ]

    for i, (lbl, col, y, tc) in enumerate(unet_rb):
        _box(ax, cx, y, bw, 0.5, lbl, col, text_color=tc, fontsize=8)
        if i < len(unet_rb) - 1:
            _arrow(ax, cx, y - 0.25, cx,
                   unet_rb[i + 1][2] + 0.25, color=C_ARROW)

    # Time embedding branch
    _box(ax, 8.5, 6.0, 2.5, 0.5, "Time (B, 1280)", C_TIME,
         text_color=C_DARK_TEXT, fontsize=8)
    _box(ax, 8.5, 5.2, 2.5, 0.5, "SiLU → Linear\n1280 → out_ch", C_TIME,
         text_color=C_DARK_TEXT, fontsize=7)
    _arrow(ax, 8.5, 5.75, 8.5, 5.45, color=C_TIME)
    _arrow(ax, 8.5, 4.95, cx + bw / 2, 4.7, color=C_TIME, ls="--")

    # Skip connection
    ax.annotate("", xy=(1.0, 1.3), xytext=(1.0, 8.2),
                arrowprops=dict(arrowstyle="-|>", color=C_SKIP,
                                linewidth=1.5, linestyle="--"))
    _arrow(ax, 1.0, 1.3, cx - bw / 2, 1.3, color=C_SKIP, ls="--")
    ax.text(0.35, 4.5, "Identity\nor\nConv 1×1", fontsize=7, color=C_SKIP,
            ha="center", rotation=90, fontstyle="italic")

    fig.tight_layout(pad=2)
    fig.savefig(os.path.join(OUT_DIR, "residual_blocks.png"))
    plt.close(fig)
    print("  ✓ residual_blocks.png")


# ===================================================================
# 7. CLIP Text Encoder
# ===================================================================

def draw_clip() -> None:
    """CLIP text encoder: embedding + 12 transformer layers."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 14)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.set_title("CLIP Text Encoder  (clip.py)", fontsize=14,
                 fontweight="bold", color=C_CLIP, pad=10)

    cx, bw = 5.0, 4.5

    blocks = [
        ("Input Tokens  (B, 77)", C_LIGHT_BG, 13.2, C_DARK_TEXT),
    ]

    _box(ax, cx, 13.2, bw, 0.5, "Input Tokens  (B, 77)",
         C_LIGHT_BG, text_color=C_DARK_TEXT, fontsize=9, edge_color="#aaa")

    # Embedding sub-block
    rect = FancyBboxPatch((cx - bw / 2 - 0.3, 11.0), bw + 0.6, 1.8,
                          boxstyle="round,pad=0.1,rounding_size=0.2",
                          facecolor=C_CLIP, alpha=0.1, edgecolor=C_CLIP,
                          linewidth=1.3, linestyle="--", zorder=1)
    ax.add_patch(rect)
    ax.text(cx - bw / 2, 12.65, "CLIPEmbedding", fontsize=9,
            color=C_CLIP, fontweight="bold")

    _box(ax, 3.5, 12.2, 2.8, 0.5, "Token Embedding\nnn.Embedding(49408, 768)",
         C_CLIP, fontsize=7)
    _box(ax, 7.0, 12.2, 2.8, 0.5, "Position Embedding\nnn.Parameter(77, 768)",
         C_CLIP, fontsize=7)
    _box(ax, cx, 11.3, 2.0, 0.5, "(+) Add", C_CLIP_DARK, fontsize=9)

    _arrow(ax, cx, 12.95, 3.5, 12.45, color=C_ARROW)
    _arrow(ax, cx, 12.95, 7.0, 12.45, color=C_ARROW)
    _arrow(ax, 3.5, 11.95, cx, 11.55, color=C_ARROW)
    _arrow(ax, 7.0, 11.95, cx, 11.55, color=C_ARROW)

    # Transformer block (shown once, with ×12 label)
    rect2 = FancyBboxPatch((cx - bw / 2 - 0.3, 4.3), bw + 0.6, 6.4,
                           boxstyle="round,pad=0.1,rounding_size=0.2",
                           facecolor=C_CLIP, alpha=0.08, edgecolor=C_CLIP,
                           linewidth=1.5, linestyle="-", zorder=1)
    ax.add_patch(rect2)
    ax.text(cx + bw / 2 + 0.05, 10.5, "×12", fontsize=14,
            color=C_CLIP, fontweight="bold", ha="left")
    ax.text(cx - bw / 2, 10.5, "CLIPLayer (Transformer Block)",
            fontsize=9, color=C_CLIP, fontweight="bold")

    tf_blocks = [
        ("LayerNorm", "#5d6d7e", 9.9),
        ("Causal Self-Attention\n(12 heads × 64-dim)", C_ATTN, 9.0),
        ("(+) Residual", C_RESID, 8.1),
        ("LayerNorm", "#5d6d7e", 7.3),
        ("Linear  768 → 3072  (4×)", C_CLIP, 6.5),
        ("QuickGELU:  x · σ(1.702·x)", "#16a085", 5.7),
        ("Linear  3072 → 768", C_CLIP, 4.9),
        ("(+) Residual", C_RESID, 4.1),  # Intentionally moved down
    ]

    _arrow(ax, cx, 11.05, cx, tf_blocks[0][2] + 0.25, color=C_ARROW)

    for i, (lbl, col, y) in enumerate(tf_blocks):
        h = 0.5 if "\n" not in lbl else 0.65
        _box(ax, cx, y, bw - 0.4, h, lbl, col, fontsize=8)
        if i < len(tf_blocks) - 1:
            ny = tf_blocks[i + 1][2]
            nh = 0.5 if "\n" not in tf_blocks[i + 1][0] else 0.65
            _arrow(ax, cx, y - h / 2, cx, ny + nh / 2, color=C_ARROW)

    # Residual connections
    # First residual: from before LN1 to after SA
    ax.annotate("", xy=(cx - bw / 2 + 0.1, 8.1), xytext=(cx - bw / 2 + 0.1, 9.9 + 0.3),
                arrowprops=dict(arrowstyle="-|>", color=C_SKIP,
                                linewidth=1.2, linestyle="--"))
    # Second residual: from before LN2 to after FFN
    ax.annotate("", xy=(cx - bw / 2 + 0.1, 4.1), xytext=(cx - bw / 2 + 0.1, 7.3 + 0.3),
                arrowprops=dict(arrowstyle="-|>", color=C_SKIP,
                                linewidth=1.2, linestyle="--"))

    # Final LayerNorm + output
    _arrow(ax, cx, 3.85, cx, 3.35, color=C_ARROW)
    _box(ax, cx, 3.0, bw - 0.4, 0.5, "Final LayerNorm", "#5d6d7e", fontsize=9)
    _arrow(ax, cx, 2.75, cx, 2.25, color=C_ARROW)
    _box(ax, cx, 1.9, bw, 0.5, "Context Embeddings  (B, 77, 768)",
         C_LIGHT_BG, text_color=C_DARK_TEXT, fontsize=9, edge_color="#aaa")

    fig.savefig(os.path.join(OUT_DIR, "clip_text_encoder.png"))
    plt.close(fig)
    print("  ✓ clip_text_encoder.png")


# ===================================================================
# 8. Diffusion Process (Denoising Loop)
# ===================================================================

def draw_diffusion_process() -> None:
    """Complete pipeline: input → scheduler loop → output."""
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(-0.5, 17)
    ax.set_ylim(-1, 9.5)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(8, 8.6, "Stable Diffusion Diffusion Process", ha="center",
            fontsize=16, fontweight="bold", color=C_DARK_TEXT)

    # ===== LEFT SIDE: INPUTS =====
    # Random noise
    _box(ax, 1.0, 6.5, 1.5, 1.0, "Random\nNoise", "#555555", fontsize=9)
    _arrow(ax, 1.75, 6.0, 2.75, 5.0, color=C_ARROW, lw=2.5)

    # VAE Encoder
    rect = FancyBboxPatch((2.3, 4.5), 1.4, 1.2,
                          boxstyle="round,pad=0.1,rounding_size=0.2",
                          facecolor=C_VAE_ENC, alpha=0.9, edgecolor=C_VAE_ENC,
                          linewidth=2, zorder=3)
    ax.add_patch(rect)
    ax.text(3.0, 5.1, "Encoder", ha="center", va="center",
            fontsize=10, color=C_TEXT, fontweight="bold", zorder=4)

    # Z latent (noisy)
    _box(ax, 4.2, 5.0, 1.2, 0.8, "z_T", "#9b59b6", fontsize=10)
    _arrow(ax, 3.7, 5.0, 3.6, 5.0, color=C_ARROW, lw=2.5)

    # Text prompt
    _box(ax, 1.0, 3.0, 1.5, 0.8, "Text\nPrompt", "#555555", fontsize=9)
    _arrow(ax, 1.75, 2.6, 2.75, 2.0, color=C_ARROW, lw=2.5)

    # CLIP Encoder
    rect2 = FancyBboxPatch((2.3, 1.5), 1.4, 1.0,
                           boxstyle="round,pad=0.1,rounding_size=0.2",
                           facecolor=C_CLIP, alpha=0.9, edgecolor=C_CLIP,
                           linewidth=2, zorder=3)
    ax.add_patch(rect2)
    ax.text(3.0, 2.0, "CLIP", ha="center", va="center",
            fontsize=10, color=C_TEXT, fontweight="bold", zorder=4)

    # Context embeddings
    _box(ax, 4.2, 2.0, 1.2, 0.8, "Ctx", C_CLIP, fontsize=10)
    _arrow(ax, 3.7, 2.0, 3.6, 2.0, color=C_ARROW, lw=2.5)

    # ===== CENTER: DIFFUSION LOOP =====
    # Scheduler box (large, containing the loop)
    scheduler_rect = FancyBboxPatch((5.0, 0.8), 6.0, 6.5,
                                    boxstyle="round,pad=0.2,rounding_size=0.3",
                                    facecolor=C_TIME, alpha=0.08, edgecolor=C_TIME,
                                    linewidth=2.0, linestyle="-", zorder=1)
    ax.add_patch(scheduler_rect)
    ax.text(8.0, 7.2, "Scheduler  (T denoising steps)", fontsize=11,
            color=C_TIME, fontweight="bold")

    # Time embedding path
    _box(ax, 8.0, 6.5, 1.2, 0.6, "Timestep t", C_TIME, fontsize=8)
    _arrow(ax, 8.0, 6.2, 8.0, 5.7, color=C_TIME, lw=1.5)
    _box(ax, 8.0, 5.3, 1.5, 0.6, "Time Embed", C_TIME, fontsize=8)

    # Iterative denoising steps visualization
    n_steps = 4
    step_x = [5.5, 6.8, 8.1, 9.4]
    step_labels = ["t=T", "t=T-1", "...", "t=1"]
    step_colors = ["#e74c3c", "#e67e22", "#f39c12", "#27ae60"]

    for i, (sx, lbl, col) in enumerate(zip(step_x, step_labels, step_colors)):
        _box(ax, sx, 4.0, 1.0, 0.7, lbl, col, fontsize=9,
             text_color=C_TEXT if col != "#f39c12" else C_DARK_TEXT)
        # Arrow to U-Net
        _arrow(ax, sx, 3.65, sx, 3.2, color=C_ARROW, lw=1.2)

    # U-Net central box
    unet_rect = FancyBboxPatch((5.2, 1.8), 4.6, 1.0,
                               boxstyle="round,pad=0.1,rounding_size=0.2",
                               facecolor=C_UNET, alpha=0.9, edgecolor=C_UNET,
                               linewidth=2, zorder=3)
    ax.add_patch(unet_rect)
    ax.text(7.5, 2.3, "U-Net  ε_θ(z_t, t, ctx)", ha="center", va="center",
            fontsize=10, color=C_TEXT, fontweight="bold", zorder=4)

    # Arrows from steps to U-Net
    for sx in step_x:
        _arrow(ax, sx, 3.2, 7.5, 2.8, color=C_UNET, ls="--", lw=1.0)

    # Context feeding in
    _arrow(ax, 4.8, 2.0, 5.2, 2.3, color=C_CLIP, lw=2.0)
    ax.text(4.5, 1.65, "ctx", fontsize=8, color=C_CLIP, fontweight="bold")

    # Time embedding feeding in
    _arrow(ax, 8.0, 4.9, 8.0, 2.8, color=C_TIME, lw=2.0)
    ax.text(7.3, 3.8, "t", fontsize=8, color=C_TIME, fontweight="bold")

    # Predicted noise output
    _box(ax, 7.5, 1.2, 1.5, 0.5, "Predicted ε", C_UNET, fontsize=8)
    _arrow(ax, 7.5, 1.55, 7.5, 1.8, color=C_ARROW, lw=1.5)

    # Scheduler step (iterative update)
    ax.text(7.5, 0.5, "z_{t-1} ← Scheduler(z_t, ε)", ha="center",
            fontsize=8, color=C_DARK_TEXT, fontstyle="italic")

    # ===== RIGHT SIDE: OUTPUT =====
    # Clean latent z_0 exiting scheduler
    _arrow(ax, 11.0, 3.5, 12.0, 3.5, color=C_ARROW, lw=2.5)
    _box(ax, 12.5, 3.5, 1.2, 0.8, "z_0", "#27ae60", fontsize=10)

    # VAE Decoder
    rect3 = FancyBboxPatch((13.3, 3.0), 1.4, 1.2,
                           boxstyle="round,pad=0.1,rounding_size=0.2",
                           facecolor=C_VAE_DEC, alpha=0.9, edgecolor=C_VAE_DEC,
                           linewidth=2, zorder=3)
    ax.add_patch(rect3)
    ax.text(14.0, 3.6, "Decoder", ha="center", va="center",
            fontsize=10, color=C_TEXT, fontweight="bold", zorder=4)

    # Generated image output
    _arrow(ax, 14.7, 3.5, 15.3, 3.5, color=C_ARROW, lw=2.5)
    _box(ax, 15.8, 3.5, 0.9, 0.9, "X'", C_LIGHT_BG, text_color=C_DARK_TEXT,
         fontsize=12, edge_color="#999")
    ax.text(15.8, 2.8, "Generated\nImage", ha="center", fontsize=8,
            color=C_DARK_TEXT, fontstyle="italic")

    # Legend
    legend_x = 0.5
    legend_y = -0.4
    ax.text(legend_x, legend_y, "Legend:", fontsize=10,
            fontweight="bold", color=C_DARK_TEXT)

    # Legend items
    _box(ax, legend_x + 0.6, legend_y - 0.35,
         0.8, 0.3, "Input", "#555555", fontsize=7)
    ax.text(legend_x + 1.5, legend_y - 0.35,
            "User inputs", fontsize=7, va="center")

    _box(ax, legend_x + 3.2, legend_y - 0.35,
         0.8, 0.3, "Process", C_UNET, fontsize=7)
    ax.text(legend_x + 4.1, legend_y - 0.35,
            "Neural network step", fontsize=7, va="center")

    _box(ax, legend_x + 6.2, legend_y - 0.35, 0.8,
         0.3, "Output", "#27ae60", fontsize=7)
    ax.text(legend_x + 7.1, legend_y - 0.35,
            "Generated result", fontsize=7, va="center")

    fig.savefig(os.path.join(OUT_DIR, "diffusion_process.png"))
    plt.close(fig)
    print("  ✓ diffusion_process.png")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("Generating architecture diagrams …")
    draw_pipeline()
    draw_unet()
    draw_vae()
    draw_attention()
    draw_unet_attention_block()
    draw_residual_blocks()
    draw_clip()
    draw_diffusion_process()
    draw_dependency_graph()
    print(f"\nAll diagrams saved to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
