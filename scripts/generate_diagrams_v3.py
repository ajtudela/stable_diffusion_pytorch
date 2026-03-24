"""Generate architecture diagrams for the Stable Diffusion implementation.

Creates publication-quality PNG diagrams using Graphviz for:
  1. Overall Stable Diffusion pipeline
  2. U-Net architecture (classic U-shape)
  3. VAE Encoder / Decoder
  4. Transformer / Attention mechanism
  5. UNET Attention Block (Self-Attn -> Cross-Attn -> FFN)
  6. Residual blocks (VAE and U-Net variants)
  7. CLIP Text Encoder
  8. Diffusion Denoising Process

All images are saved to the ``docs/`` directory.

Requires:
  - Graphviz system binary (``dot``) on PATH.
  - Python ``graphviz`` package (``pip install graphviz``).
"""

from __future__ import annotations

import os

import graphviz

# ---------------------------------------------------------------------------
# Output directory -- align with README image paths (docs/)
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "docs"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
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
C_GEGLU = "#b7950b"
C_NORM = "#5d6d7e"
C_ACT = "#16a085"
C_GRAY = "#7f8c8d"
C_DOWNSAMPLE = "#d35400"
C_UPSAMPLE = "#1e8449"

# Shared node defaults
# Cross-platform font: prefer Segoe UI (Windows), fall back to Arial
_FONT = "Segoe UI"
_FONT_BOLD = "Segoe UI Bold"

_NODE_DEFAULTS: dict[str, str] = dict(
    shape="box",
    style="rounded,filled",
    fontname=_FONT,
    fontsize="10",
    margin="0.15,0.08",
)


def _render(g: graphviz.Digraph, name: str) -> None:
    """Render graph *g* to PNG in OUT_DIR and print confirmation."""
    path = g.render(
        filename=os.path.join(OUT_DIR, name),
        format="png",
        cleanup=True,
    )
    print(f"  > {os.path.basename(path)}")


# ===================================================================
# 1. Overall Stable Diffusion Pipeline
# ===================================================================


def draw_pipeline() -> None:
    """Stable Diffusion pipeline with denoising loop (merged overview).

    Layout inspired by the classic SD pipeline diagram:
    - Text prompt → Tokenizer → CLIP → text embeddings ↓ U-Net (top row)
    - Noise seed → Latents → U-Net → Scheduler loop (middle row)
    - Output latents → VAE Decoder → Generated Image (right)
    """
    g = graphviz.Digraph(
        "pipeline",
        graph_attr=dict(
            rankdir="LR",
            label="Stable Diffusion \u2014 Pipeline Overview",
            labelloc="t",
            fontsize="20",
            fontname=_FONT_BOLD,
            fontcolor=C_DARK_TEXT,
            bgcolor="white",
            pad="0.5",
            nodesep="0.35",
            ranksep="0.6",
            dpi="180",
        ),
        node_attr=_NODE_DEFAULTS,
        edge_attr=dict(fontname=_FONT, fontsize="9"),
    )

    # ── Left column: prompt + inputs ────────────────────────────────
    g.node("prompt", "Text Prompt\n\"a person surfing a wave\"",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)

    with g.subgraph(name="cluster_inputs") as inp:
        inp.attr(
            label="Input Images",
            fontsize="12", fontname=_FONT_BOLD, fontcolor=C_DARK_TEXT,
            style="rounded,bold", color="#999999", bgcolor="#f5f5f5",
        )
        inp.node("image_in", "Input Image\n(optional)\n512 × 512",
                 fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)
        inp.node("noise", "Latent Seed\n(noise)\n64 × 64",
                 fillcolor="#444444", fontcolor=C_TEXT,
                 style="rounded,filled")

    # ── Top row: text path ─────────────────────────────────────────
    g.node("tokenizer", "Text\nTokenizer",
           fillcolor=C_CLIP_DARK, fontcolor=C_TEXT)
    g.node("clip", "CLIP",
           fillcolor=C_CLIP, fontcolor=C_TEXT,
           width="1.2", height="0.7")
    g.node("text_emb", "Text\nEmbeddings\n(77 \u00d7 768)",
           fillcolor=C_CLIP, fontcolor=C_TEXT,
           style="rounded,filled,dashed")

    g.edge("prompt", "tokenizer", color=C_CLIP, penwidth="2")
    g.edge("tokenizer", "clip", color=C_CLIP, penwidth="2")
    g.edge("clip", "text_emb", color=C_CLIP, penwidth="2",
           label="  768  ")

    g.node("latents", "Latents\n64 \u00d7 64",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)

    # ── Centre: U-Net + Scheduler loop ─────────────────────────────
    with g.subgraph(name="cluster_denoise") as loop:
        loop.attr(
            label="Denoising Loop",
            fontsize="13", fontname=_FONT_BOLD, fontcolor=C_TIME,
            style="rounded,bold", color=C_TIME, bgcolor="#fef9e7",
        )
        loop.node(
            "unet", "U-Net",
            fillcolor=C_UNET, fontcolor=C_TEXT,
            width="1.8", height="1.0",
            fontsize="14", fontname=_FONT_BOLD,
        )
        loop.node(
            "scheduler", "Scheduler\n(denoise step)",
            fillcolor=C_DARK_TEXT, fontcolor=C_TEXT,
            fontsize="10",
        )
        loop.edge("unet", "scheduler",
                  color=C_UNET, penwidth="2",
                  label="  predicted noise  ",
                  fontsize="8")
        # Scheduler feeds back to U-Net (loop within the denoising cluster)
        loop.edge("scheduler", "unet",
                  color=C_TIME, penwidth="2",
                  label="  repeat N times  ",
                  fontcolor=C_TIME, fontname=_FONT_BOLD,
                  fontsize="9",
                  style="bold",
                  constraint="false")

    g.node("latents", "Latents\n64 × 64",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)

    # ── Left: VAE Encoder (inputs → latents) ────────────────
    g.node("vae_enc", "VAE\nEncoder",
           fillcolor=C_VAE_ENC, fontcolor=C_TEXT,
           shape="trapezium", orientation="270",
           width="1.6", height="0.8")

    # ── Right: VAE Decoder (latents → image) ────────────────────
    g.node("out_latents", "Text\nConditioned\nLatents\n64 \u00d7 64",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)
    g.node("vae_dec", "VAE\nDecoder",
           fillcolor=C_VAE_DEC, fontcolor=C_TEXT,
           shape="invtrapezium", orientation="270",
           width="1.6", height="0.8")
    g.node("image_out", "Generated\nImage\n512 \u00d7 512",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)

    # ── Main flow edges ────────────────────────────────────────────
    # Inputs cluster → VAE Encoder → Latents
    g.edge("image_in", "vae_enc", color=C_VAE_ENC, penwidth="2")
    g.edge("vae_enc", "latents", color=C_VAE_ENC, penwidth="2")
    # Latents → U-Net
    g.edge("latents", "unet", color=C_DARK_TEXT, penwidth="2",
           label="  Latents  ")
    # Text embeddings → U-Net
    g.edge("text_emb", "unet", color=C_CLIP, penwidth="2",
           style="dashed")
    # Scheduler → output latents (after final iteration)
    g.edge("scheduler", "out_latents", color=C_UNET, penwidth="2",
           label="  final  ",
           fontsize="8")
    # Output latents → VAE Decoder → Image
    g.edge("out_latents", "vae_dec", color=C_VAE_DEC, penwidth="2")
    g.edge("vae_dec", "image_out", color=C_VAE_DEC, penwidth="2")

    # ── Layout hints ───────────────────────────────────────────────
    # In LR layout, rank="same" = same vertical column.
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("tokenizer")
        s.node("vae_enc")

    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("clip")
        s.node("latents")

    # Vertical ordering: text prompt above input images
    g.edge("prompt", "image_in", style="invis")
    g.edge("tokenizer", "vae_enc", style="invis")
    g.edge("clip", "latents", style="invis")

    _render(g, "pipeline_overview")


# ===================================================================
# 2. U-Net Architecture
# ===================================================================


def draw_unet() -> None:
    """U-Net in classic U-shape inspired by the original paper diagram.

    Uses color-coded arrows for different operations and a legend.
    Encoder path goes down-left, decoder path goes up-right,
    with horizontal skip (copy & concat) connections.
    """
    # Colour definitions for arrow types
    C_CONV = "#1565C0"       # conv 3x3 / main flow
    C_POOL = "#8B0000"       # max pool / downsample
    C_UP = "#1B5E20"         # up-conv / upsample
    C_COPY = "#9E9E9E"       # copy & crop (skip)
    C_CONV1 = "#00838F"      # conv 1x1 output

    # Block colours: encoder lighter, decoder slightly different
    C_ENC_BLOCK = "#90CAF9"
    C_DEC_BLOCK = "#81D4FA"
    C_BN_BLOCK = "#7986CB"

    g = graphviz.Digraph(
        "unet",
        graph_attr=dict(
            rankdir="TB",
            label="U-Net Architecture \u2014 Noise Prediction Network",
            labelloc="t",
            fontsize="18",
            fontname=_FONT_BOLD,
            fontcolor=C_DARK_TEXT,
            bgcolor="white",
            pad="0.5",
            nodesep="0.7",
            ranksep="0.55",
            dpi="180",
            splines="line",
        ),
        node_attr=dict(
            shape="box",
            style="filled",
            fontname=_FONT,
            fontsize="9",
            margin="0.12,0.06",
        ),
        edge_attr=dict(fontname=_FONT, fontsize="8"),
    )

    # --- I/O nodes ---
    g.node("input", "Noisy Latent\n(B, 4, H/8, W/8)",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT,
           style="rounded,filled")
    g.node("output", "Predicted Noise\n(B, 4, H/8, W/8)",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT,
           style="rounded,filled")

    # --- Encoder blocks ---
    enc_blocks = [
        ("enc0", "Conv 4\u2192320\n2\u00d7(Res+Attn)\n"
                 "320 ch", "H/8 \u00d7 W/8"),
        ("enc1", "2\u00d7(Res 320\u2192640+Attn)\n"
                 "640 ch", "H/16 \u00d7 W/16"),
        ("enc2", "2\u00d7(Res 640\u21921280+Attn)\n"
                 "1280 ch", "H/32 \u00d7 W/32"),
        ("enc3", "2\u00d7 ResBlock\n"
                 "1280 ch", "H/64 \u00d7 W/64"),
    ]
    for nid, lbl, res in enc_blocks:
        g.node(nid, f"{lbl}\n{res}",
               fillcolor=C_ENC_BLOCK, fontcolor=C_DARK_TEXT,
               width="2.2", group="enc")

    # --- Bottleneck ---
    g.node("bn",
           "Bottleneck\nRes \u2192 Self-Attn(8h,160d) \u2192 Res\n"
           "1280 ch \u00b7 H/64 \u00d7 W/64",
           fillcolor=C_BN_BLOCK, fontcolor=C_TEXT,
           width="3.5")

    # --- Decoder blocks ---
    dec_blocks = [
        ("dec0", "2\u00d7Res\n"
                 "1280 ch", "H/64\u2192H/32"),
        ("dec1", "2\u00d7(Res+Attn)\n"
                 "1280 ch", "H/32\u2192H/16"),
        ("dec2", "2\u00d7(Res+Attn)\n"
                 "640 ch", "H/16\u2192H/8"),
        ("dec3", "3\u00d7(Res+Attn)\n"
                 "320 ch", "H/8 \u00d7 W/8"),
    ]
    for nid, lbl, res in dec_blocks:
        g.node(nid, f"{lbl}\n{res}",
               fillcolor=C_DEC_BLOCK, fontcolor=C_DARK_TEXT,
               width="2.2", group="dec")

    # --- Output conv ---
    g.node("out_conv", "GN \u2192 SiLU \u2192 Conv 320\u21924",
           fillcolor=C_GRAY, fontcolor=C_TEXT, group="dec")

    # --- Downsample nodes (explicit, colour-coded) ---
    for i in range(3):
        g.node(f"down{i}", "\u2193 stride 2",
               fillcolor="#FFCDD2", fontcolor=C_POOL,
               shape="ellipse", width="1.0", height="0.3",
               fontsize="8", group="enc")

    # --- Upsample nodes (explicit, colour-coded) ---
    for i in range(3):
        g.node(f"up{i}", "\u2191 upsample \u00d72",
               fillcolor="#C8E6C9", fontcolor=C_UP,
               shape="ellipse", width="1.0", height="0.3",
               fontsize="8", group="dec")

    # --- Legend ---
    g.node(
        "legend",
        ("Legend\n"
         "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
         "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
         "\u2192  conv 3\u00d73 / ResBlock + Attn\n"
         "\u2192  max pool / stride-2 downsample\n"
         "\u2192  upsample \u00d72 + conv\n"
         "\u2192  copy & concat (skip)\n"
         "\u2192  conv 1\u00d71 output\n\n"
         "\u23f1 Time Emb \u2192 all ResBlocks\n"
         "\U0001f4dd CLIP Context \u2192 cross-attn\n"
         "   in enc0\u20132, bn, dec1\u20133"),
        shape="note", style="filled",
        fillcolor="#f9f9f9", fontcolor=C_DARK_TEXT,
        fontname=_FONT, fontsize="9", color="#cccccc",
    )

    # --- Rank constraints (classic U-shape) ---
    # Row 0 (top): input, out_conv, output, legend
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("input")
        s.node("out_conv")
        s.node("output")
        s.node("legend")
    # Row 1: enc0 / dec3
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("enc0")
        s.node("dec3")
    # Row 1.5: down0 / up2
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("down0")
        s.node("up2")
    # Row 2: enc1 / dec2
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("enc1")
        s.node("dec2")
    # Row 2.5: down1 / up1
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("down1")
        s.node("up1")
    # Row 3: enc2 / dec1
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("enc2")
        s.node("dec1")
    # Row 3.5: down2 / up0
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("down2")
        s.node("up0")
    # Row 4: enc3 / dec0
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("enc3")
        s.node("dec0")
    # Row 5 (bottom): bottleneck

    # --- Encoder flow (down, left column) ---
    g.edge("input", "enc0", color=C_CONV, penwidth="2")
    g.edge("enc0", "down0", color=C_POOL, penwidth="2")
    g.edge("down0", "enc1", color=C_POOL, penwidth="2")
    g.edge("enc1", "down1", color=C_POOL, penwidth="2")
    g.edge("down1", "enc2", color=C_POOL, penwidth="2")
    g.edge("enc2", "down2", color=C_POOL, penwidth="2")
    g.edge("down2", "enc3", color=C_POOL, penwidth="2")
    g.edge("enc3", "bn", color=C_CONV, penwidth="2")

    # --- Decoder flow (up, right column) ---
    g.edge("bn", "dec0", color=C_CONV, penwidth="2")
    g.edge("dec0", "up0", color=C_UP, penwidth="2",
           constraint="false")
    g.edge("up0", "dec1", color=C_UP, penwidth="2",
           constraint="false")
    g.edge("dec1", "up1", color=C_UP, penwidth="2",
           constraint="false")
    g.edge("up1", "dec2", color=C_UP, penwidth="2",
           constraint="false")
    g.edge("dec2", "up2", color=C_UP, penwidth="2",
           constraint="false")
    g.edge("up2", "dec3", color=C_UP, penwidth="2",
           constraint="false")
    g.edge("dec3", "out_conv", color=C_CONV, penwidth="2",
           constraint="false")
    g.edge("out_conv", "output", color=C_CONV1, penwidth="2",
           constraint="false")

    # Invisible edges to help decoder column ordering (top→bottom)
    g.edge("dec3", "up2", style="invis")
    g.edge("up2", "dec2", style="invis")
    g.edge("dec2", "up1", style="invis")
    g.edge("up1", "dec1", style="invis")
    g.edge("dec1", "up0", style="invis")
    g.edge("up0", "dec0", style="invis")

    # --- Skip connections (horizontal, same rank) ---
    skip_pairs = [
        ("enc3", "dec0", "concat 1280ch"),
        ("enc2", "dec1", "concat 1280ch"),
        ("enc1", "dec2", "concat 640ch"),
        ("enc0", "dec3", "concat 320ch"),
    ]
    for src, dst, lbl in skip_pairs:
        g.edge(src, dst, label=f"  {lbl}  ", color=C_COPY,
               style="dashed", penwidth="2", fontcolor=C_COPY,
               constraint="false", arrowhead="vee")

    _render(g, "unet_architecture")


# ===================================================================
# 3. VAE Encoder & Decoder
# ===================================================================


def draw_vae() -> None:
    """VAE encoder and decoder as two side-by-side subgraphs."""
    g = graphviz.Digraph(
        "vae",
        graph_attr=dict(
            rankdir="TB",
            label="VAE Encoder & Decoder",
            labelloc="t",
            fontsize="18",
            fontname=_FONT_BOLD,
            fontcolor=C_DARK_TEXT,
            bgcolor="white",
            pad="0.4",
            nodesep="0.4",
            ranksep="0.35",
            dpi="180",
            compound="true",
            newrank="true",
        ),
        node_attr=_NODE_DEFAULTS,
        edge_attr=dict(fontname=_FONT, fontsize="8"),
    )

    # --- Encoder ---
    with g.subgraph(name="cluster_enc") as enc:
        enc.attr(
            label="VAE Encoder  (image \u2192 latent)",
            labelloc="t", fontsize="14", fontname=_FONT_BOLD,
            fontcolor=C_VAE_ENC,
            style="rounded,dashed", color=C_VAE_ENC, bgcolor="#fdf2e9",
        )
        enc_blocks = [
            ("e_in", "Input Image\n(B, 3, H, W)", C_LIGHT_BG, C_DARK_TEXT, ""),
            ("e_conv0", "Conv2d 3\u2192128", C_VAE_ENC, C_TEXT, "H \u00d7 W"),
            ("e_res1", "ResBlock(128,128) \u00d72", C_VAE_ENC, C_TEXT, ""),
            ("e_down1", "\u2193 Conv stride=2", C_DOWNSAMPLE, C_TEXT, ""),
            ("e_res2", "ResBlock(128\u2192256)\nResBlock(256,256)",
             C_VAE_ENC, C_TEXT, "H/2 \u00d7 W/2"),
            ("e_down2", "\u2193 Conv stride=2", C_DOWNSAMPLE, C_TEXT, ""),
            ("e_res3", "ResBlock(256\u2192512)\nResBlock(512,512)",
             C_VAE_ENC, C_TEXT, "H/4 \u00d7 W/4"),
            ("e_down3", "\u2193 Conv stride=2", C_DOWNSAMPLE, C_TEXT, ""),
            ("e_res4", "ResBlock(512,512) \u00d73",
             C_VAE_ENC, C_TEXT, "H/8 \u00d7 W/8"),
            ("e_attn", "Self-Attention(512)", C_ATTN, C_TEXT, ""),
            ("e_res5", "ResBlock(512,512)", C_VAE_ENC, C_TEXT, ""),
            ("e_norm", "GroupNorm(32) \u2192 SiLU", C_GRAY, C_TEXT, ""),
            ("e_proj", "Conv 512\u21928 \u2192 Conv 1\u00d71",
             C_VAE_ENC, C_TEXT, ""),
            ("e_reparam",
             "Split \u03bc + log \u03c3\u00b2\n"
             "z = \u03bc + \u03c3\u00b7\u03b5  \u00d7 0.18215",
             C_RESID, C_TEXT, ""),
            ("e_out", "Latent z\n(B, 4, H/8, W/8)",
             C_LIGHT_BG, C_DARK_TEXT, ""),
        ]
        for nid, lbl, fc, tc, _ in enc_blocks:
            enc.node(nid, lbl, fillcolor=fc, fontcolor=tc)
        for i in range(len(enc_blocks) - 1):
            kw2: dict[str, str] = {}
            res = enc_blocks[i][4]
            if res:
                kw2["xlabel"] = res
                kw2["fontcolor"] = "#888888"
                kw2["fontsize"] = "8"
            enc.edge(enc_blocks[i][0], enc_blocks[i + 1][0],
                     color=C_VAE_ENC, **kw2)

    # --- Decoder ---
    with g.subgraph(name="cluster_dec") as dec:
        dec.attr(
            label="VAE Decoder  (latent \u2192 image)",
            labelloc="t", fontsize="14", fontname=_FONT_BOLD,
            fontcolor=C_VAE_DEC,
            style="rounded,dashed", color=C_VAE_DEC, bgcolor="#eafaf1",
        )
        dec_blocks = [
            ("d_in", "Latent z / 0.18215\n(B, 4, H/8, W/8)",
             C_LIGHT_BG, C_DARK_TEXT, ""),
            ("d_proj", "Conv 4\u21924 (1\u00d71)\nConv 4\u2192512",
             C_VAE_DEC, C_TEXT, "H/8 \u00d7 W/8"),
            ("d_res0", "ResBlock(512,512)", C_VAE_DEC, C_TEXT, ""),
            ("d_attn", "Self-Attention(512)", C_ATTN, C_TEXT, ""),
            ("d_res1", "ResBlock(512,512) \u00d74", C_VAE_DEC, C_TEXT, ""),
            ("d_up1", "\u2191 Upsample \u00d72 + Conv",
             C_UPSAMPLE, C_TEXT, ""),
            ("d_res2", "ResBlock(512,512) \u00d73",
             C_VAE_DEC, C_TEXT, "H/4 \u00d7 W/4"),
            ("d_up2", "\u2191 Upsample \u00d72 + Conv",
             C_UPSAMPLE, C_TEXT, ""),
            ("d_res3", "ResBlock(512\u2192256)\nResBlock(256,256) \u00d72",
             C_VAE_DEC, C_TEXT, "H/2 \u00d7 W/2"),
            ("d_up3", "\u2191 Upsample \u00d72 + Conv",
             C_UPSAMPLE, C_TEXT, ""),
            ("d_res4", "ResBlock(256\u2192128)\nResBlock(128,128) \u00d72",
             C_VAE_DEC, C_TEXT, "H \u00d7 W"),
            ("d_norm", "GroupNorm(32) \u2192 SiLU", C_GRAY, C_TEXT, ""),
            ("d_conv_out", "Conv 128\u21923", C_VAE_DEC, C_TEXT, ""),
            ("d_out", "RGB Image\n(B, 3, H, W)",
             C_LIGHT_BG, C_DARK_TEXT, ""),
        ]
        for nid, lbl, fc, tc, _ in dec_blocks:
            dec.node(nid, lbl, fillcolor=fc, fontcolor=tc)
        for i in range(len(dec_blocks) - 1):
            kw3: dict[str, str] = {}
            res = dec_blocks[i][4]
            if res:
                kw3["xlabel"] = res
                kw3["fontcolor"] = "#888888"
                kw3["fontsize"] = "8"
            dec.edge(dec_blocks[i][0], dec_blocks[i + 1][0],
                     color=C_VAE_DEC, **kw3)

    # Force side-by-side layout: anchor top and bottom nodes
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("e_in")
        s.node("d_in")
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("e_out")
        s.node("d_out")

    _render(g, "vae_encoder_decoder")


# ===================================================================
# 4. Attention Mechanisms
# ===================================================================


def draw_attention() -> None:
    """Self-Attention & Cross-Attention side by side."""
    g = graphviz.Digraph(
        "attention",
        graph_attr=dict(
            rankdir="TB",
            label="Attention Mechanisms",
            labelloc="t", fontsize="18", fontname=_FONT_BOLD,
            fontcolor=C_DARK_TEXT, bgcolor="white",
            pad="0.4", nodesep="0.3", ranksep="0.35", dpi="180",
            splines="ortho",
        ),
        node_attr=_NODE_DEFAULTS,
        edge_attr=dict(fontname=_FONT, fontsize="8"),
    )

    # --- Self-Attention ---
    with g.subgraph(name="cluster_sa") as sa:
        sa.attr(
            label="Multi-Head Self-Attention",
            labelloc="t", fontsize="14", fontname=_FONT_BOLD,
            fontcolor=C_ATTN,
            style="rounded,dashed", color=C_ATTN, bgcolor="#f5f5f5",
        )
        sa_nodes = [
            ("sa_in", "Input x\n(B, SeqLen, d_embed)",
             C_LIGHT_BG, C_DARK_TEXT),
            ("sa_proj", "Linear \u2192 3\u00b7d_embed\n"
             "(fused Q, K, V projection)", C_ATTN, C_TEXT),
            ("sa_split", "Split \u2192 Q, K, V", C_ATTN, C_TEXT),
            ("sa_reshape", "Reshape\n(B, n_heads, Seq, d_head)",
             C_NORM, C_TEXT),
            ("sa_scores", "Attention Scores\nQ \u00b7 K\u1d40 / \u221ad_head",
             C_RESID, C_TEXT),
            ("sa_mask",
             "Causal Mask (optional)\n\u2212\u221e for future positions",
             C_GRAY, C_TEXT),
            ("sa_softmax", "Softmax \u2192 Weights", C_ATTN, C_TEXT),
            ("sa_weighted", "Weighted Sum: W \u00b7 V", C_ATTN, C_TEXT),
            ("sa_merge", "Merge Heads\n(B, Seq, d_embed)", C_NORM, C_TEXT),
            ("sa_out_proj", "Output Projection\nLinear \u2192 d_embed",
             C_ATTN, C_TEXT),
        ]
        for nid, lbl, fc, tc in sa_nodes:
            sa.node(nid, lbl, fillcolor=fc, fontcolor=tc)
        for i in range(len(sa_nodes) - 1):
            sa.edge(sa_nodes[i][0], sa_nodes[i + 1][0], color=C_ATTN)

    # --- Cross-Attention ---
    with g.subgraph(name="cluster_ca") as ca:
        ca.attr(
            label="Multi-Head Cross-Attention",
            labelloc="t", fontsize="14", fontname=_FONT_BOLD,
            fontcolor=C_CROSS,
            style="rounded,dashed", color=C_CROSS, bgcolor="#f5f5f5",
        )
        ca.node("ca_x", "Queries x (image)\n(B, H\u00b7W, d_embed)",
                fillcolor=C_UNET, fontcolor=C_TEXT)
        ca.node("ca_y", "Context y (text)\n(B, 77, 768)",
                fillcolor=C_CLIP, fontcolor=C_TEXT)

        ca.node("ca_q", "Q = W_q \u00b7 x",
                fillcolor=C_CROSS, fontcolor=C_TEXT)
        ca.node("ca_k", "K = W_k \u00b7 y",
                fillcolor=C_CROSS, fontcolor=C_TEXT)
        ca.node("ca_v", "V = W_v \u00b7 y",
                fillcolor=C_CROSS, fontcolor=C_TEXT)

        ca_rest = [
            ("ca_reshape", "Reshape\n(B, n_heads, Seq, d_head)",
             C_NORM, C_TEXT),
            ("ca_scores",
             "Attention Scores\nQ \u00b7 K\u1d40 / \u221ad_head",
             C_RESID, C_TEXT),
            ("ca_softmax", "Softmax \u2192 Weights\n(no causal mask)",
             C_CROSS, C_TEXT),
            ("ca_weighted", "Weighted Sum: W \u00b7 V",
             C_CROSS, C_TEXT),
            ("ca_merge", "Merge Heads", C_NORM, C_TEXT),
            ("ca_out_proj", "Output Projection\nLinear \u2192 d_embed",
             C_CROSS, C_TEXT),
            ("ca_out", "Output\n(B, SeqLen_Q, d_embed)",
             C_LIGHT_BG, C_DARK_TEXT),
        ]
        for nid, lbl, fc, tc in ca_rest:
            ca.node(nid, lbl, fillcolor=fc, fontcolor=tc)

        ca.edge("ca_x", "ca_q", color=C_UNET)
        ca.edge("ca_y", "ca_k", color=C_CLIP)
        ca.edge("ca_y", "ca_v", color=C_CLIP)
        ca.edge("ca_q", "ca_reshape", color=C_CROSS)
        ca.edge("ca_k", "ca_reshape", color=C_CROSS)
        ca.edge("ca_v", "ca_reshape", color=C_CROSS)
        for i in range(len(ca_rest) - 1):
            ca.edge(ca_rest[i][0], ca_rest[i + 1][0], color=C_CROSS)

    _render(g, "attention_mechanisms")


# ===================================================================
# 5. UNET Attention Block
# ===================================================================


def draw_unet_attention_block() -> None:
    """Self-Attn -> Cross-Attn -> GeGLU FFN with residual connections.

    The ``(+) Residual`` nodes live **outside** their respective clusters
    so that the skip-connection arrows route *below* each sub-block
    instead of alongside it.
    """
    g = graphviz.Digraph(
        "unet_attn_block",
        graph_attr=dict(
            rankdir="TB",
            label="U-Net Transformer Attention Block",
            labelloc="t", fontsize="18", fontname=_FONT_BOLD,
            fontcolor=C_DARK_TEXT, bgcolor="white",
            pad="0.4", nodesep="0.5", ranksep="0.45", dpi="180",
            splines="ortho",
        ),
        node_attr=_NODE_DEFAULTS,
        edge_attr=dict(fontname=_FONT, fontsize="8"),
    )

    # ── Pre-processing ─────────────────────────────────────────────
    g.node("feat_in", "Feature Map  (B, C, H, W)",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)
    g.node("gn", "GroupNorm(32, C)", fillcolor=C_GRAY, fontcolor=C_TEXT)
    g.node("conv_in", "Conv 1\u00d71  (channel proj)",
           fillcolor=C_UNET, fontcolor=C_TEXT)
    g.node("flatten", "Flatten \u2192 (B, H\u00b7W, C)",
           fillcolor=C_NORM, fontcolor=C_TEXT)

    g.edge("feat_in", "gn", color=C_UNET)
    g.edge("gn", "conv_in", color=C_UNET)
    g.edge("conv_in", "flatten", color=C_UNET)

    # ── Self-Attention sub-block ───────────────────────────────────
    with g.subgraph(name="cluster_sa") as sa:
        sa.attr(
            label="Self-Attention", fontsize="12",
            fontname=_FONT_BOLD, fontcolor=C_ATTN,
            style="rounded,dashed", color=C_ATTN, bgcolor="#eef2f5",
        )
        sa.node("sa_ln", "LayerNorm",
                fillcolor=C_NORM, fontcolor=C_TEXT)
        sa.node("sa_attn", "Multi-Head Self-Attention\n(8 heads)",
                fillcolor=C_ATTN, fontcolor=C_TEXT)
        sa.edge("sa_ln", "sa_attn", color=C_ATTN)

    # (+) Residual BELOW cluster_sa
    g.node("sa_res", "(+) Residual",
           fillcolor=C_RESID, fontcolor=C_TEXT)

    g.edge("flatten", "sa_ln", color=C_UNET)
    g.edge("sa_attn", "sa_res", color=C_ATTN)
    # Skip: flatten → sa_res  (routes below SA block, left side)
    g.edge("flatten:w", "sa_res:w", color=C_SKIP, style="dashed",
           xlabel="residual", fontcolor=C_SKIP, constraint="false")

    # ── Cross-Attention sub-block ──────────────────────────────────
    with g.subgraph(name="cluster_ca") as ca:
        ca.attr(
            label="Cross-Attention", fontsize="12",
            fontname=_FONT_BOLD, fontcolor=C_CLIP,
            style="rounded,dashed", color=C_CLIP, bgcolor="#eef2f5",
        )
        ca.node("ca_ln", "LayerNorm",
                fillcolor=C_NORM, fontcolor=C_TEXT)
        ca.node("ca_attn", "Multi-Head Cross-Attention\n(8 heads)",
                fillcolor=C_CLIP_DARK, fontcolor=C_TEXT)
        ca.edge("ca_ln", "ca_attn", color=C_CLIP)

    # (+) Residual BELOW cluster_ca
    g.node("ca_res", "(+) Residual",
           fillcolor=C_RESID, fontcolor=C_TEXT)

    # CLIP context → right side so it doesn't collide with residuals
    g.node("clip_ctx", "CLIP Context\n(B, 77, 768)",
           fillcolor=C_CLIP, fontcolor=C_TEXT)
    g.edge("clip_ctx:s", "ca_attn:e", color=C_CLIP, style="dashed",
           xlabel="K, V from text", fontcolor=C_CLIP,
           constraint="false")

    g.edge("sa_res", "ca_ln", color=C_UNET)
    g.edge("ca_attn", "ca_res", color=C_CLIP)
    # Skip: sa_res → ca_res  (routes below CA block, left side)
    g.edge("sa_res:w", "ca_res:w", color=C_SKIP, style="dashed",
           constraint="false")

    # ── GeGLU FFN sub-block ────────────────────────────────────────
    with g.subgraph(name="cluster_ff") as ff:
        ff.attr(
            label="GeGLU Feed-Forward", fontsize="12",
            fontname=_FONT_BOLD, fontcolor=C_GEGLU,
            style="rounded,dashed", color=C_GEGLU, bgcolor="#fef9e7",
        )
        ff.node("ff_ln", "LayerNorm",
                fillcolor=C_NORM, fontcolor=C_TEXT)
        ff.node("ff_up", "Linear \u2192 8\u00b7C\n(split: value + gate)",
                fillcolor=C_GEGLU, fontcolor=C_TEXT)
        ff.node("ff_geglu", "value \u00d7 GELU(gate)",
                fillcolor=C_GEGLU, fontcolor=C_TEXT)
        ff.node("ff_down", "Linear \u2192 C",
                fillcolor=C_GEGLU, fontcolor=C_TEXT)
        ff.edge("ff_ln", "ff_up", color=C_GEGLU)
        ff.edge("ff_up", "ff_geglu", color=C_GEGLU)
        ff.edge("ff_geglu", "ff_down", color=C_GEGLU)

    # (+) Residual BELOW cluster_ff
    g.node("ff_res", "(+) Residual",
           fillcolor=C_RESID, fontcolor=C_TEXT)

    g.edge("ca_res", "ff_ln", color=C_UNET)
    g.edge("ff_down", "ff_res", color=C_GEGLU)
    # Skip: ca_res → ff_res  (routes below FFN block, left side)
    g.edge("ca_res:w", "ff_res:w", color=C_SKIP, style="dashed",
           constraint="false")

    # ── Post-processing ────────────────────────────────────────────
    g.node("reshape_out", "Reshape \u2192 (B, C, H, W)",
           fillcolor=C_NORM, fontcolor=C_TEXT)
    g.node("conv_out", "Conv 1\u00d71 + Long Residual",
           fillcolor=C_UNET, fontcolor=C_TEXT)

    g.edge("ff_res", "reshape_out", color=C_UNET)
    g.edge("reshape_out", "conv_out", color=C_UNET)
    # Long residual → far left (:w) separate from short residuals
    g.edge("feat_in:w", "conv_out:w", color=C_SKIP, style="dashed",
           xlabel="long residual", fontcolor=C_SKIP,
           constraint="false")

    _render(g, "unet_attention_block")


# ===================================================================
# 6. Residual Blocks
# ===================================================================


def draw_residual_blocks() -> None:
    """VAE ResBlock and U-Net ResBlock (with time conditioning)."""
    g = graphviz.Digraph(
        "residual_blocks",
        graph_attr=dict(
            rankdir="TB",
            label="Residual Blocks",
            labelloc="t", fontsize="18", fontname=_FONT_BOLD,
            fontcolor=C_DARK_TEXT, bgcolor="white",
            pad="0.4", nodesep="0.25", ranksep="0.35", dpi="180",
            splines="ortho",
        ),
        node_attr=_NODE_DEFAULTS,
        edge_attr=dict(fontname=_FONT, fontsize="8"),
    )

    # --- VAE Residual Block ---
    with g.subgraph(name="cluster_vae_res") as vae:
        vae.attr(
            label="VAE Residual Block  (vae_utils.py)",
            labelloc="t", fontsize="13", fontname=_FONT_BOLD,
            fontcolor=C_VAE_ENC,
            style="rounded,dashed", color=C_VAE_ENC, bgcolor="#fdf2e9",
        )
        vae_nodes = [
            ("vr_in", "Input (B, in_ch, H, W)", C_LIGHT_BG, C_DARK_TEXT),
            ("vr_gn1", "GroupNorm(32)", C_GRAY, C_TEXT),
            ("vr_silu1", "SiLU", C_ACT, C_TEXT),
            ("vr_conv1", "Conv 3\u00d73  in\u2192out",
             C_VAE_ENC, C_TEXT),
            ("vr_gn2", "GroupNorm(32)", C_GRAY, C_TEXT),
            ("vr_silu2", "SiLU", C_ACT, C_TEXT),
            ("vr_conv2", "Conv 3\u00d73  out\u2192out",
             C_VAE_ENC, C_TEXT),
            ("vr_add", "(+) Add", C_RESID, C_TEXT),
            ("vr_out", "Output (B, out_ch, H, W)",
             C_LIGHT_BG, C_DARK_TEXT),
        ]
        for nid, lbl, fc, tc in vae_nodes:
            vae.node(nid, lbl, fillcolor=fc, fontcolor=tc)
        for i in range(len(vae_nodes) - 1):
            vae.edge(vae_nodes[i][0], vae_nodes[i + 1][0],
                     color=C_VAE_ENC)
        vae.edge("vr_in", "vr_add", color=C_SKIP, style="dashed",
                 xlabel="Identity or\nConv 1\u00d71",
                 fontcolor=C_SKIP, constraint="false")

    # --- U-Net Residual Block ---
    with g.subgraph(name="cluster_unet_res") as unet:
        unet.attr(
            label="U-Net Residual Block  (unet_utils.py)",
            labelloc="t", fontsize="13", fontname=_FONT_BOLD,
            fontcolor=C_UNET,
            style="rounded,dashed", color=C_UNET, bgcolor="#f4ecf7",
        )
        unet_nodes = [
            ("ur_in", "Feature (B, in_ch, H, W)",
             C_LIGHT_BG, C_DARK_TEXT),
            ("ur_gn1", "GroupNorm(32)", C_GRAY, C_TEXT),
            ("ur_silu1", "SiLU", C_ACT, C_TEXT),
            ("ur_conv1", "Conv 3\u00d73  in\u2192out", C_UNET, C_TEXT),
            ("ur_add_t", "(+) Add  (feature + time)",
             C_TIME, C_DARK_TEXT),
            ("ur_gn2", "GroupNorm(32)", C_GRAY, C_TEXT),
            ("ur_silu2", "SiLU", C_ACT, C_TEXT),
            ("ur_conv2", "Conv 3\u00d73  out\u2192out", C_UNET, C_TEXT),
            ("ur_add", "(+) Add  (+ residual)", C_RESID, C_TEXT),
            ("ur_out", "Output (B, out_ch, H, W)",
             C_LIGHT_BG, C_DARK_TEXT),
        ]
        for nid, lbl, fc, tc in unet_nodes:
            unet.node(nid, lbl, fillcolor=fc, fontcolor=tc)
        for i in range(len(unet_nodes) - 1):
            unet.edge(unet_nodes[i][0], unet_nodes[i + 1][0],
                      color=C_UNET)

        # Time embedding branch
        unet.node("ur_time", "Time (B, 1280)",
                  fillcolor=C_TIME, fontcolor=C_DARK_TEXT)
        unet.node("ur_time_proj",
                  "SiLU \u2192 Linear\n1280 \u2192 out_ch",
                  fillcolor=C_TIME, fontcolor=C_DARK_TEXT)
        unet.edge("ur_time", "ur_time_proj", color=C_TIME)
        unet.edge("ur_time_proj", "ur_add_t",
                  color=C_TIME, style="dashed")

        # Skip connection
        unet.edge("ur_in", "ur_add", color=C_SKIP, style="dashed",
                  xlabel="Identity or\nConv 1\u00d71",
                  fontcolor=C_SKIP, constraint="false")

    _render(g, "residual_blocks")


# ===================================================================
# 7. CLIP Text Encoder
# ===================================================================


def draw_clip() -> None:
    """CLIP text encoder: embedding + 12 transformer layers."""
    g = graphviz.Digraph(
        "clip",
        graph_attr=dict(
            rankdir="TB",
            label="CLIP Text Encoder  (clip.py)",
            labelloc="t", fontsize="18", fontname=_FONT_BOLD,
            fontcolor=C_DARK_TEXT, bgcolor="white",
            pad="0.4", nodesep="0.3", ranksep="0.4", dpi="180",
            splines="ortho",
        ),
        node_attr=_NODE_DEFAULTS,
        edge_attr=dict(fontname=_FONT, fontsize="8"),
    )

    # Input
    g.node("tokens", "Input Tokens  (B, 77)",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)

    # Embedding cluster
    with g.subgraph(name="cluster_embed") as emb:
        emb.attr(
            label="CLIPEmbedding", fontsize="12",
            fontname=_FONT_BOLD, fontcolor=C_CLIP,
            style="rounded,dashed", color=C_CLIP, bgcolor="#eaf2fa",
        )
        emb.node("tok_emb",
                 "Token Embedding\nnn.Embedding(49408, 768)",
                 fillcolor=C_CLIP, fontcolor=C_TEXT)
        emb.node("pos_emb",
                 "Position Embedding\nnn.Parameter(77, 768)",
                 fillcolor=C_CLIP, fontcolor=C_TEXT)
        emb.node("emb_add", "(+) Add",
                 fillcolor=C_CLIP_DARK, fontcolor=C_TEXT)
        emb.edge("tok_emb", "emb_add", color=C_CLIP)
        emb.edge("pos_emb", "emb_add", color=C_CLIP)

    g.edge("tokens", "tok_emb", color=C_CLIP)
    g.edge("tokens", "pos_emb", color=C_CLIP)

    # Transformer block (x12)
    with g.subgraph(name="cluster_tf") as tf:
        tf.attr(
            label="CLIPLayer (Transformer Block)  \u00d712",
            fontsize="13", fontname=_FONT_BOLD, fontcolor=C_CLIP,
            style="rounded,bold", color=C_CLIP, bgcolor="#eaf2fa",
        )
        tf_nodes = [
            ("tf_ln1", "LayerNorm", C_NORM, C_TEXT),
            ("tf_sa", "Causal Self-Attention\n(12 heads \u00d7 64-dim)",
             C_ATTN, C_TEXT),
            ("tf_res1", "(+) Residual", C_RESID, C_TEXT),
            ("tf_ln2", "LayerNorm", C_NORM, C_TEXT),
            ("tf_fc1", "Linear 768 \u2192 3072 (4\u00d7)",
             C_CLIP, C_TEXT),
            ("tf_gelu", "QuickGELU: x \u00b7 \u03c3(1.702\u00b7x)",
             C_ACT, C_TEXT),
            ("tf_fc2", "Linear 3072 \u2192 768", C_CLIP, C_TEXT),
            ("tf_res2", "(+) Residual", C_RESID, C_TEXT),
        ]
        for nid, lbl, fc, tc in tf_nodes:
            tf.node(nid, lbl, fillcolor=fc, fontcolor=tc)
        for i in range(len(tf_nodes) - 1):
            tf.edge(tf_nodes[i][0], tf_nodes[i + 1][0], color=C_CLIP)

        # Residual skip connections
        tf.edge("tf_ln1", "tf_res1", color=C_SKIP, style="dashed",
                constraint="false", xlabel="skip", fontcolor=C_SKIP)
        tf.edge("tf_ln2", "tf_res2", color=C_SKIP, style="dashed",
                constraint="false", xlabel="skip", fontcolor=C_SKIP)

    g.edge("emb_add", "tf_ln1", color=C_CLIP, penwidth="2")

    # Final norm + output
    g.node("final_ln", "Final LayerNorm",
           fillcolor=C_NORM, fontcolor=C_TEXT)
    g.node("output", "Context Embeddings\n(B, 77, 768)",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)

    g.edge("tf_res2", "final_ln", color=C_CLIP, penwidth="2")
    g.edge("final_ln", "output", color=C_CLIP, penwidth="2")

    _render(g, "clip_text_encoder")


# ===================================================================
# 8. Diffusion Process (Denoising Loop)
# ===================================================================
# Main
# ===================================================================


def main() -> None:
    """Generate all architecture diagrams."""
    print("Generating architecture diagrams ...")
    draw_pipeline()
    draw_unet()
    draw_vae()
    draw_attention()
    draw_unet_attention_block()
    draw_residual_blocks()
    draw_clip()
    print(f"\nAll diagrams saved to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
