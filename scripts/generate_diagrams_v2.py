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
    """High-level pipeline: Text -> CLIP -> U-Net <- VAE <-> Image."""
    g = graphviz.Digraph(
        "pipeline",
        graph_attr=dict(
            rankdir="TB",
            label="Stable Diffusion \u2014 Pipeline Overview",
            labelloc="t",
            fontsize="18",
            fontname=_FONT_BOLD,
            fontcolor=C_DARK_TEXT,
            bgcolor="white",
            pad="0.4",
            nodesep="0.5",
            ranksep="0.6",
            dpi="180",
        ),
        node_attr=_NODE_DEFAULTS,
        edge_attr=dict(fontname=_FONT, fontsize="8"),
    )

    # Inputs
    g.node("prompt", "Text Prompt",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)
    g.node("image_in", "Input Image\n(optional, for img2img)",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)
    g.node("noise", "Random Noise\nz_T ~ N(0, I)",
           fillcolor="#555555", fontcolor=C_TEXT)

    # CLIP
    g.node("tokenizer", "BPE Tokenizer\n(B, 77)",
           fillcolor=C_CLIP_DARK, fontcolor=C_TEXT)
    g.node("clip", "CLIP Text Encoder\n12\u00d7 Transformer Blocks\n\u2192 (B, 77, 768)",
           fillcolor=C_CLIP, fontcolor=C_TEXT)

    # VAE Encoder
    g.node("vae_enc",
           "VAE Encoder\n3\u2192128\u2192256\u2192512\u21924 ch\n\u21938\u00d7 spatial",
           fillcolor=C_VAE_ENC, fontcolor=C_TEXT)

    # Time
    g.node("timestep", "Timestep  t",
           fillcolor=C_TIME, fontcolor=C_DARK_TEXT)
    g.node("time_emb", "Time Embedding\nSinusoidal \u2192 MLP\n320 \u2192 1280",
           fillcolor=C_TIME, fontcolor=C_DARK_TEXT)

    # U-Net
    g.node("unet",
           "Diffusion U-Net\nEncoder \u2192 Bottleneck \u2192 Decoder\n"
           "conditioned on text + timestep",
           fillcolor=C_UNET, fontcolor=C_TEXT,
           width="3.5", height="0.9")

    # VAE Decoder
    g.node("vae_dec",
           "VAE Decoder\n4\u2192512\u2192256\u2192128\u21923 ch\n\u21918\u00d7 spatial",
           fillcolor=C_VAE_DEC, fontcolor=C_TEXT)

    # Output
    g.node("image_out", "Generated Image",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)

    # Edges
    g.edge("prompt", "tokenizer", color=C_CLIP)
    g.edge("tokenizer", "clip", color=C_CLIP)
    g.edge("clip", "unet", label="  context\n  (B, 77, 768)",
           color=C_CLIP, style="dashed")
    g.edge("image_in", "vae_enc", color=C_VAE_ENC)
    g.edge("vae_enc", "unet", label="  latent z\n  (B, 4, H/8, W/8)",
           color=C_VAE_ENC)
    g.edge("noise", "unet", label="  z_T", color="#999999", style="dashed")
    g.edge("timestep", "time_emb", color=C_TIME)
    g.edge("time_emb", "unet", label="  (B, 1280)",
           color=C_TIME, style="dashed")
    g.edge("unet", "vae_dec",
           label="  denoised latent z_0\n  (B, 4, H/8, W/8)", color=C_UNET)
    g.edge("vae_dec", "image_out", color=C_VAE_DEC)

    # Horizontal alignment hints
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("prompt")
        s.node("image_in")
        s.node("noise")
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("timestep")
        s.node("clip")

    _render(g, "pipeline_overview")


# ===================================================================
# 2. U-Net Architecture
# ===================================================================


def draw_unet() -> None:
    """U-Net encoder-bottleneck-decoder with skip connections."""
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
            nodesep="0.4",
            ranksep="0.55",
            dpi="180",
        ),
        node_attr=_NODE_DEFAULTS,
        edge_attr=dict(fontname=_FONT, fontsize="8"),
    )

    # Inputs
    g.node("input", "Noisy Latent\n(B, 4, H/8, W/8)",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)
    g.node("context", "CLIP Context\n(B, 77, 768)",
           fillcolor=C_CLIP, fontcolor=C_TEXT)
    g.node("time", "Time Embedding\n(B, 1280)",
           fillcolor=C_TIME, fontcolor=C_DARK_TEXT)

    # Encoder stages
    enc_info = [
        ("enc0", "Conv 4\u2192320\n+ 2\u00d7(ResBlock + Attention)\nH/8 \u00b7 320 ch"),
        ("enc1", "Downsample \u21932\n2\u00d7(Res 320\u2192640 + Attn)\nH/16 \u00b7 640 ch"),
        ("enc2", "Downsample \u21932\n2\u00d7(Res 640\u21921280 + Attn)\nH/32 \u00b7 1280 ch"),
        ("enc3", "Downsample \u21932\n2\u00d7 ResBlock\nH/64 \u00b7 1280 ch"),
    ]
    for nid, lbl in enc_info:
        g.node(nid, lbl, fillcolor=C_UNET_ENC, fontcolor=C_TEXT)

    # Bottleneck
    g.node("bn",
           "Bottleneck\nResBlock \u2192 Self-Attention(8h, 160d) \u2192 ResBlock\n"
           "1280 ch \u00b7 H/64",
           fillcolor=C_UNET_BN, fontcolor=C_TEXT, width="4")

    # Decoder stages
    dec_info = [
        ("dec0", "2\u00d7 Res + Upsample \u21912\nH/64\u2192H/32 \u00b7 1280 ch"),
        ("dec1", "2\u00d7(Res+Attn) + Upsample \u21912\nH/32\u2192H/16 \u00b7 1280 ch"),
        ("dec2", "2\u00d7(Res+Attn) + Upsample \u21912\nH/16\u2192H/8 \u00b7 640 ch"),
        ("dec3", "3\u00d7(ResBlock + Attention)\nH/8 \u00b7 320 ch"),
    ]
    for nid, lbl in dec_info:
        g.node(nid, lbl, fillcolor=C_UNET_DEC, fontcolor=C_TEXT)

    # Output
    g.node("out_conv", "GroupNorm \u2192 SiLU \u2192 Conv 320\u21924",
           fillcolor=C_GRAY, fontcolor=C_TEXT)
    g.node("output", "Predicted Noise\n(B, 4, H/8, W/8)",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)

    # Main flow
    g.edge("input", "enc0", color=C_UNET, penwidth="2")
    for i in range(3):
        g.edge(f"enc{i}", f"enc{i+1}", color=C_UNET, penwidth="2")
    g.edge("enc3", "bn", color=C_UNET, penwidth="2")
    g.edge("bn", "dec0", color=C_UNET, penwidth="2")
    for i in range(3):
        g.edge(f"dec{i}", f"dec{i+1}", color=C_UNET, penwidth="2")
    g.edge("dec3", "out_conv", color=C_UNET, penwidth="2")
    g.edge("out_conv", "output", color=C_UNET, penwidth="2")

    # Skip connections
    skip_pairs = [
        ("enc3", "dec0", "skip 1280ch"),
        ("enc2", "dec1", "skip 1280ch"),
        ("enc1", "dec2", "skip 640ch"),
        ("enc0", "dec3", "skip 320ch"),
    ]
    skip_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
    for (src, dst, lbl), col in zip(skip_pairs, skip_colors):
        g.edge(src, dst, label=f"  {lbl}  ", color=col, style="dashed",
               penwidth="1.8", fontcolor=col, constraint="false")

    # -- Conditioning legend (replaces individual dotted arrows to reduce
    #    visual clutter).  A compact annotation on the right explains
    #    which stages receive time / context conditioning.
    g.node(
        "cond_legend",
        (
            "Conditioning\n"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            "\u23f1 Time Emb \u2192 all ResBlocks\n"
            "(additive, after first conv)\n\n"
            "\U0001f4dd CLIP Context \u2192 cross-attn\n"
            "in enc0-2, bn, dec1-3"
        ),
        shape="note",
        style="filled",
        fillcolor="#f9f9f9",
        fontcolor=C_DARK_TEXT,
        fontname=_FONT,
        fontsize="9",
        color="#cccccc",
    )

    # Light arrows from context / time to the legend to visually connect
    g.edge("context", "cond_legend", style="dashed", color=C_CLIP,
           arrowhead="none", constraint="false")
    g.edge("time", "cond_legend", style="dashed", color=C_TIME,
           arrowhead="none", constraint="false")

    # Horizontal alignment for inputs
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("input")
        s.node("context")
        s.node("time")
        s.node("cond_legend")

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
            nodesep="0.25",
            ranksep="0.35",
            dpi="180",
            compound="true",
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

    # Link encoder output to decoder input
    g.edge("e_out", "d_in", label="  latent space  ",
           color="#555555", style="dashed", penwidth="2",
           ltail="cluster_enc", lhead="cluster_dec")

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
    """Self-Attn -> Cross-Attn -> GeGLU FFN with residual connections."""
    g = graphviz.Digraph(
        "unet_attn_block",
        graph_attr=dict(
            rankdir="TB",
            label="U-Net Transformer Attention Block",
            labelloc="t", fontsize="18", fontname=_FONT_BOLD,
            fontcolor=C_DARK_TEXT, bgcolor="white",
            pad="0.4", nodesep="0.25", ranksep="0.35", dpi="180",
        ),
        node_attr=_NODE_DEFAULTS,
        edge_attr=dict(fontname=_FONT, fontsize="8"),
    )

    # Pre-processing
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

    # --- Self-Attention sub-block ---
    with g.subgraph(name="cluster_sa") as sa:
        sa.attr(
            label="Self-Attention", fontsize="12",
            fontname=_FONT_BOLD, fontcolor=C_ATTN,
            style="rounded,dashed", color=C_ATTN, bgcolor="#eef2f5",
        )
        sa.node("sa_ln", "LayerNorm", fillcolor=C_NORM, fontcolor=C_TEXT)
        sa.node("sa_attn", "Multi-Head Self-Attention\n(8 heads)",
                fillcolor=C_ATTN, fontcolor=C_TEXT)
        sa.node("sa_res", "(+) Residual",
                fillcolor=C_RESID, fontcolor=C_TEXT)
        sa.edge("sa_ln", "sa_attn", color=C_ATTN)
        sa.edge("sa_attn", "sa_res", color=C_ATTN)

    g.edge("flatten", "sa_ln", color=C_UNET)
    g.edge("flatten", "sa_res", color=C_SKIP, style="dashed",
           label="  residual", fontcolor=C_SKIP, constraint="false")

    # --- Cross-Attention sub-block ---
    with g.subgraph(name="cluster_ca") as ca:
        ca.attr(
            label="Cross-Attention", fontsize="12",
            fontname=_FONT_BOLD, fontcolor=C_CLIP,
            style="rounded,dashed", color=C_CLIP, bgcolor="#eef2f5",
        )
        ca.node("ca_ln", "LayerNorm", fillcolor=C_NORM, fontcolor=C_TEXT)
        ca.node("ca_attn", "Multi-Head Cross-Attention\n(8 heads)",
                fillcolor=C_CLIP_DARK, fontcolor=C_TEXT)
        ca.node("ca_res", "(+) Residual",
                fillcolor=C_RESID, fontcolor=C_TEXT)
        ca.edge("ca_ln", "ca_attn", color=C_CLIP)
        ca.edge("ca_attn", "ca_res", color=C_CLIP)

    g.node("clip_ctx", "CLIP Context\n(B, 77, 768)",
           fillcolor=C_CLIP, fontcolor=C_TEXT)
    g.edge("clip_ctx", "ca_attn", color=C_CLIP, style="dashed",
           label="  K, V from text", fontcolor=C_CLIP, constraint="false")
    g.edge("sa_res", "ca_ln", color=C_UNET)
    g.edge("sa_res", "ca_res", color=C_SKIP, style="dashed",
           constraint="false")

    # --- GeGLU FFN sub-block ---
    with g.subgraph(name="cluster_ff") as ff:
        ff.attr(
            label="GeGLU Feed-Forward", fontsize="12",
            fontname=_FONT_BOLD, fontcolor=C_GEGLU,
            style="rounded,dashed", color=C_GEGLU, bgcolor="#fef9e7",
        )
        ff.node("ff_ln", "LayerNorm", fillcolor=C_NORM, fontcolor=C_TEXT)
        ff.node("ff_up", "Linear \u2192 8\u00b7C\n(split: value + gate)",
                fillcolor=C_GEGLU, fontcolor=C_TEXT)
        ff.node("ff_geglu", "value \u00d7 GELU(gate)",
                fillcolor=C_GEGLU, fontcolor=C_TEXT)
        ff.node("ff_down", "Linear \u2192 C",
                fillcolor=C_GEGLU, fontcolor=C_TEXT)
        ff.node("ff_res", "(+) Residual",
                fillcolor=C_RESID, fontcolor=C_TEXT)
        ff.edge("ff_ln", "ff_up", color=C_GEGLU)
        ff.edge("ff_up", "ff_geglu", color=C_GEGLU)
        ff.edge("ff_geglu", "ff_down", color=C_GEGLU)
        ff.edge("ff_down", "ff_res", color=C_GEGLU)

    g.edge("ca_res", "ff_ln", color=C_UNET)
    g.edge("ca_res", "ff_res", color=C_SKIP, style="dashed",
           constraint="false")

    # Output
    g.node("reshape_out", "Reshape \u2192 (B, C, H, W)",
           fillcolor=C_NORM, fontcolor=C_TEXT)
    g.node("conv_out", "Conv 1\u00d71 + Long Residual",
           fillcolor=C_UNET, fontcolor=C_TEXT)

    g.edge("ff_res", "reshape_out", color=C_UNET)
    g.edge("reshape_out", "conv_out", color=C_UNET)
    g.edge("feat_in", "conv_out", color=C_SKIP, style="dashed",
           label="  long residual", fontcolor=C_SKIP, constraint="false")

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
                 label="Identity or\nConv 1\u00d71",
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
                  label="Identity or\nConv 1\u00d71",
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
                constraint="false", label="  skip")
        tf.edge("tf_ln2", "tf_res2", color=C_SKIP, style="dashed",
                constraint="false", label="  skip")

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


def draw_diffusion_process() -> None:
    """Iterative denoising: z_T -> z_0 via scheduler and U-Net."""
    g = graphviz.Digraph(
        "diffusion_process",
        graph_attr=dict(
            rankdir="LR",
            label="Diffusion Denoising Process",
            labelloc="t", fontsize="18", fontname=_FONT_BOLD,
            fontcolor=C_DARK_TEXT, bgcolor="white",
            pad="0.5", nodesep="0.4", ranksep="0.7", dpi="180",
        ),
        node_attr=_NODE_DEFAULTS,
        edge_attr=dict(fontname=_FONT, fontsize="8"),
    )

    # --- Inputs (left) ---
    g.node("prompt", "Text\nPrompt",
           fillcolor="#555555", fontcolor=C_TEXT)
    g.node("clip", "CLIP\nEncoder", fillcolor=C_CLIP, fontcolor=C_TEXT)
    g.node("ctx", "Context\n(B, 77, 768)",
           fillcolor=C_CLIP, fontcolor=C_TEXT)
    g.node("noise", "Random\nNoise z_T",
           fillcolor="#555555", fontcolor=C_TEXT)
    g.node("vae_enc", "VAE\nEncoder",
           fillcolor=C_VAE_ENC, fontcolor=C_TEXT)
    g.node("img_in", "Input Image\n(optional)",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)

    g.edge("prompt", "clip", color=C_CLIP, penwidth="2")
    g.edge("clip", "ctx", color=C_CLIP, penwidth="2")
    g.edge("img_in", "vae_enc", color=C_VAE_ENC, penwidth="2")

    # --- Scheduler loop (centre) ---
    with g.subgraph(name="cluster_loop") as loop:
        loop.attr(
            label="Scheduler  (T denoising steps)",
            fontsize="13", fontname=_FONT_BOLD, fontcolor=C_TIME,
            style="rounded,bold", color=C_TIME, bgcolor="#fef9e7",
        )

        steps = [
            ("t_T", "t = T", "#e74c3c"),
            ("t_T1", "t = T\u22121", "#e67e22"),
            ("t_dots", "\u00b7 \u00b7 \u00b7", "#f39c12"),
            ("t_1", "t = 1", "#27ae60"),
        ]
        for nid, lbl, col in steps:
            tc = C_TEXT if col != "#f39c12" else C_DARK_TEXT
            loop.node(nid, lbl, fillcolor=col, fontcolor=tc, width="0.8")
        for i in range(len(steps) - 1):
            loop.edge(steps[i][0], steps[i + 1][0],
                      color=C_TIME, penwidth="2")

        loop.node(
            "unet",
            "U-Net  \u03b5_\u03b8(z_t, t, ctx)\n\n"
            "Predicts noise at each step",
            fillcolor=C_UNET, fontcolor=C_TEXT, width="2.5",
        )
        loop.node("sched_step", "z_{t\u22121} = Scheduler(z_t, \u03b5)",
                  fillcolor=C_TIME, fontcolor=C_DARK_TEXT)

        for nid, _, _ in steps:
            loop.edge(nid, "unet", color=C_UNET, style="dashed")
        loop.edge("unet", "sched_step", color=C_UNET, penwidth="2")

    # Feed context and noise into loop
    g.edge("ctx", "unet", label="  cross-attn", color=C_CLIP,
           style="dashed", fontcolor=C_CLIP)
    g.edge("noise", "t_T", color="#999999", penwidth="2")
    g.edge("vae_enc", "t_T", color=C_VAE_ENC, style="dashed",
           label="  latent z")

    # --- Output (right) ---
    g.node("z0", "Clean Latent\nz_0",
           fillcolor="#27ae60", fontcolor=C_TEXT)
    g.node("vae_dec", "VAE\nDecoder",
           fillcolor=C_VAE_DEC, fontcolor=C_TEXT)
    g.node("img_out", "Generated\nImage",
           fillcolor=C_LIGHT_BG, fontcolor=C_DARK_TEXT)

    g.edge("sched_step", "z0", color=C_TIME, penwidth="2")
    g.edge("z0", "vae_dec", color=C_VAE_DEC, penwidth="2")
    g.edge("vae_dec", "img_out", color=C_VAE_DEC, penwidth="2")

    _render(g, "diffusion_process")


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
    draw_diffusion_process()
    print(f"\nAll diagrams saved to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
