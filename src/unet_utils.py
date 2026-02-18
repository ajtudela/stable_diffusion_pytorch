
import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention, CrossAttention


class UNET_ResidualBlock(nn.Module):
    """
    Residual block with time-step conditioning for the U-Net.

    Processes spatial features and injects time information so the
    network can adapt its behaviour to the current noise level.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_time: int = 1280
    ) -> None:
        """
        Initialize the U-Net residual block with time conditioning.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input feature map.
        out_channels : int
            Number of channels in the output feature map.
        n_time : int
            Dimensionality of the time-conditioning vector
            (1280 by default, matching ``TimeEmbedding`` output).
        """
        super().__init__()

        # GroupNorm normalises the input features (32 groups) to
        # stabilise training, independent of batch size.
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)

        # 3x3 conv transforms feature channels from in_channels to
        # out_channels. padding=1 preserves spatial dimensions.
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)

        # Linear layer projects the 1280-dim time embedding to
        # out_channels so it can be added element-wise to the spatial
        # feature map (after broadcasting over H and W).
        self.linear_time = nn.Linear(n_time, out_channels)

        # GroupNorm applied after merging features and time information.
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)

        # Final 3x3 conv refines the merged representation.
        # Keeps out_channels unchanged.
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection: if in_channels != out_channels we need a
        # 1x1 conv to match dimensions; otherwise identity is used.
        if in_channels == out_channels:
            self.residual_layer: nn.Module = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0)

    def forward(
        self,
        feature: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply residual block with time conditioning.

        Parameters
        ----------
        feature : torch.Tensor
            Spatial features of shape (Batch, In_ch, H, W).
        time : torch.Tensor
            Time-conditioning vector of shape (Batch, 1280).

        Returns
        -------
        torch.Tensor
            Output features of shape (Batch, Out_ch, H, W).
        """
        # feature: (Batch, In_channels, Height, Width)
        # time:    (Batch, 1280)

        # Save input for the skip / residual connection.
        residue = feature

        # Normalise input features across channel groups.
        # (Batch, In_ch, H, W) -> (Batch, In_ch, H, W)
        feature = self.groupnorm_feature(feature)

        # SiLU activation on the normalised features.
        feature = F.silu(feature)

        # Convolve to transform channel count: In_ch -> Out_ch.
        # (Batch, In_ch, H, W) -> (Batch, Out_ch, H, W)
        feature = self.conv_feature(feature)

        # Activate the time embedding with SiLU before projection,
        # adding non-linearity to the conditioning path.
        # (Batch, 1280) -> (Batch, 1280)
        time = F.silu(time)

        # Project time from 1280 to Out_ch so it can be added to the
        # spatial feature map.
        # (Batch, 1280) -> (Batch, Out_ch)
        time = self.linear_time(time)

        # Add time information to every spatial position.
        # unsqueeze(-1).unsqueeze(-1) broadcasts (Batch, Out_ch) to
        # (Batch, Out_ch, 1, 1) which then broadcasts over H and W.
        # (Batch, Out_ch, H, W) + (Batch, Out_ch, 1, 1)
        # -> (Batch, Out_ch, H, W)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        # Normalise the merged features.
        merged = self.groupnorm_merged(merged)

        # SiLU activation before the final convolution.
        merged = F.silu(merged)

        # Final 3x3 conv refines the merged features.
        # (Batch, Out_ch, H, W) -> (Batch, Out_ch, H, W)
        merged = self.conv_merged(merged)

        # Add the skip connection (possibly projected via 1x1 conv).
        # This allows gradients to bypass the block and stabilises
        # training of deep networks.
        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    """
    Transformer-style attention block for the U-Net.

    Applies three sub-blocks in sequence:
      1. Self-attention  -- spatial positions attend to each other.
      2. Cross-attention -- spatial positions attend to text context.
      3. GeGLU feed-forward -- position-wise non-linear transformation.

    Each sub-block uses a pre-norm residual pattern.
    """

    def __init__(self, n_head: int, n_embd: int, d_context: int = 768) -> None:
        """
        Initialize the U-Net transformer attention block.

        Parameters
        ----------
        n_head : int
            Number of attention heads. The total channel count is
            computed as ``n_head * n_embd``.
        n_embd : int
            Embedding dimension per attention head.
        d_context : int
            Dimensionality of the cross-attention context input
            (768 by default, matching the CLIP text encoder output).
        """
        super().__init__()

        # Total channel count = n_head * n_embd (e.g. 8 * 40 = 320).
        channels = n_head * n_embd

        # GroupNorm applied to the 2-D feature map before flattening it
        # into a sequence for the transformer layers. eps=1e-6 for
        # numerical stability.
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)

        # 1x1 conv projects the channel dimension before entering the
        # transformer (acts as a linear layer per spatial position).
        # kernel_size=1, padding=0 keeps spatial dims intact.
        self.conv_input = nn.Conv2d(
            channels, channels, kernel_size=1, padding=0)

        # --- Self-Attention sub-block ---
        # Pre-norm before self-attention.
        self.layernorm_1 = nn.LayerNorm(channels)

        # Multi-head self-attention: lets every spatial position exchange
        # information with every other, capturing global spatial structure.
        # No bias in projections to match the original SD weights.
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

        # --- Cross-Attention sub-block ---
        # Pre-norm before cross-attention.
        self.layernorm_2 = nn.LayerNorm(channels)

        # Multi-head cross-attention: queries come from the image features,
        # keys/values come from the CLIP text context (d_context = 768).
        # This is the mechanism that steers image generation toward the
        # text prompt.
        self.attention_2 = CrossAttention(
            n_head, channels, d_context, in_proj_bias=False)

        # --- GeGLU Feed-Forward sub-block ---
        # Pre-norm before the feed-forward network.
        self.layernorm_3 = nn.LayerNorm(channels)

        # GeGLU first layer: projects to 4*channels*2 because the
        # output is split in half -- one half is the value, the other
        # half is the gate.  GeGLU = value * GELU(gate).
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)

        # Second FFN layer projects back from 4*channels to channels.
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        # 1x1 conv to project the transformer output back to the
        # original channel space before adding the long residual.
        self.conv_output = nn.Conv2d(
            channels, channels, kernel_size=1, padding=0)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply self-attention, cross-attention and feed-forward to the
        feature map, conditioned on the text context.

        Parameters
        ----------
        x : torch.Tensor
            Spatial features of shape (Batch, Channels, H, W).
        context : torch.Tensor
            Text conditioning of shape (Batch, SeqLen, 768).

        Returns
        -------
        torch.Tensor
            Output features of shape (Batch, Channels, H, W).
        """
        # x:       (Batch, Channels, H, W)
        # context: (Batch, SeqLen, d_context)

        # Save for the long skip connection that wraps the entire block.
        residue_long = x

        # Normalise the 2-D feature map.
        x = self.groupnorm(x)

        # 1x1 conv: linear channel projection per spatial position.
        x = self.conv_input(x)

        n, c, h, w = x.shape

        # Flatten spatial dims and transpose to sequence form so that
        # attention layers can operate on a (Batch, SeqLen, Channels)
        # tensor, where SeqLen = H * W.
        # (Batch, C, H, W) -> (Batch, C, H*W) -> (Batch, H*W, C)
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)

        # ---- Self-Attention sub-block (Pre-LN residual) ----

        # Save for the short skip around self-attention.
        residue_short = x

        # Pre-norm before self-attention.
        x = self.layernorm_1(x)

        # Self-attention: every spatial position attends to all others,
        # building global spatial coherence.
        x = self.attention_1(x)

        # Residual connection around self-attention.
        x += residue_short

        # ---- Cross-Attention sub-block (Pre-LN residual) ----

        # Save for the short skip around cross-attention.
        residue_short = x

        # Pre-norm before cross-attention.
        x = self.layernorm_2(x)

        # Cross-attention: spatial positions (queries) attend to text
        # tokens (keys/values), pulling in semantic guidance from the
        # prompt.
        x = self.attention_2(x, context)

        # Residual connection around cross-attention.
        x += residue_short

        # ---- GeGLU Feed-Forward sub-block (Pre-LN residual) ----

        # Save for the short skip around the FFN.
        residue_short = x

        # Pre-norm before the feed-forward network.
        x = self.layernorm_3(x)

        # GeGLU projection: the linear layer outputs 2 * 4*channels,
        # which is split into a value branch and a gate branch.
        # Gating allows the network to selectively suppress activations.
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)

        # GeGLU activation: element-wise product of the value and
        # GELU-activated gate.
        x = x * F.gelu(gate)

        # Project back from 4*channels to channels.
        x = self.linear_geglu_2(x)

        # Residual connection around the FFN.
        x += residue_short

        # Reshape back from sequence form to 2-D spatial feature map.
        # (Batch, H*W, C) -> (Batch, C, H*W) -> (Batch, C, H, W)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        # 1x1 conv + long residual connection that spans the entire
        # attention block.  This allows the block to be bypassed if
        # necessary, stabilising training of very deep U-Nets.
        return self.conv_output(x) + residue_long
