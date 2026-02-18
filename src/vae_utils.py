import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    """
    Attention block for the VAE that applies self-attention over the
    spatial dimensions of a feature map.

    Convolutional layers only capture local context. This block
    flattens the H×W spatial grid into a sequence and runs
    SelfAttention so every pixel can attend to every other pixel,
    capturing global structure at the bottleneck.
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize the attention block.

        Parameters
        ----------
        channels : int
            Number of feature channels in the input feature map.
        """
        super().__init__()

        # Group Normalization before attention stabilizes the feature
        # distribution (32 groups over `channels` channels).
        # GroupNorm is preferred over BatchNorm because it is
        # independent of batch size.
        self.groupnorm = nn.GroupNorm(32, channels)

        # Single-head self-attention over the `channels`-dim embedding.
        # One head is sufficient here because the spatial feature maps
        # are already rich; multi-head is more important in text models.
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply group norm and self-attention with a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (Batch, Channels, H, W).

        Returns
        -------
        torch.Tensor
            Output of the same shape (Batch, Channels, H, W).
        """
        # Store the input so it can be added back at the end.
        # The residual connection lets gradients bypass the attention
        # operation, making training more stable.
        # (Batch, C, H, W)
        residue = x

        # Normalize activations before the attention operation.
        # Pre-norm (norm before the sub-layer) is used here rather
        # than post-norm because it gives more stable gradient flow.
        x = self.groupnorm(x)

        # Unpack spatial dimensions so we can reshape the tensor.
        # n=batch, c=channels, h=height, w=width.
        n, c, h, w = x.shape

        # Flatten Height and Width into a single sequence dimension.
        # Self-attention expects (Batch, SeqLen, Embedding).
        # Each pixel position becomes one element of the sequence
        # and its channel vector is the embedding.
        # (Batch, C, H, W) -> (Batch, C, H*W)
        x = x.view((n, c, h * w))

        # Transpose to put the sequence axis before the channel axis,
        # matching the convention (Batch, SeqLen, Embedding) that
        # SelfAttention expects.
        # (Batch, C, H*W) -> (Batch, H*W, C)
        x = x.transpose(-1, -2)

        # Run scaled dot-product self-attention.
        # Every pixel position can now attend to every other position,
        # enabling the model to relate distant spatial features.
        # (Batch, H*W, C) -> (Batch, H*W, C)
        x = self.attention(x)

        # Transpose back so channels are the second dimension again,
        # matching the standard (Batch, C, SeqLen) layout before
        # reshaping to the 4-D spatial tensor.
        # (Batch, H*W, C) -> (Batch, C, H*W)
        x = x.transpose(-1, -2)

        # Restore the 2-D spatial layout by splitting the sequence
        # dimension back into the original H and W.
        # (Batch, C, H*W) -> (Batch, C, H, W)
        x = x.view((n, c, h, w))

        # Add the residual (skip connection): combines the attended
        # representation with the original input so the block only
        # needs to learn a residual correction, not the full mapping.
        x += residue

        return x


class VAE_ResidualBlock(nn.Module):
    """
    Residual block used throughout the VAE encoder and decoder.

    Applies two normalisation → activation → convolution stages with
    a skip connection. If the input and output channel counts differ,
    a 1×1 convolution projects the skip connection to the correct
    channel dimension so the addition remains valid.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the residual block.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input feature map.
        out_channels : int
            Number of channels in the output feature map.
        """
        super().__init__()

        # First GroupNorm: normalizes the input activations.
        # 32 groups is a standard choice that balances between
        # LayerNorm (1 group) and InstanceNorm (C groups).
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)

        # First convolution: mixes spatial context while optionally
        # changing the number of channels from in_channels to
        # out_channels. padding=1 keeps spatial dimensions unchanged.
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        # Second GroupNorm: normalizes activations after the first
        # convolution to keep the distribution stable before the
        # second non-linearity.
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)

        # Second convolution: refines the features at out_channels
        # without changing the channel count or spatial size.
        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        # Residual (skip) connection projection:
        # If in_channels == out_channels, the skip connection passes
        # the input through unchanged (Identity).
        # Otherwise, a 1×1 convolution re-projects the input to
        # out_channels so both branches have the same shape for addition.
        # NOTE: the original code used '==' (comparison) instead of '='
        # (assignment) — a bug that prevented the layer from being
        # registered. Fixed to use '=' here.
        if in_channels == out_channels:
            self.residual_layer: nn.Module = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the residual block to the input feature map.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (Batch, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output feature map of shape (Batch, out_channels, H, W).
        """
        # Save the original input for the skip connection.
        # (Batch, in_channels, H, W)
        residue = x

        # Normalize before the first activation: stabilizes the
        # distribution of inputs to the subsequent non-linearity.
        x = self.groupnorm_1(x)

        # SiLU activation (Swish): f(x) = x * sigmoid(x).
        # Smoother than ReLU; its non-zero gradient for negative
        # inputs improves training in deep generative models.
        x = F.silu(x)

        # First 3×3 convolution: extracts local spatial features and
        # maps in_channels -> out_channels.
        # (Batch, in_channels, H, W) -> (Batch, out_channels, H, W)
        x = self.conv_1(x)

        # Normalize after the channel expansion to keep magnitudes
        # well-behaved before the second non-linearity.
        x = self.groupnorm_2(x)

        # Second SiLU activation: introduces additional non-linearity
        # so the block can represent more complex functions.
        x = F.silu(x)

        # Second 3×3 convolution: further refines features while
        # keeping the channel count and spatial size unchanged.
        # (Batch, out_channels, H, W) -> (Batch, out_channels, H, W)
        x = self.conv_2(x)

        # Add the skip connection so the block learns a residual
        # correction on top of the input rather than a full mapping.
        # residual_layer projects the input if channel counts differ.
        return x + self.residual_layer(residue)
