import torch
from torch import nn
from torch.nn import functional as F
from .unet_utils import UNET_AttentionBlock, UNET_ResidualBlock


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time-step embedding projected through a two-layer MLP.

    Transforms a scalar diffusion time-step into a high-dimensional
    vector that conditions the U-Net, telling it how noisy the input is.
    """

    def __init__(self, n_embd: int) -> None:
        """
        Initialize the time-step embedding MLP.

        Parameters
        ----------
        n_embd : int
            Dimensionality of the input sinusoidal embedding
            (320 in the default Stable Diffusion configuration).
        """
        super().__init__()

        # First linear layer expands the embedding dimension by 4x.
        # A wider hidden representation gives the network more capacity
        # to encode the non-linear relationship between time-step and
        # noise level.
        # (Batch, 320) -> (Batch, 1280)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)

        # Second linear layer keeps the expanded 4x dimension which
        # becomes the time-conditioning vector used throughout the U-Net.
        # (Batch, 1280) -> (Batch, 1280)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a sinusoidal time embedding through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Pre-computed sinusoidal embedding of shape (Batch, 320).

        Returns
        -------
        torch.Tensor
            Time-conditioning vector of shape (Batch, 1280).
        """
        # First linear projection: expand dimensionality.
        # (Batch, 320) -> (Batch, 1280)
        x = self.linear_1(x)

        # SiLU activation (Swish): introduces non-linearity so the
        # network can learn complex mappings from time-step to
        # conditioning signal.
        x = F.silu(x)

        # Second linear projection: refines the time embedding while
        # keeping the 1280-dim output that the rest of the U-Net expects.
        # (Batch, 1280) -> (Batch, 1280)
        x = self.linear_2(x)

        return x


class Upsample(nn.Module):
    """
    Spatial 2x upsampling followed by a 3x3 convolution.

    Nearest-neighbour upsampling avoids the checkerboard artefacts of
    transposed convolutions; the subsequent conv smooths the result
    and makes it learnable.
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize the upsampling module.

        Parameters
        ----------
        channels : int
            Number of input and output feature channels
            (preserved through the upsampling + convolution).
        """
        super().__init__()

        # 3x3 conv after upsampling: blends the duplicated pixels and
        # adds learnable parameters so the network controls how
        # neighbouring pixels are combined.
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample spatial dimensions by 2x and convolve.

        Parameters
        ----------
        x : torch.Tensor
            Feature map of shape (Batch, Channels, H, W).

        Returns
        -------
        torch.Tensor
            Upsampled features of shape (Batch, Channels, 2*H, 2*W).
        """
        # Nearest-neighbour interpolation doubles H and W without
        # introducing learnable parameters or blurring bias.
        # (Batch, C, H, W) -> (Batch, C, 2*H, 2*W)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # 3x3 conv smooths the blocky upsampled output.
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    """
    Sequential container that routes inputs depending on layer type.

    - UNET_AttentionBlock receives (x, context).
    - UNET_ResidualBlock  receives (x, time).
    - All other layers    receive  (x) only.

    This avoids having to write separate forward logic for every
    combination of layers within each encoder / decoder stage.
    """

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Sequentially apply each child layer with the appropriate inputs.

        Parameters
        ----------
        x : torch.Tensor
            Spatial features.
        context : torch.Tensor
            Text conditioning from CLIP.
        time : torch.Tensor
            Time-step conditioning embedding.

        Returns
        -------
        torch.Tensor
            Output features after all layers.
        """
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                # Attention blocks need both features and text context.
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                # Residual blocks need features and time conditioning.
                x = layer(x, time)
            else:
                # Plain layers (Conv2d, Upsample, etc.) only need x.
                x = layer(x)
        return x


class UNET(nn.Module):
    """
    U-Net noise-prediction network for the latent diffusion model.

    Architecture follows an encoder-bottleneck-decoder structure with
    skip connections between encoder and decoder stages at matching
    resolutions (like the original U-Net).  Feature channels increase
    as spatial resolution decreases in the encoder, and the process is
    reversed in the decoder.
    """

    def __init__(self) -> None:
        super().__init__()

        # ============================================================
        # ENCODER -- progressively downsamples spatial resolution while
        # increasing channel count. Each stage produces a feature map
        # that is saved for the corresponding decoder skip connection.
        # ============================================================
        self.encoders = nn.ModuleList([
            # --- Stage 0: Initial projection (H/8 x W/8, 320 ch) ---
            # Project the 4-channel latent input to 320 feature channels.
            SwitchSequential(
                nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            # Two residual + attention blocks refine features at 320 ch.
            # Attention at 8 heads * 40 = 320 channels captures spatial
            # dependencies at this scale.
            SwitchSequential(
                UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(
                UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # --- Downsample: H/8 -> H/16 ---
            # Strided 3x3 conv halves spatial dims while keeping 320 ch.
            SwitchSequential(
                nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            # --- Stage 1: (H/16 x W/16, 640 ch) ---
            # Expand channels 320 -> 640 and refine with attention.
            # 8 heads * 80 = 640 channels.
            SwitchSequential(
                UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(
                UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # --- Downsample: H/16 -> H/32 ---
            SwitchSequential(
                nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            # --- Stage 2: (H/32 x W/32, 1280 ch) ---
            # Expand channels 640 -> 1280.  8 heads * 160 = 1280 ch.
            SwitchSequential(
                UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(
                UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # --- Downsample: H/32 -> H/64 ---
            SwitchSequential(
                nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            # --- Stage 3: (H/64 x W/64, 1280 ch) ---
            # Two residual blocks without attention (spatial resolution
            # is very small so global attention is less beneficial).
            SwitchSequential(
                UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(
                UNET_ResidualBlock(1280, 1280)),
        ])

        # ============================================================
        # BOTTLENECK -- deepest point of the U-Net.
        # Residual + attention + residual refines the most compressed
        # representation before the decoder begins upsampling.
        # ============================================================
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        # ============================================================
        # DECODER -- progressively upsamples back to the original
        # latent resolution.  At each stage, encoder skip features are
        # concatenated along the channel axis (doubling the channel
        # count at the input of each residual block).
        # ============================================================
        self.decoders = nn.ModuleList([
            # --- Stage 3 decoder: (H/64 x W/64) ---
            # Input channels = 1280 (bottleneck) + 1280 (skip) = 2560.
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280)),

            # Last block at H/64 includes an Upsample to go H/64 -> H/32.
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            # --- Stage 2 decoder: (H/32 x W/32, 1280 ch) ---
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            # Last block at H/32 upsamples to H/16.
            SwitchSequential(
                UNET_ResidualBlock(1920, 1280),
                UNET_AttentionBlock(8, 160),
                Upsample(1280)),

            # --- Stage 1 decoder: (H/16 x W/16, 640 ch) ---
            SwitchSequential(
                UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(
                UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            # Last block at H/16 upsamples to H/8.
            SwitchSequential(
                UNET_ResidualBlock(960, 640),
                UNET_AttentionBlock(8, 80),
                Upsample(640)),

            # --- Stage 0 decoder: (H/8 x W/8, 320 ch) ---
            SwitchSequential(
                UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(
                UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(
                UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the full U-Net: encoder -> bottleneck -> decoder with skip
        connections.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent of shape (Batch, 4, H/8, W/8).
        context : torch.Tensor
            CLIP text embeddings of shape (Batch, 77, 768).
        time : torch.Tensor
            Time-conditioning vector of shape (Batch, 1280).

        Returns
        -------
        torch.Tensor
            Predicted noise of shape (Batch, 320, H/8, W/8).
        """
        # Collect encoder outputs for skip connections.
        skip_connections: list[torch.Tensor] = []

        # --- Encoder pass ---
        for layers in self.encoders:
            x = layers(x, context, time)
            # Store each encoder output; the decoder will concatenate
            # them in reverse order.
            skip_connections.append(x)

        # --- Bottleneck ---
        x = self.bottleneck(x, context, time)

        # --- Decoder pass ---
        for layers in self.decoders:
            # Pop the last saved encoder feature map and concatenate it
            # with the current decoder features along the channel axis.
            # This provides the decoder with high-resolution details
            # that would otherwise be lost after downsampling.
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class UNET_OutputLayer(nn.Module):
    """
    Final output projection of the U-Net.

    Normalises, activates, and projects the 320-channel decoder output
    back to 4 latent channels (matching the VAE latent space).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the U-Net output projection layer.

        Parameters
        ----------
        in_channels : int
            Number of channels from the U-Net decoder (typically 320).
        out_channels : int
            Number of output latent channels (typically 4, matching
            the VAE latent space).
        """
        super().__init__()

        # GroupNorm with 32 groups over the decoder output channels.
        self.groupnorm = nn.GroupNorm(32, in_channels)

        # 3x3 conv projects from in_channels (320) to out_channels (4)
        # to produce the final noise prediction in the latent space.
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project U-Net features to latent-space noise prediction.

        Parameters
        ----------
        x : torch.Tensor
            U-Net decoder output of shape (Batch, 320, H/8, W/8).

        Returns
        -------
        torch.Tensor
            Predicted noise of shape (Batch, 4, H/8, W/8).
        """
        # Normalise across channel groups.
        # (Batch, 320, H/8, W/8) -> (Batch, 320, H/8, W/8)
        x = self.groupnorm(x)

        # SiLU non-linearity before the final conv.
        x = F.silu(x)

        # Project to 4 latent channels.
        # (Batch, 320, H/8, W/8) -> (Batch, 4, H/8, W/8)
        x = self.conv(x)

        return x


class Diffusion(nn.Module):
    """
    Top-level latent diffusion model.

    Wraps the time embedding MLP, the U-Net, and the output projection
    into a single module that takes a noisy latent, a text context, and
    a time-step, and predicts the noise to be removed.
    """

    def __init__(self) -> None:
        super().__init__()

        # MLP that transforms the 320-dim sinusoidal time-step encoding
        # into a 1280-dim conditioning vector for the U-Net.
        self.time_embedding = TimeEmbedding(320)

        # U-Net noise-prediction backbone.
        self.unet = UNET()

        # Final projection: converts 320 decoder channels -> 4 latent
        # channels to produce the noise prediction.
        self.final = UNET_OutputLayer(320, 4)

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the noise present in a noisy latent.

        Parameters
        ----------
        latent : torch.Tensor
            Noisy latent of shape (Batch, 4, H/8, W/8).
        context : torch.Tensor
            CLIP text embeddings of shape (Batch, 77, 768).
        time : torch.Tensor
            Sinusoidal time-step embedding of shape (Batch, 320).

        Returns
        -------
        torch.Tensor
            Predicted noise of shape (Batch, 4, H/8, W/8).
        """
        # Expand the sinusoidal time-step into a richer conditioning
        # vector used by every residual block in the U-Net.
        # (Batch, 320) -> (Batch, 1280)
        time = self.time_embedding(time)

        # Run the full U-Net: encoder -> bottleneck -> decoder with
        # skip connections, conditioned on text and time.
        # (Batch, 4, H/8, W/8) -> (Batch, 320, H/8, W/8)
        output = self.unet(latent, context, time)

        # Project back to 4 latent channels.
        # (Batch, 320, H/8, W/8) -> (Batch, 4, H/8, W/8)
        output = self.final(output)

        return output
