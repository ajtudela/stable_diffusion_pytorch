import torch
from torch import nn
from .vae_utils import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Decoder(nn.Sequential):
    """
    VAE Decoder: upsamples a latent code from the compressed latent
    space back to full image resolution.

    Mirrors the encoder architecture by progressively increasing
    spatial resolution (via upsampling) while decreasing the number
    of feature channels until the original (3, H, W) image shape
    is recovered.
    """

    def __init__(self) -> None:
        super().__init__(
            # 1x1 convolution that acts as a channel-wise projection on
            # the 4-channel latent input. No spatial change; prepares the
            # latent for the next layer.
            # (Batch, 4, H/8, W/8) -> (Batch, 4, H/8, W/8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # Expand the 4 latent channels to 512 feature maps using 3x3 conv.
            # padding=1 keeps spatial dimensions unchanged while increasing
            # depth.
            # (Batch, 4, H/8, W/8) -> (Batch, 512, H/8, W/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            # Residual block to refine features without changing channels or
            # resolution. Skip connections prevent vanishing gradients and
            # preserve low-level info.
            VAE_ResidualBlock(512, 512),

            # Self-attention block over spatial positions of the feature map.
            # Captures long-range dependencies (e.g. global coherence) at this
            # compressed 1/8 resolution, where the cost is still manageable.
            VAE_AttentionBlock(512),

            # Three consecutive residual blocks to deepen the non-linear
            # transformation at the bottleneck resolution before upsampling.
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # Additional residual block; mirrors the encoder bottleneck depth
            # so the decoder has sufficient capacity to invert the encoding.
            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            # --- First upsampling stage: H/8 -> H/4 ---

            # Nearest-neighbour upsampling doubles spatial resolution cheaply.
            # No learnable parameters; avoids checkerboard artefacts common in
            # transposed convolutions.
            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/4, W/4)
            nn.Upsample(scale_factor=2),

            # 3x3 conv after upsampling to smooth nearest-neighbour
            # artifacts and let the network learn how to blend the
            # duplicated pixels. padding=1 keeps spatial resolution
            # unchanged.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # Three residual blocks to process features at the new H/4
            # resolution, giving the network capacity to reconstruct
            # mid-level structure.
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # --- Second upsampling stage: H/4 -> H/2 ---

            # Double spatial resolution again, moving closer to the
            # original image size.
            # (Batch, 512, H/4, W/4) -> (Batch, 512, H/2, W/2)
            nn.Upsample(scale_factor=2),

            # Smoothing conv after upsample; same rationale as the first stage.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # First residual block also halves the channel count 512 -> 256.
            # Reducing channels as resolution grows keeps computation balanced.
            # (Batch, 512, H/2, W/2) -> (Batch, 256, H/2, W/2)
            VAE_ResidualBlock(512, 256),

            # Two more residual blocks to refine features at H/2 resolution
            # withthe reduced 256-channel representation.
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # --- Third upsampling stage: H/2 -> H ---

            # Final upsample reaches full image resolution.
            # (Batch, 256, H/2, W/2) -> (Batch, 256, H, W)
            nn.Upsample(scale_factor=2),

            # Smoothing conv at full resolution to blend upsampled pixels.
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            # First residual block reduces channels 256 -> 128 at
            # full resolution. Lower channel count at high resolution
            # saves memory and computation.
            # (Batch, 256, H, W) -> (Batch, 128, H, W)
            VAE_ResidualBlock(256, 128),

            # Two more residual blocks refine fine spatial details
            # (textures, edges) at the final full resolution with 128 channels.
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # GroupNorm normalises across 32 groups of channels,
            # stabilising training and improving quality. Placed before
            # the final activation, as recommended.
            nn.GroupNorm(32, 128),

            # SiLU (Sigmoid Linear Unit / Swish) activation: smooth,
            # non-monotonic non-linearity that empirically outperforms
            # ReLU in diffusion models.
            nn.SiLU(),

            # Final 3x3 conv projects 128 feature channels down to
            # 3 RGB channels, producing the reconstructed image.
            # padding=1 preserves spatial size.
            # (Batch, 128, H, W) -> (Batch, 3, H, W)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of latent codes into RGB images.

        Parameters
        ----------
        x : torch.Tensor
            Latent tensor of shape (Batch, 4, H/8, W/8) sampled from the
            VAE's latent distribution.

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor of shape (Batch, 3, H, W).
        """
        # Reverse the encoder's scaling: the VAE encoder multiplies latents by
        # 0.18215 to keep unit variance in the latent space. Dividing here
        # restores the original magnitude before passing through the
        # decoder layers.
        # (Batch, 4, H/8, W/8)
        x /= 0.18215

        # Sequentially apply every sub-module registered in __init__.
        # nn.Sequential's __iter__ yields them in insertion order.
        for module in self:
            x = module(x)

        # x is now a full-resolution RGB image.
        # (Batch, 3, H, W)
        return x
