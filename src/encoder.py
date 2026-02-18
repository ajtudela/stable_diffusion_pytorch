import torch
from torch import nn
from torch.nn import functional as F
from .vae_utils import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    """
    Variational Autoencoder (VAE) Encoder for Stable Diffusion.

    Compresses an input image from pixel space into a compact latent
    distribution (mean and log-variance) in the latent space. The spatial
    resolution is reduced by 8x while the channel depth is progressively
    increased to capture richer semantic features at each scale.
    """

    def __init__(self) -> None:
        # The encoder progressively downsamples the spatial dimensions (H, W)
        # by 8x while increasing the number of feature channels.
        # This forces the network to represent the image in a dense,
        # high-level latent space.
        super().__init__(
            # Project the 3 input RGB channels up to 128 feature maps.
            # kernel_size=3, padding=1 keeps spatial resolution unchanged
            # (same convolution).
            # (Batch, 3, H, W) -> (Batch, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # Residual block at 128 channels: refines features without
            # changing resolution. Skip connections allow gradients to flow
            # more easily during training.
            # (Batch, 128, H, W) -> (Batch, 128, H, W)
            VAE_ResidualBlock(128, 128),

            # Second residual block at 128 channels:
            # adds more representational depth
            # before the first downsampling step.
            # (Batch, 128, H, W) -> (Batch, 128, H, W)
            VAE_ResidualBlock(128, 128),

            # Strided convolution (stride=2) to halve the spatial dimensions.
            # padding=0 because we manually apply asymmetric padding
            # in forward() to handle even-dimension images correctly.
            # (Batch, 128, H, W) -> (Batch, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # Expand channels from 128 to 256 while keeping H/2 x W/2.
            # More channels = richer feature representation at this scale.
            # (Batch, 128, H/2, W/2) -> (Batch, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),

            # Residual block at 256 channels:
            # further refines mid-level features.
            # (Batch, 256, H/2, W/2) -> (Batch, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256),

            # Second downsampling: halve spatial dimensions again via
            # strided convolution.
            # (Batch, 256, H/2, W/2) -> (Batch, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # Expand channels from 256 to 512 at H/4 x W/4.
            # At this scale, features represent higher-level structures
            # (textures, objects).
            # (Batch, 256, H/4, W/4) -> (Batch, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),

            # Residual block at 512 channels: deepens the representation
            # at this scale.
            # (Batch, 512, H/4, W/4) -> (Batch, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512),

            # Third and final downsampling: reduces spatial dims to H/8 x W/8.
            # This is the bottleneck spatial resolution of the latent space.
            # (Batch, 512, H/4, W/4) -> (Batch, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # Residual block at 512 channels at the bottleneck resolution.
            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            # Second residual block at bottleneck: builds deeper
            # abstract representations.
            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            # Third residual block at bottleneck:
            # ensures sufficient model capacity before
            # applying global attention.
            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            # Self-attention block: allows every spatial position to
            # attend to every other. This captures long-range dependencies
            # that local convolutions cannot model, which is crucial for
            # global coherence in generated images.
            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            VAE_AttentionBlock(512),

            # Residual block after attention: integrates the attention output
            # back into the feature stream.
            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            # Group Normalization (32 groups over 512 channels):
            # normalizes activations in a way that is independent of batch
            # size, improving training stability. Applied before the
            # activation to standardize the distribution of features.
            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            nn.GroupNorm(32, 512),

            # SiLU (Sigmoid Linear Unit / Swish) activation:
            # f(x) = x * sigmoid(x).
            # Smoother than ReLU, empirically improves performance
            # in generative models.
            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            nn.SiLU(),

            # Project 512 feature channels down to 8 channels.
            # The 8 channels will encode both mean (4 ch) and
            # log-variance (4 ch) of the latent Gaussian distribution.
            # (Batch, 512, H/8, W/8) -> (Batch, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # 1x1 convolution: a channel-wise linear projection with
            # no spatial mixing.
            # Acts as a final linear transformation to adjust channel-wise
            # statistics before splitting into mean and log-variance.
            # NOTE: 'adding=0' is a typo — should be 'padding=0',
            # but 1x1 convolutions do not require padding regardless,
            # so the output shape is unaffected.
            # (Batch, 8, H/8, W/8) -> (Batch, 8, H/8, W/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode an input image into a latent sample using the
        reparameterization trick.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (Batch, 3, Height, Width).
        noise : torch.Tensor
            Standard normal noise of shape (Batch, 4, Height/8, Width/8),
            used for the reparameterization trick to sample from
            the latent distribution.

        Returns
        -------
        torch.Tensor
            Sampled latent tensor of shape (Batch, 4, Height/8, Width/8),
            scaled to match the expected input range of the diffusion U-Net.
        """
        # Iterate over all layers registered in the
        # nn.Sequential parent class.
        for module in self:
            # Detect strided convolutions (downsampling layers) by
            # checking stride == (2,2). These layers require padding=0 so
            # we manually add one pixel of asymmetric
            # padding on the right and bottom sides. This is needed to
            # preserve exact half-resolution dimensions for images with even
            # spatial sizes, avoiding off-by-one errors that symmetric padding
            # would introduce.
            # Padding order: (left, right, top, bottom)
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            # Pass the tensor through the current layer
            # (conv, residual, attention, norm, etc.).
            x = module(x)

        # Split the 8-channel output into two 4-channel tensors
        # along the channel dimension. The network predicts the
        # parameters of a Gaussian: mean (μ) and log-variance (log σ²).
        # Using log-variance instead of variance keeps the network
        # output unconstrained (can be negative),
        # which is more numerically stable during training.
        # (Batch, 8, H/8, W/8) -> 2x (Batch, 4, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Clamp log_variance to the range [-30, 20] for numerical stability.
        # Extremely large values would cause exp() to overflow to inf,
        # and extremely small values would collapse the variance to zero.
        # (Batch, 4, H/8, W/8) -> (Batch, 4, H/8, W/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # Exponentiate log-variance to recover the actual variance σ².
        # We stored log σ² so that the model can predict any positive value
        # freely; exp() maps it back to the required non-negative domain.
        # (Batch, 4, H/8, W/8) -> (Batch, 4, H/8, W/8)
        variance = log_variance.exp()

        # Compute the standard deviation σ = sqrt(σ²).
        # We need σ (not σ²) for the reparameterization trick below.
        # (Batch, 4, H/8, W/8) -> (Batch, 4, H/8, W/8)
        stdev = variance.sqrt()

        # Reparameterization trick: sample z ~ N(μ, σ²) as
        # z = μ + σ * ε, where ε ~ N(0, 1) is the externally provided
        # noise tensor. This formulation keeps the sampling step
        # differentiable with respect to μ and σ, which is essential for
        # backpropagating gradients through the stochastic latent.
        # Z = N(0, 1) -> Z' = N(mean, variance) via Z' = mean + stdev * Z
        # (Batch, 4, H/8, W/8)
        x = mean + stdev * noise

        # Scale the latent vector by the empirical constant 0.18215.
        # This value was determined from the distribution of latents
        # in the original Stable Diffusion training data and ensures
        # the latents have unit variance, matching the expected input
        # scale of the diffusion U-Net.
        x *= 0.18215

        return x
