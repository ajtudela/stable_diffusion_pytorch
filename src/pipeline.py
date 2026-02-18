"""
Stable Diffusion inference pipeline.

This module provides the complete pipeline for generating images from text
prompts using a pre-trained latent diffusion model, CLIP text encoder, VAE,
and diffusion sampler.
"""

import torch
from typing import Optional, Dict, Any
import numpy as np
from tqdm import tqdm
from .ddpm import DDPMSampler

# Image dimensions for the input and output
WIDTH: int = 512
HEIGHT: int = 512

# Latent space dimensions (reduced by 8x compared to image space)
LATENTS_WIDTH: int = WIDTH // 8
LATENTS_HEIGHT: int = HEIGHT // 8


def generate(
    prompt: str,
    uncond_prompt: str,
    input_image: Optional[Any] = None,
    strength: float = 0.8,
    do_cfg: bool = True,
    cfg_scale: float = 7.5,
    sampler_name: str = "ddpm",
    n_inference_steps: int = 50,
    models: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    idle_device: Optional[str] = None,
    tokenizer: Any = None
) -> np.ndarray:
    """
    Generate an image from a text prompt using the latent diffusion model.

    Parameters
    ----------
    prompt : str
        The text prompt describing the desired image.
    uncond_prompt : str
        Unconditional (empty/negative) prompt used for
        classifier-free guidance.
    input_image : Optional[Any]
        PIL Image for image-to-image generation. If None, generates from noise.
    strength : float
        Controls how much noise is added to the input image
        (0 < strength <= 1).
        Higher values mean more denoising steps are taken. Only used when
        input_image is provided.
    do_cfg : bool
        Whether to apply classifier-free guidance (CFG). If True, both
        conditional and unconditional predictions are generated and blended.
    cfg_scale : float
        Classifier-free guidance scale. Controls the influence of the prompt
        on the generation. Higher values make the output follow the prompt
        more closely.
    sampler_name : str
        Name of the noise sampler to use (currently only 'ddpm' is supported).
    n_inference_steps : int
        Number of denoising steps. More steps generally produce higher quality
        images but take longer.
    models : Optional[Dict[str, Any]]
        Dictionary containing the pre-trained models:
        - 'clip': CLIP text encoder
        - 'encoder': VAE encoder (for input_image encoding)
        - 'diffusion': U-Net diffusion model
        - 'decoder': VAE decoder
    seed : Optional[int]
        Random seed for reproducibility. If None, uses a random seed.
    device : str
        PyTorch device for computation ('cuda', 'cpu', etc.).
    idle_device : Optional[str]
        Device to move models to when not in use (for memory efficiency).
    tokenizer : Optional[Any]
        Tokenizer for converting text prompts to token sequences.

    Returns
    -------
    np.ndarray
        Generated image as a numpy array of shape (H, W, 3) with values
        in the range [0, 255] (uint8).

    Raises
    ------
    ValueError
        If strength is not in the range (0, 1].
    ValueError
        If an unknown sampler name is provided.
    """
    # Suppress gradient computation for inference to save memory.
    with torch.no_grad():
        # Validate the strength parameter for image-to-image generation.
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")

        # Define a helper function to move models to idle_device
        # when not in use.
        # This frees GPU memory by moving idle models back to CPU.
        if idle_device:
            def to_idle(x): return x.to(idle_device)
        else:
            # If no idle device is specified, do nothing
            # (keep on compute device).
            def to_idle(x): return x

        # Create a random number generator for reproducibility.
        # Generator must be on the same device as the tensors it produces.
        generator = torch.Generator(device=device)
        if seed is None:
            # If no seed is provided, use a random one.
            generator.seed()
        else:
            # Otherwise, seed the generator for reproducible results.
            generator.manual_seed(seed)

        # --- Text Encoding: Convert prompts to CLIP embeddings ---

        clip = models['clip']  # type: ignore[index]
        # Move CLIP to compute device.
        clip.to(device)

        if do_cfg:
            # Classifier-Free Guidance (CFG) requires both conditional and
            # unconditional embeddings. The unconditional embedding guides
            # the model towards regions of the latent space that don't
            # correspond to any particular text.

            # Tokenize the conditional prompt (what we want to generate).
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding='max_length', max_length=77
            ).input_ids
            # (Batch, Seq_len) -> convert to torch tensor on device.
            cond_tokens = torch.tensor(
                cond_tokens, dtype=torch.long, device=device)
            # Pass tokens through CLIP to get contextual embeddings.
            # (Batch, Seq_len) -> (Batch, Seq_len, Dim=768)
            cond_context = clip(cond_tokens)

            # Tokenize the unconditional prompt
            # (empty string, tells model 'no prompt').
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding='max_length', max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(
                uncond_tokens, dtype=torch.long, device=device)
            # (Batch, Seq_len) -> (Batch, Seq_len, Dim=768)
            uncond_context = clip(uncond_tokens)

            # Stack conditional and unconditional contexts.
            # The diffusion model will process both in parallel for efficiency.
            # (2, Seq_len, Dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Without classifier-free guidance, we only encode the prompt.
            tokens = tokenizer.batch_encode_plus(  # type: ignore[union-attr]
                [prompt], padding='max_length',  max_length=77
            ).input_ids
            tokens = torch.tensor(
                tokens, dtype=torch.long, device=device)
            # (1, 77, 768)
            context = clip(tokens)

        # Move CLIP to idle device to free GPU memory during diffusion.
        to_idle(clip)

        # --- Sampler Setup ---

        if sampler_name == 'ddpm':
            # DDPM (Denoising Diffusion Probabilistic Models) is the sampler
            # that schedules the noise levels and transition probabilities
            # during the reverse diffusion process.
            sampler = DDPMSampler(generator)
            # Configure the number of denoising steps for the sampler.
            # More steps give higher quality but slower generation.
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f'Unknown sampler: {sampler_name}')

        # Shape of latent tensors: (Batch=1, Channels=4, Height, Width)
        # The diffusion model works in the VAE latent space (8x compressed).
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # --- Latent Initialization ---

        if input_image:
            # Image-to-Image generation: encode the input image
            # into latent space, add noise according to the
            # strength parameter, then denoise.

            encoder = models['encoder']  # type: ignore[index]
            # Move encoder to compute device.
            encoder.to(device)

            # Resize the input image to the expected size and
            # convert to tensor.
            # (H, W, C) = (512, 512, 3)
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # Convert numpy array to torch tensor with float32 precision.
            # (H, W, C) -> (Batch=1, H, W, C)
            input_image_tensor = torch.tensor(
                input_image_tensor, dtype=torch.float32).unsqueeze(0)

            # Rescale pixel values from [0, 255] to [-1, 1] range.
            # This matches the range the VAE encoder expects.
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Batch, H, W, C) -> (Batch, C, H, W)
            # Reorder dimensions to match PyTorch's channel-first convention.
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Generate random noise with the same shape as the latents.
            # This noise is used in the VAE encoder's reparameterization trick.
            encoder_noise = torch.randn(
                latents_shape, generator=generator, device=device)

            # Encode the image into latent space.
            # (Batch, C, H, W) -> (Batch, 4, H/8, W/8)
            latents = encoder(input_image_tensor, encoder_noise)

            # For image-to-image, we don't start from pure noise.
            # Instead, we start from the encoded image latent and add noise
            # according to the strength parameter. Higher strength = more noise
            # is added = more denoising steps are taken
            # (more generation freedom).
            sampler.set_strength(strength=strength)
            # Add noise to the encoded latents based on the schedule.
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            # Move encoder to idle device to free GPU memory.
            to_idle(encoder)
        else:
            # Text-to-Image generation: start from pure random noise
            # and denoise the full diffusion process.
            # Sample random latents from a standard normal distribution.
            # (Batch, 4, Latents_H, Latents_W)
            latents = torch.randn(
                latents_shape, generator=generator, device=device)

        diffusion = models['diffusion']  # type: ignore[index]
        # Move diffusion model to compute device.
        diffusion.to(device)

        # Denoising loop: iteratively denoise the latents through
        # diffusion timesteps.
        # This is the core diffusion process where each step removes
        # a small amount of noise.
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # Generate sinusoidal time embedding for the current timestep.
            # The embedding is used to condition the diffusion model
            # on which step we're at.
            # Output shape: (1, 320) - a sinusoidal positional encoding.
            time_embedding = get_time_embedding(timestep).to(device)

            # Prepare model input: the latent representation to be denoised.
            # Shape: (Batch, 4, Latents_H, Latents_W)
            model_input = latents

            if do_cfg:
                # Classifier-Free Guidance (CFG) preparation:
                # Duplicate the latents to process both
                # conditional and unconditional paths.
                # Shape transformation: (Batch, 4, Latents_H, Latents_W) ->
                # (2*Batch, 4, Latents_H, Latents_W)
                # This allows the model to generate noise predictions for both
                # the prompt and the empty prompt, which we'll blend together.
                model_input = model_input.repeat(2, 1, 1, 1)

            # Forward pass through the diffusion U-NET model.
            # Input: latents with time and text embeddings as conditioning.
            # Output: predicted noise that should be removed from the latents.
            # The model predicts what noise was added at this timestep.
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # Split predictions into conditional (with prompt)
                # and unconditional (no prompt).
                output_cond, output_uncond = model_output.chunk(2)
                # Apply classifier-free guidance formula:
                # noise_final = cfg_scale * (noise_cond - noise_uncond)
                # + noise_uncond
                # This amplifies the influence of the prompt by scaling
                # the difference and adding it back to the
                # unconditional prediction.
                # Higher cfg_scale = stronger adherence to the prompt.
                model_output = cfg_scale * \
                    (output_cond - output_uncond) + output_uncond

            # Sampler step: remove the predicted noise from the latents.
            # The sampler applies the reverse diffusion step using
            # the predicted noise.
            # This moves the latents closer to the target clean
            # image distribution.
            # Input t is the current timestep integer,
            # used to index the noise schedule.
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models['decoder']  # type: ignore[index]
        # Move decoder to compute device.
        decoder.to(device)

        # Run the latents through the decoder of the VAE
        # VAE Decoder: convert latent representations back to pixel space.
        # Input shape: (Batch, 4, Latents_H, Latents_W)
        # Output shape: (Batch, 3, Height, Width) - 3 RGB channels
        images = decoder(latents)
        to_idle(decoder)

        # Post-processing: rescale the images from model output range
        # to standard image range.
        # Model outputs are in [-1, 1] range (normalized during VAE training).
        # Rescale to [0, 255] for standard 8-bit color representation.
        # The clamp=True ensures values stay within [0, 255] bounds
        # (no overflow).
        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        # Dimension permutation: convert from PyTorch channel-first to
        # standard image format.
        # PyTorch convention: (Batch, C, H, W) - channels are dimension 1
        # Standard image format: (Batch, H, W, C) - channels are last dimension
        # This is required for numpy/PIL compatibility and human viewing.
        images = images.permute(0, 2, 3, 1)

        # Convert to numpy array and standard image format.
        # Move tensor to CPU and convert to uint8 data type
        # (valid range 0-255).
        # Convert PyTorch tensor to numpy array for further
        # processing or saving.
        images = images.to('cpu', torch.uint8).numpy()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return the first (and typically only) image from the batch.
        # The generate function processes one prompt at a time,
        # so batch size is 1.
        return images[0]


def rescale(
    x: torch.Tensor,
    old_range: tuple,
    new_range: tuple,
    clamp: bool = False
) -> torch.Tensor:
    """
    Rescale tensor values from one range to another using
    linear transformation.

    Performs min-max normalization:
    maps values from [old_min, old_max] to [new_min, new_max]

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with values to rescale. Can be any shape.
    old_range : tuple
        Tuple of (old_min, old_max) representing the input value range.
    new_range : tuple
        Tuple of (new_min, new_max) representing the target output range.
    clamp : bool, optional
        If True, clamp output values to [new_min, new_max] bounds to prevent
        floating-point precision issues from exceeding the target range.
        Default is False.

    Returns
    -------
    torch.Tensor
        Rescaled tensor with same shape as input, with values in new_range.

    Example
    -------
    Rescale image from [0, 255] to [-1, 1]:
        >>> rescale(image_tensor, (0, 255), (-1, 1))
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    # Linear transformation: normalize to [0, 1] then scale to target range
    x = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    if clamp:
        # Clamp output to exact range bounds to prevent floating point errors
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep: int) -> torch.Tensor:
    """
    Create a sinusoidal time embedding for a given diffusion timestep.

    Generates a fixed positional encoding using sine and cosine functions
    at different frequencies, similar to the Transformer architecture.
    This embedding conditions the diffusion model on which step of the
    denoising process we're currently at.

    The embedding uses 160 frequency bands (produced by concatenating
    cos and sin, yielding 320 total dimensions to match the diffusion
    model's expected input size).

    Parameters
    ----------
    timestep : int
        The diffusion timestep (integer from 0 to num_steps-1) for which
        to generate the time embedding. Represents the noise level
        in the diffusion process.

    Returns
    -------
    torch.Tensor
        Sinusoidal time embedding of shape (1, 320).
        The shape (1, 320) is (batch_size=1, embedding_dims=320).
        Batch dimension is included for compatibility with model
        forward passes.

    Notes
    -----
    - Uses base frequency of 10000 to create logarithmically-spaced
    frequency bands.
    - Frequency formula: freq_i = 10000^(-2i/160) for i in [0, 160)
    - Output combines both sine and cosine: [cos(x), sin(x)] for robustness.
    - The specific value 160 frequency bands is arbitrary but matches
    this model's architecture.

    Example
    -------
    >>> embedding = get_time_embedding(10)  # Timestep 10
    >>> embedding.shape
    torch.Size([1, 320])
    """
    # Calculate frequency bands using exponential decay (logarithmic spacing).
    # This creates 160 different frequencies from high to low.
    # Formula: freq_i = 10000^(-2*i / d) where d=320
    # (embedding dimensions total)
    freqs = torch.pow(
        10000,
        -torch.arange(start=0, end=160, dtype=torch.float32) / 160
    )
    # Broadcast multiply: convert timestep scalar to (1, 160)
    # frequency-weighted values.
    # timestep: scalar integer -> (1, 1) tensor
    # freqs: (160,) -> (1, 160) by broadcasting
    # Result x: (1, 160) - scaled frequencies for the given timestep
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Concatenate cosine and sine components for richer positional information.
    # [cos(x), sin(x)] gives both phase and amplitude information
    # at each frequency.
    # Shape (1, 160) + (1, 160) -> (1, 320) concatenated on dimension -1
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
