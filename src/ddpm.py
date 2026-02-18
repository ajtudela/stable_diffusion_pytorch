"""DDPM sampler for reverse diffusion process.

Denoising Diffusion Probabilistic Models (DDPM) sampler that handles
the reverse diffusion process for generative modeling and inference.
Computes noise schedules and performs iterative denoising steps that
convert noise into coherent images.
"""

import torch
import numpy as np


class DDPMSampler:
    """DDPM sampler for iterative denoising and noise-to-image generation."""

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120
    ) -> None:
        """Initialize the DDPM sampler with noise schedules.

        Sets up variance schedule and cumulative products for efficient
        noise scheduling.

        Parameters
        ----------
        generator : torch.Generator
            PyTorch random number generator for reproducible sampling.
        num_training_steps : int, optional
            Total number of diffusion steps. Default is 1000.
        beta_start : float, optional
            Starting variance value. Default is 0.00085.
        beta_end : float, optional
            Ending variance value. Default is 0.0120.
        """
        # Create noise schedule using linear spacing in sqrt space,
        # then square.
        # This provides better numerical properties than
        # pure linear scheduling.
        self.betas = torch.linspace(
            beta_start ** 0.5,
            beta_end ** 0.5,
            num_training_steps,
            dtype=torch.float32) ** 2

        # Compute alphas: fraction of signal retained at each step (1 - beta).
        self.alphas = 1.0 - self.betas

        # Cumulative product of alphas for reparameterization enabling
        # direct sampling at any timestep without intermediate steps.
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Constant 1.0 for boundary conditions when previous timestep < 0.
        self.one = torch.tensor(1.0)

        # Store random generator for reproducibility.
        self.generator = generator

        # Store total training steps count.
        self.num_train_timesteps = num_training_steps

        # Initialize timesteps in reverse order (1000, 999, ..., 1, 0).
        # These are subsampled based on num_inference_steps during inference.
        self.timesteps = torch.from_numpy(
            np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50) -> None:
        """Set number of inference steps and resample timesteps accordingly.

        During inference, we reduce steps from 1000 to fewer (e.g., 50) by
        uniform sampling. Fewer steps = faster but potentially lower quality.

        Parameters
        ----------
        num_inference_steps : int, optional
            Number of denoising steps for inference. Default is 50.
        """
        # Store the number of inference steps to use.
        self.num_inference_steps = num_inference_steps

        # Calculate stride: uniformly sample every nth timestep.
        step_ratio = self.num_train_timesteps // self.num_inference_steps

        # Create uniformly-spaced timesteps from the full schedule.
        # Multiply by step_ratio, round, reverse order, and convert to int64.
        timesteps = (np.arange(0, num_inference_steps)
                     * step_ratio).round()[::-1].copy().astype(np.int64)

        # Convert to PyTorch tensor for use in denoising loop.
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """Get the previous timestep in the denoising sequence.

        Computes the previous timestep by subtracting the step ratio.

        Parameters
        ----------
        timestep : int
            Current timestep index.

        Returns
        -------
        int
            Previous timestep (may be negative).
        """
        # Subtract stride to get previous timestep in the sampled schedule.
        prev_t = timestep - (
            self.num_train_timesteps // self.num_inference_steps)
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        """Compute the variance for the reverse diffusion step.

        Calculates variance for reverse process denoising using DDPM formula.

        Parameters
        ----------
        timestep : int
            Current timestep for which to compute variance.

        Returns
        -------
        torch.Tensor
            Variance at the given timestep, clamped to minimum of 1e-20.
        """
        # Get previous timestep for accessing alpha values at t-1.
        prev_t = self._get_previous_timestep(timestep)

        # Get cumulative product of alphas at current timestep.
        alpha_prod_t = self.alphas_cumprod[timestep]

        # Get cumulative product at previous timestep (use 1.0 if negative).
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one)

        # Compute clamped beta relating current and previous alphas.
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # Apply DDPM formula (7) for posterior variance.
        variance = (1 - alpha_prod_t_prev) / \
            (1 - alpha_prod_t) * current_beta_t

        # Clamp to prevent numerical issues (log of 0, division by 0).
        variance = torch.clamp(variance, min=1e-20)
        return variance

    def set_strength(self, strength: float = 1.0) -> None:
        """Set denoising strength for image-to-image generation.

        Controls how much denoising to perform. strength=1.0 is full denoising,
        strength=0.0 means minimal changes to the input image.

        Parameters
        ----------
        strength : float, optional
            Denoising strength in [0.0, 1.0]. Default is 1.0.
        """
        # Calculate which step to start from based on strength.
        # Higher strength = more denoising steps = more change to image.
        start_step = self.num_inference_steps - \
            int(self.num_inference_steps * strength)

        # Slice timesteps to skip early steps (shorter denoising schedule).
        self.timesteps = self.timesteps[start_step:]

        # Store start step for potential later use.
        self.start_step = start_step

    def step(
        self,
        timestep: int,
        latents: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """Perform a single reverse diffusion denoising step.

        Given predicted noise from the model, compute the previous latent state
        by removing the predicted noise. Core operation in inference.

        Parameters
        ----------
        timestep : int
            Current diffusion timestep.
        latents : torch.Tensor
            Noisy latent sample x_t.
            Shape: (batch, channels, height, width).
        model_output : torch.Tensor
            Predicted noise from the diffusion U-NET. Same shape as latents.

        Returns
        -------
        torch.Tensor
            Denoised latent sample x_{t-1}. Same shape as input.
            For t=0, no stochastic noise; for t>0, Gaussian noise is added.
        """
        # Store timestep in shorter variable name.
        t = timestep

        # Get previous timestep for accessing alpha schedules at t-1.
        prev_t = self._get_previous_timestep(t)

        # Get cumulative product of alphas at current timestep.
        alpha_prod_t = self.alphas_cumprod[t]

        # Get cumulative product at previous timestep (use 1.0 if negative).
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one)

        # Compute complementary noise levels (1 - alpha_prod).
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Compute single-step alpha and beta values.
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Reconstruct x_0 from predicted noise (DDPM formula 15).
        # x_0 = (x_t - sqrt(1 - a_t) * pred) / sqrt(a_t)
        pred_original_sample = (latents - beta_prod_t **
                                0.5 * model_output) / alpha_prod_t ** 0.5

        # Coefficient for predicted original sample in the mean calculation.
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t

        # Coefficient for current sample x_t in the mean calculation.
        current_sample_coeff = (
            current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t)

        # Compute the mean of x_{t-1} distribution by weighted combination.
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample +
            current_sample_coeff * latents
        )

        # Initialize variance (stochastic noise) to zero.
        # At t=0 (final step), we add no noise (deterministic).
        variance = 0
        if t > 0:
            # For t > 0, sample Gaussian noise for stochasticity.
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype
            )
            # Scale by square root of variance for this timestep.
            variance = self._get_variance(t) ** 0.5 * noise

        # Reparameterization: X = mean + std * Z where Z ~ N(0,1).
        # Add scaled noise to the predicted mean for final denoised sample.
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        """Add noise to samples according to the diffusion noise schedule.

        Implements forward diffusion process: adds noise to clean samples at
        a given timestep. Used in image-to-image generation to encode input.

        Formula: x_t = sqrt(a_t) * x_0 + sqrt(1 - a_t) * noise, where a_t
        is the cumulative product of alphas.

        Parameters
        ----------
        original_samples : torch.FloatTensor
            Clean input samples. Shape: (batch, channels, height, width).
        timesteps : torch.IntTensor
            Timestep indices. Values in [0, num_training_steps).

        Returns
        -------
        torch.FloatTensor
            Noisy samples at timestep. Same shape as original_samples.
        """
        # Move alpha schedule to match device and dtype of input samples.
        # Ensures GPU compatibility and numerical precision matching.
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype)

        # Move timesteps to the same device as samples for indexing operations.
        timesteps = timesteps.to(original_samples.device)

        # Extract and compute sqrt of alpha product at given timesteps.
        # This is the coefficient for the original signal (how much to retain).
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5

        # Flatten to 1D before dimension expansion for broadcasting.
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        # Add dimensions to match original_samples shape for broadcasting.
        # If original_samples is 4D (batch, channels, H, W),
        # expand from (batch,) to (batch, 1, 1, 1).
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # Compute sqrt(1 - alphas_cumprod): noise coefficient.
        # Increases with timestep (more noise at higher t).
        # At t=0: near 0. At t=max: near 1 (pure noise).
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        # Flatten before dimension expansion.
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        # Expand dimensions to match original_samples for broadcasting.
        while (
                len(sqrt_one_minus_alpha_prod.shape) <
                len(original_samples.shape)
        ):
            sqrt_one_minus_alpha_prod = (
                sqrt_one_minus_alpha_prod.unsqueeze(-1))

        # Sample Gaussian noise with same shape as original samples.
        # Using the stored generator for reproducibility and consistency.
        # Shape: same as original_samples
        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype
        )

        # Combine clean samples and noise using schedule (DDPM formula 4).
        # x_t = sqrt(a_t)*x_0 + sqrt(1-a_t)*noise
        # Low t: mostly original. High t: mostly noise.
        # Shape: same as original_samples
        noisy_samples = (sqrt_alpha_prod * original_samples) + \
            sqrt_one_minus_alpha_prod * noise
        return noisy_samples
