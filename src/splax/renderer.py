"""Define differentiable renderer"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt

from splax import cameras, gaussians

Float = jt.Float
Array = jt.Array


@chex.dataclass
class Renderer:
    """Differentiable renderer class."""

    def evaluate_gaussian(
        self,
        pixel: Float[Array, "2"],
        mean: Float[Array, "2"],
        inv_cov: Float[Array, "2 2"],
    ) -> Float[Array, "1"]:
        """Evaluate Gaussians at pixel location."""
        delta = pixel - mean
        dist = 0.5 * (delta @ inv_cov @ delta.T)
        return jnp.exp(-dist).flatten()

    def _pad_pixels(
        self, pixels: Float[Array, "P 2"], num_pixels: int, chunk_size: int
    ) -> Float[Array, "C S 2"]:
        """Pad pixels to enable chunking even when num_pixels is not divisible by chunk_size."""
        pad_size = (chunk_size - (num_pixels % chunk_size)) % chunk_size
        pixels_padded = jnp.pad(pixels, ((0, pad_size), (0, 0)), mode="edge")

        num_chunks = pixels_padded.shape[0] // chunk_size
        pixels_chunks = pixels_padded.reshape(num_chunks, chunk_size, 2)

        return pixels_chunks

    @partial(jax.jit, static_argnames=["chunk_size"])
    def render_image(
        self,
        params: gaussians.GaussianParams,
        camera_params: cameras.CameraParams,
        chunk_size: int = 4096,
    ) -> Float[Array, "H W 3"]:
        """
        Renders the scene.
        """
        # Project geometry
        project_all_gaussians = jax.vmap(cameras.project_gaussians, (0, None))
        means2d, cov2d, depths = project_all_gaussians(params, camera_params)

        # Sort by depth
        sorted_indices = jnp.argsort(depths)

        means2d = means2d[sorted_indices]  # (N, 2)
        cov2d = cov2d[sorted_indices]  # (N, 2, 2)

        # Add small epsilon to diagonal for numerical stability (and anti-aliasing)
        # This follows the original implementation
        cov2d = cov2d + jnp.eye(2) * 0.3

        opacity = jax.nn.sigmoid(params.opacity[sorted_indices])  # (N, 1)
        color = jax.nn.sigmoid(params.sh[sorted_indices])  # (N, 3)
        inv_cov2d = jnp.linalg.inv(cov2d)  # (N, 2, 2)

        # Create pixel grid
        xx, yy = jnp.meshgrid(
            jnp.arange(camera_params.img_width), jnp.arange(camera_params.img_height)
        )
        pixels = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)  # (H*W, 2)

        # Compute Mahalanobis distance for every pixel vs every gaussian
        # Because we cannot load the full (H*W, N, 1) array into memory, we chunk the pixels
        # We then use jax.lax.scan to iterate over the chunks

        # Pad pixels to be divisible by chunk_size
        num_pixels = pixels.shape[0]
        pixels_chunks = self._pad_pixels(pixels, num_pixels, chunk_size)

        evaluate_batch_gaussians = jax.vmap(
            self.evaluate_gaussian, (None, 0, 0)
        )  # (N, chunk_size, 1)
        evaluate_chunk_pixels = jax.vmap(
            evaluate_batch_gaussians, (0, None, None)
        )  # (chunk_size, N, 1)

        def scan_over_pixels(carry, batch_pixels):
            G = evaluate_chunk_pixels(batch_pixels, means2d, inv_cov2d)  # (chunk_size, N, 1)

            # Alpha compositing: T = sum(c_i * alpha_i * T_i) where T_i = prod(1-alpha_j)
            alpha = opacity * G  # (chunk_size, N, 1)

            # Transmittance T_i = prod_{j<i} (1 - alpha_j)
            # cumprod gives [1-a0, (1-a0)(1-a1), ...], but
            # we need [1, 1-a0, (1-a0)(1-a1), ...]
            transmittance = jnp.cumprod(1.0 - alpha + 1e-10, axis=1)  # (chunk_size, N, 1)
            transmittance = jnp.concatenate(
                [jnp.ones((chunk_size, 1, 1)), transmittance[:, :-1, :]], axis=1
            )

            pixel_color_chunk = jnp.sum(color * alpha * transmittance, axis=1)  # (chunk_size, 3)

            return carry, pixel_color_chunk

        # Use checkpointing to avoid storing intermediates for the backward pass
        # This reduces memory usage from O(H*W*N) to O(chunk_size*N) + O(H*W)
        scan_over_pixels = jax.remat(scan_over_pixels)

        _, pixel_colors = jax.lax.scan(
            scan_over_pixels, None, pixels_chunks
        )  # (num_chunks, chunk_size, 3)

        # Remove padding and reshape
        pixel_colors = pixel_colors.reshape(-1, 3)[:num_pixels]
        pixel_colors = pixel_colors.reshape(camera_params.img_height, camera_params.img_width, 3)

        return pixel_colors
