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
class SplaxRenderer:
    """Differentiable renderer class."""

    def evaluate_gaussians_per_pixel(
        self,
        pixel: Float[Array, "2"],
        mean: Float[Array, "N 2"],
        inv_cov: Float[Array, "N 3"],
    ) -> jax.Array:
        """
        Evaluate Gaussians using precomputed inverse covariance coefficients as in original paper.
        P(x, y) = exp(-0.5 (A*dx^2 + B*dx*dy + C*dy^2))
        """
        dx = pixel[0] - mean[:, 0]
        dy = pixel[1] - mean[:, 1]

        power = -0.5 * (inv_cov[:, 0] * dx**2 + inv_cov[:, 1] * dx * dy + inv_cov[:, 2] * dy**2)

        return jnp.exp(power)

    @partial(jax.jit, static_argnames=["tile_size", "max_gaussians"])
    def render_image(
        self,
        params: gaussians.GaussianParams,
        camera_params: cameras.CameraParams,
        tile_size: int = 16,
        max_gaussians: int = 500,
    ) -> Float[Array, "H W 3"]:
        """
        Renders the scene using a tile-based approach.
        """
        project_all_gaussians = jax.vmap(cameras.project_gaussians, (0, None))
        means2d, cov2d, depths = project_all_gaussians(params, camera_params)

        # Push invalid gaussians to the end by setting depth to infinity
        valid_mask = depths > 0.2
        depths = jnp.where(valid_mask, depths, jnp.inf)

        sorted_indices = jnp.argsort(depths)

        means2d = means2d[sorted_indices]  # [N, 2]
        cov2d = cov2d[sorted_indices]  # [N, 2, 2]
        opacity = jax.nn.sigmoid(params.opacity[sorted_indices])  # [N, 1]
        color = jax.nn.sigmoid(params.sh[sorted_indices])  # [N, 3]
        valid_mask = valid_mask[sorted_indices]

        cov2d = cov2d + jnp.eye(2) * 0.3  # Numerical stability
        inv_cov2d = jnp.linalg.inv(cov2d)  # [N, 2, 2]

        # Compute gaussians radius (3 sigma for coverage 99.7%)
        # The radius is defined as the higest variance (largest eigenvalue
        # of the covariance matrix). Eigenvalues of 2x2 matrix are T/2 +- sqrt(T^2/4 - D)
        # where T is trace and D is determinant. We only need the largest eigenvalue,
        # so we can compute it as T/2 + sqrt(T^2/4 - D)
        a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
        trace = a + c
        det = a * c - b * b
        term = jnp.sqrt(jnp.maximum(0.0, trace**2 - 4 * det))
        lambda_max = (trace + term) / 2.0
        radii = 3.0 * jnp.sqrt(lambda_max)  # (N,)

        radii = jnp.where(valid_mask, radii, 0.0)

        H, W = camera_params.img_height, camera_params.img_width
        num_tiles_x = (W + tile_size - 1) // tile_size
        num_tiles_y = (H + tile_size - 1) // tile_size

        tx = jnp.arange(num_tiles_x) * tile_size
        ty = jnp.arange(num_tiles_y) * tile_size
        tile_min_x, tile_min_y = jnp.meshgrid(tx, ty)
        # tile_min_x -> [0, 16, 32, ..., 0, 16, 32, ...]
        # tile_min_y -> [0, 0, 0, ..., 16, 16, 16, ...]
        tile_min_x, tile_min_y = tile_min_x.flatten(), tile_min_y.flatten()
        tile_max_x, tile_max_y = tile_min_x + tile_size, tile_min_y + tile_size

        g_min_x = means2d[:, 0] - radii
        g_max_x = means2d[:, 0] + radii
        g_min_y = means2d[:, 1] - radii
        g_max_y = means2d[:, 1] + radii

        # Tiles shape [T], gaussians [N] -> overlap [T, N]
        overlap = (
            (tile_min_x[:, None] < g_max_x[None, :])
            & (tile_max_x[:, None] > g_min_x[None, :])
            & (tile_min_y[:, None] < g_max_y[None, :])
            & (tile_max_y[:, None] > g_min_y[None, :])
        )

        # Select top K gaussians per tile (where K = max_gaussians)
        # where overlap is True. Gaussians are sorted by depth,
        # so we prefer smaller indices
        N = means2d.shape[0]
        indices = jnp.arange(N)
        scores = jnp.where(overlap, -indices.astype(jnp.float32), -jnp.inf)

        # Get top K indices (meaning the closest gaussians)
        top_scores, top_indices = jax.lax.top_k(scores, min(N, max_gaussians))  # [T, K]
        valid_mask = top_scores > -jnp.inf  # [T, K]

        tile_means = means2d[top_indices]
        tile_inv_covs = inv_cov2d[top_indices]
        tile_opacities = opacity[top_indices]
        tile_colors = color[top_indices]
        tile_valid = top_scores > -jnp.inf

        def render_tile(min_x, min_y, means, inv_covs, opacities, colors, valid):
            """
            Renders a single tile. Input shapes are [K, ...], output is [tile_size, tile_size, 3]
            """
            py, px = jnp.meshgrid(jnp.arange(tile_size), jnp.arange(tile_size), indexing="ij")
            px = min_x + px  # Transform to global coords
            py = min_y + py

            pixels = jnp.stack([px, py], axis=-1).reshape(-1, 2)  # [P, 2]

            # Optimize calculation of Malahanobis distance (as in original paper)
            # inv_cov[idx] = [[A, B/2], [B/2, C]]
            # P = [x y] @ inv_cov @ [x y]^T
            # P(x, y) = -0.5 (A * dx^2 + B * dx * dy + C * dy^2)
            inv_covs_A = inv_covs[:, 0, 0]
            inv_covs_B = inv_covs[:, 0, 1] * 2.0
            inv_covs_C = inv_covs[:, 1, 1]
            conics = jnp.stack([inv_covs_A, inv_covs_B, inv_covs_C], axis=-1)

            G = jax.vmap(self.evaluate_gaussians_per_pixel, (0, None, None))(pixels, means, conics)

            alpha = opacities.flatten() * G * valid[None, :]  # Apply valid mask here

            # Transmittance T_i = prod(1 - alpha_j) for j < i
            # We append a 1.0 at the start for the first gaussian
            transmittance = jnp.cumprod(1.0 - alpha + 1e-10, axis=1)
            transmittance = jnp.concatenate(
                [jnp.ones((pixels.shape[0], 1)), transmittance[:, :-1]], axis=1
            )

            # [P, K, 3] * [P, K, 1] * [P, K, 1] -> [P, 3]
            pixel_color = jnp.sum(
                colors[None, :, :] * alpha[..., None] * transmittance[..., None], axis=1
            )

            # Mask out-of-bounds pixels (image width or height might not be divisible by tile_size)
            pixel_valid_mask = (pixels[:, 0] < W) & (pixels[:, 1] < H)
            pixel_color = pixel_color * pixel_valid_mask[:, None]

            return pixel_color.reshape(tile_size, tile_size, 3)

        rendered_tiles = jax.vmap(render_tile)(
            tile_min_x,
            tile_min_y,
            tile_means,
            tile_inv_covs,
            tile_opacities,
            tile_colors,
            tile_valid,
        )

        # Reconstruct image
        grid = rendered_tiles.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
        # [Ny, Nx, T, T, 3] -> [Ny, T, Nx, T, 3]
        grid = grid.transpose(0, 2, 1, 3, 4)
        # [Ny, T, Nx, T, 3] -> [Ny * T, Nx * T, 3]
        image = grid.reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)

        # Crop padding
        return image[:H, :W, :]
