import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import PIL.Image
from jax.scipy.special import logit
from transformers import pipeline

from splax.utils import project_depth_to_3d, transform_to_world

Float = jt.Float
Array = jt.Array


@chex.dataclass
class GaussianParams:
    xyz: Float[Array, "N 3"]
    sh: Float[Array, "N 3"]
    opacity: Float[Array, "N 1"]  # sigmoid
    log_scale: Float[Array, "N 3"]  # log scale
    quaternion: Float[Array, "N 4"]  # quaternion (w, x, y, z)

    @classmethod
    def make_gaussian_params(
        cls,
        N: int,
    ) -> "GaussianParams":
        """Create default GaussianParams."""
        xyz = jax.random.uniform(jax.random.PRNGKey(0), (N, 3)) * 2.0 - 1.0
        sh = logit(jax.random.uniform(jax.random.PRNGKey(0), (N, 3)))
        opacity = logit(jax.random.uniform(jax.random.PRNGKey(0), (N, 1)))
        scale = jax.random.uniform(jax.random.PRNGKey(0), (N, 3)) * 0.01
        log_scale = jnp.log(scale)
        quaternion = jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]), (N, 1))

        return GaussianParams(
            xyz=xyz,
            sh=sh,
            opacity=opacity,
            log_scale=log_scale,
            quaternion=quaternion,
        )

    def at(self, idx):
        """Allows slicing: params[0] or params[idx]"""
        return jax.tree.map(lambda x: x[idx], self)

    def initialize_with_depth(self, images: Array, cameras: list) -> None:
        """Initialize the Gaussian positions using given depths and camera position."""

        num_gaussians = self.sh.shape[0]
        num_images = images.shape[0]
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
        images_pil = [
            PIL.Image.fromarray(np.array(images[i] * 255.0).astype("uint8"))
            for i in range(num_images)
        ]

        depth_torch = [pipe(image)["predicted_depth"] for image in images_pil]
        depth = [jnp.asarray(depth) for depth in depth_torch]
        depth = jnp.stack(depth, axis=0)  # [N, H, W]

        w2c = jnp.stack([cam.w2c for cam in cameras])
        c2w = jnp.linalg.inv(w2c)
        focal_x = jnp.array([cam.focal_x for cam in cameras])
        focal_y = jnp.array([cam.focal_y for cam in cameras])

        points_cam = jax.vmap(project_depth_to_3d, in_axes=(0, 0, 0, None, None))(
            depth,
            focal_x,
            focal_y,
            depth.shape[2] / 2,
            depth.shape[1] / 2,
        )  # [N, H, W, 3]

        points_world = jax.vmap(transform_to_world)(points_cam, c2w)
        points_world = points_world.reshape(-1, 3)

        # Sample to get N gaussians
        indices = jax.random.randint(
            shape=(num_gaussians,),
            key=jax.random.PRNGKey(0),
            minval=0,
            maxval=points_world.shape[0] - 1,
            dtype=jnp.int32,
        )
        rgb_colors = images.reshape(-1, 3)[indices]
        rgb_colors = jnp.clip(rgb_colors, 1e-5, 1.0 - 1e-5)
        self.sh = logit(rgb_colors)
        self.xyz = points_world[indices]


def quaternion_to_rot_matrix(q: Float[Array, "4"]) -> Float[Array, "3 3"]:
    """Convert a quaternion to a rotation matrix."""
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q

    return jnp.array(
        [
            [2 * (w**2 + x**2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 2 * (w**2 + y**2) - 1, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w**2 + z**2) - 1],
        ]
    )


def compute_covariance_3d(
    log_scale: Float[Array, "3"],
    quaternion: Float[Array, "4"],
) -> Float[Array, "3 3"]:
    """Compute the 3D covariance matrix: Sigma = R * S * S^T * R^T"""
    S = jnp.diag(jnp.exp(log_scale))  # (3, 3)
    R = quaternion_to_rot_matrix(quaternion)  # (3, 3)

    RS = R @ S  # (3, 3)
    Cov = RS @ RS.transpose(1, 0)  # (3, 3)

    return Cov
