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
        seed: int = 0,
        sh_degree: int = 3,
    ) -> "GaussianParams":
        """Create default GaussianParams."""
        random_key = jax.random.PRNGKey(seed)
        xyz = jax.random.uniform(random_key, (N, 3)) * 2.0 - 1.0
        sh_dimensions = (sh_degree + 1) ** 2
        sh_dc = logit(jnp.clip(jax.random.uniform(random_key, (N, 3)), 1e-6, 1 - 1e-6))
        sh_rest = jnp.zeros(shape=(N, (sh_dimensions - 1) * 3))
        sh = jnp.concatenate([sh_dc, sh_rest], axis=-1)
        opacity = logit(jnp.clip(jax.random.uniform(random_key, (N, 1)), 1e-6, 1 - 1e-6))
        scale = jnp.ones((N, 3)) * 0.01
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

    def initialize_with_camera_rays(self, cameras: list) -> Float[Array, "N 3"]:
        """Initialize the Gaussian positions along camera rays."""
        points_world_all = []
        for i, cam in enumerate(cameras):
            pixel_xx, pixel_yy = jnp.meshgrid(jnp.arange(cam.img_width), jnp.arange(cam.img_height))
            pixel_pos = jnp.stack(
                [
                    pixel_xx - cam.img_width / 2,
                    pixel_yy - cam.img_height / 2,
                    jnp.ones_like(pixel_xx) * cam.focal_x,
                ],
                axis=-1,
            )  # [H, W, 3]
            directions = pixel_pos / jnp.linalg.norm(pixel_pos, axis=-1, keepdims=True)
            points_cam = directions * jax.random.uniform(
                jax.random.PRNGKey(i), shape=directions.shape, minval=0.2, maxval=6
            )  # [H, W, 3]
            points_world = transform_to_world(points_cam, cam.c2w)  # [H, W, 3]
            points_world = points_world.reshape(-1, 3)
            points_world_all.append(points_world)
        points_world_all = jnp.concatenate(points_world_all, axis=0)
        indices = jax.random.randint(
            shape=(self.xyz.shape[0],),
            key=jax.random.PRNGKey(0),
            minval=0,
            maxval=points_world_all.shape[0] - 1,
            dtype=jnp.int32,
        )
        return points_world_all[indices]


def quaternion_to_rot_matrix(q: Float[Array, "4"]) -> Float[Array, "3 3"]:
    """Convert a quaternion to a rotation matrix."""
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-9)
    w, x, y, z = q

    return jnp.array(
        [
            [2 * (w**2 + x**2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 2 * (w**2 + y**2) - 1, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w**2 + z**2) - 1],
        ]
    )


def eval_sh(sh_coeffs: Float[Array, "N D"], view_dirs: Float[Array, "N 3"]) -> Float[Array, "N 3"]:
    """Evaluates Spherical Harmonics up to degree 3.

    Args:
        sh_coeffs: SH coefficients
        view_dirs: view directions from the camera to the gaussians"""
    dir = view_dirs / (jnp.linalg.norm(view_dirs, axis=-1, keepdims=True) + 1e-9)
    x, y, z = dir[..., 0], dir[..., 1], dir[..., 2]

    sh_coeffs = sh_coeffs.reshape(-1, 16, 3)

    c0 = 0.28209479177387814

    c1 = 0.4886025119029199

    c2_0 = 1.0925484305920792
    c2_1 = -1.0925484305920792
    c2_2 = 0.31539156525252005
    c2_3 = -1.0925484305920792
    c2_4 = 0.5462742152960396

    c3_0 = -0.5900435899266435
    c3_1 = 2.890611442640554
    c3_2 = -0.4570457994644658
    c3_3 = 0.3731763325901154
    c3_4 = -0.4570457994644658
    c3_5 = 1.445305721320277
    c3_6 = -0.5900435899266435

    result = c0 * sh_coeffs[:, 0]

    if sh_coeffs.shape[1] > 1:
        result = (
            result
            - c1 * y[:, None] * sh_coeffs[:, 1]
            + c1 * z[:, None] * sh_coeffs[:, 2]
            - c1 * x[:, None] * sh_coeffs[:, 3]
        )

    if sh_coeffs.shape[1] > 4:
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z

        result = (
            result
            + c2_0 * xy[:, None] * sh_coeffs[:, 4]
            + c2_1 * yz[:, None] * sh_coeffs[:, 5]
            + c2_2 * (2.0 * zz - xx - yy)[:, None] * sh_coeffs[:, 6]
            + c2_3 * xz[:, None] * sh_coeffs[:, 7]
            + c2_4 * (xx - yy)[:, None] * sh_coeffs[:, 8]
        )

    if sh_coeffs.shape[1] > 9:
        result = (
            result
            + c3_0 * y[:, None] * (3 * x * x - y * y)[:, None] * sh_coeffs[:, 9]
            + c3_1 * xy[:, None] * z[:, None] * sh_coeffs[:, 10]
            + c3_2 * y[:, None] * (4 * z * z - x * x - y * y)[:, None] * sh_coeffs[:, 11]
            + c3_3 * z[:, None] * (2 * z * z - 3 * x * x - 3 * y * y)[:, None] * sh_coeffs[:, 12]
            + c3_4 * x[:, None] * (4 * z * z - x * x - y * y)[:, None] * sh_coeffs[:, 13]
            + c3_5 * z[:, None] * (x * x - y * y)[:, None] * sh_coeffs[:, 14]
            + c3_6 * x[:, None] * (x * x - 3 * y * y)[:, None] * sh_coeffs[:, 15]
        )

    return jax.nn.sigmoid(result)


def compute_covariance_3d(
    log_scale: Float[Array, "3"],
    quaternion: Float[Array, "4"],
) -> Float[Array, "3 3"]:
    """Compute the 3D covariance matrix: Sigma = R * S * S^T * R^T"""
    scale = jnp.exp(log_scale)
    scale = jnp.clip(scale, a_max=1000.0)
    S = jnp.diag(scale)  # (3, 3)
    R = quaternion_to_rot_matrix(quaternion)  # (3, 3)

    RS = R @ S  # (3, 3)
    Cov = RS @ RS.transpose(1, 0)  # (3, 3)

    return Cov
