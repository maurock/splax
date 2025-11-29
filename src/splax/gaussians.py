import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt

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
        """Create default GaussianParams with zeros."""
        xyz = jax.random.uniform(jax.random.PRNGKey(0), (N, 3)) * 2.0 - 1.0
        sh = jax.random.uniform(jax.random.PRNGKey(0), (N, 3))
        opacity = jax.random.uniform(jax.random.PRNGKey(0), (N, 1))
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
