"""Classes and functions to define datasets."""

import json
import os

import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt
import PIL.Image

from splax.cameras import CameraParams

Float = jt.Float
Array = jt.Array


@chex.dataclass
class ColmapDataset:
    """Dataset class for COLMAP datasets."""

    path: str
    cameras: list[CameraParams] | None = None
    images: Float[Array, "N H W 3"] | None = None
    resize_image_factor: float = 1.0

    @classmethod
    def from_path(cls, path: str, **kwargs) -> "ColmapDataset":
        return cls.make_dataset(path, **kwargs)

    @classmethod
    def make_dataset(cls, path: str, **kwargs) -> "ColmapDataset":
        """Create ColmapDataset from path."""

        downscale = kwargs.get("downscale", 1.0)
        cameras, image_paths = cls.make_cameras(path, downscale=downscale)
        images = cls.make_images(image_paths, downscale=downscale)

        return ColmapDataset(
            path=path, cameras=cameras, images=images, resize_image_factor=downscale
        )

    @classmethod
    def make_cameras(cls, path: str, downscale=1) -> tuple[list[CameraParams], list[str]]:
        transform_file = f"{path}/transforms.json"
        with open(transform_file, "r") as f:
            transform_data = json.load(f)

        focal_x = float(transform_data.get("fl_x"))
        focal_y = float(transform_data.get("fl_y"))
        W = int(transform_data.get("w") / downscale)
        H = int(transform_data.get("h") / downscale)

        camera_list = []
        image_paths = []

        for frame in transform_data["frames"]:
            img_path = f"{path}/{frame['file_path']}"
            if os.path.isfile(img_path) is False:
                continue

            image_paths.append(img_path)
            c2w_jnp = jnp.array(frame["transform_matrix"])

            # We need to convert OpenGL (-Z forward) to our convention (+Z forward).
            c2w_jnp = c2w_jnp.at[0:3, 2].multiply(-1)

            w2c_jnp = jnp.linalg.inv(c2w_jnp)

            cam = CameraParams.make_camera_params(
                w2c=w2c_jnp,
                focal_x=focal_x,
                focal_y=focal_y,
                img_width=W,
                img_height=H,
            )
            camera_list.append(cam)

        return camera_list, image_paths

    @classmethod
    def make_images(cls, image_paths, downscale: float = 1.0) -> Float[Array, "N H W 3"]:
        img_list = []
        for img_path in image_paths:
            img = PIL.Image.open(img_path).convert("RGB")
            img_array = jnp.array(img) / 255.0  # Normalize to [0, 1]
            img_array = jax.image.resize(
                img_array,
                (
                    int(img_array.shape[0] / downscale),
                    int(img_array.shape[1] / downscale),
                    3,
                ),
                method="bilinear",
            )
            img_list.append(img_array)

        images = jnp.stack(img_list, axis=0)  # (N, H, W, 3)
        return images

    def get_bounding_box_cameras(self) -> Float[Array, "N 6"]:
        """Get bounding box cameras."""
        bbox_cameras = []
        for cam in self.cameras:
            c2w = jnp.linalg.inv(cam.w2c)
            cam_pos = c2w[:3, 3]
            bbox_cameras.append(cam_pos.flatten())
        bbox_cameras_array = jnp.stack(bbox_cameras, axis=0)
        bbox_min = jnp.min(bbox_cameras_array, axis=0)
        bbox_max = jnp.max(bbox_cameras_array, axis=0)
        bbox_cameras = jnp.concatenate([bbox_min, bbox_max], axis=0)

        return bbox_cameras
