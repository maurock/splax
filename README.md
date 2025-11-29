# Splax (Work in progress)
Implementation of Gaussian Splatting in pure JAX.

# Installation
```bash
conda env create -f environment.yaml
conda activate splax
```

# Download data
Example: [NeRF FOX](https://gitlab.liris.cnrs.fr/tpickles/instant-ngp-tomography/-/tree/master/data/nerf/fox)

Example of expected format (currently):
```bash
├─ data/
|   ├─ fox/
|   |   ├─ images/
|   |   |   ├─ <image1>.png
|   |   |   ├─ transforms.json
```

# Training example
```python
from splax import gaussians, loader, renderer
import optax
import random
from functools import partial

data_path = ""  # Path to dataset folder, e.g. /home/user/splax/data/fox
num_gaussians = 10000
dataset_downscale_factor = 4
max_gaussians_per_tile = 500
tile_size = 16
epochs = 10000
lr = 1e-3

# Initialize Gaussians and load data
gaussian_params = gaussians.GaussianParams.make_gaussian_params(num_gaussians)
data = loader.load_dataset(dataset_type="colmap", path=data_path, downscale=dataset_downscale_factor)
gaussian_params.initialize_with_depth(
    images=data.images,
    cameras=data.cameras
)

@partial(jax.jit)
def train_step(gaussian_params, target_image, camera, opt_state):
    @jax.value_and_grad
    def loss_fn(params):
        pixel_colors = r.render_image(params, camera, max_gaussians=max_gaussians_per_tile, tile_size=tile_size)
        loss = jnp.mean(jnp.abs(pixel_colors - target_image))
        cov_loss = 0.01 * jnp.mean(jnp.exp(params.log_scale))
        return loss + cov_loss

    loss, grads = loss_fn(gaussian_params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_gaussian_params = optax.apply_updates(gaussian_params, updates)

    return loss, new_gaussian_params, new_opt_state

r = renderer.SplaxRenderer()
optimizer = optax.adam(learning_rate=lr)
opt_state = optimizer.init(gaussian_params)
for i in range(epochs):
    cam_idx = random.randint(0, len(data.cameras) - 1)
    loss, gaussian_params, opt_state = train_step(
        gaussian_params,
        data.images[cam_idx],
        data.cameras[cam_idx],
        opt_state
    )
    pbar.set_postfix_str(f"Loss: {loss}")
```


# Roadmap
- [x] Tile-based rasterizer
- [ ] Initialize with depth
- [ ] Add datasets
- [ ] Optimize with Pallas (Mosaic GPU / Triton) 
