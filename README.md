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

# Features
- [x] Tile-based rasterizer
- [ ] Initialize with depth
- [ ] Add datasets
- [ ] Optimize with Pallas (Mosaic GPU / Triton) 
