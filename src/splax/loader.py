"""Module for loading different types of datasets."""

from splax.dataset import ColmapDataset

# Extend this to include other dataset types. E.g. AnyDataset = ColmapDataset | OtherDataset
# Dataset defintion can be added to splax.dataset and imported here.
AnyDataset = ColmapDataset
_SUPPORTED_DATASETS = {"colmap": ColmapDataset}


def load_dataset(dataset_type: str, path: str, **kwargs) -> AnyDataset:
    if dataset_type not in _SUPPORTED_DATASETS:
        options = list(_SUPPORTED_DATASETS.keys())
        raise ValueError(f"Invalid type: {dataset_type}. Options: {options}")

    return _SUPPORTED_DATASETS[dataset_type].from_path(path, **kwargs)
