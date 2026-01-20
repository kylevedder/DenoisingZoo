"""Tests for dataset item dataclasses.

Ensures all Item classes can be instantiated and have required fields.
This catches dataclass inheritance issues early.
"""

import torch
import pytest


class TestDatasetItemInstantiation:
    """Verify all dataset Item classes can be instantiated."""

    def test_cifar10_item(self):
        from dataloaders.cifar10_dataset import CIFAR10Item
        item = CIFAR10Item(
            input=torch.zeros(3, 32, 32),
            t=torch.tensor([0.5]),
            target=torch.zeros(3, 32, 32),
            unified_input=torch.zeros(4, 32, 32),
            raw_source=torch.zeros(3, 32, 32),
            raw_target=torch.zeros(3, 32, 32),
        )
        assert item.raw_source is not None
        assert item.raw_target is not None

    def test_celeba_item(self):
        from dataloaders.celeba_dataset import CelebAItem
        item = CelebAItem(
            input=torch.zeros(3, 64, 64),
            t=torch.tensor([0.5]),
            target=torch.zeros(3, 64, 64),
            unified_input=torch.zeros(4, 64, 64),
            raw_source=torch.zeros(3, 64, 64),
            raw_target=torch.zeros(3, 64, 64),
        )
        assert item.raw_source is not None
        assert item.raw_target is not None

    def test_imagenet_item(self):
        from dataloaders.imagenet_dataset import ImageNetItem
        item = ImageNetItem(
            input=torch.zeros(4, 32, 32),
            t=torch.tensor([0.5]),
            target=torch.zeros(4, 32, 32),
            unified_input=torch.zeros(5, 32, 32),
            raw_source=torch.zeros(4, 32, 32),
            raw_target=torch.zeros(4, 32, 32),
            label=0,
        )
        assert item.raw_source is not None
        assert item.raw_target is not None
        assert item.label == 0

    def test_deterministic_item(self):
        from dataloaders.deterministic_dataset import DeterministicItem
        item = DeterministicItem(
            input=torch.zeros(3, 32, 32),
            t=torch.tensor([0.5]),
            target=torch.zeros(3, 32, 32),
            unified_input=torch.zeros(4, 32, 32),
            raw_source=torch.zeros(3, 32, 32),
            raw_target=torch.zeros(3, 32, 32),
            label=0,
        )
        assert item.raw_source is not None
        assert item.raw_target is not None

    def test_kmeans_item(self):
        from dataloaders.kmeans_dataset import KMeansItem
        item = KMeansItem(
            input=torch.zeros(2),
            t=torch.tensor([0.5]),
            target=torch.zeros(2),
            unified_input=torch.zeros(3),
            raw_source=torch.zeros(2),
            raw_target=torch.zeros(2),
        )
        assert item.raw_source is not None
        assert item.raw_target is not None

    def test_synthetic_shapes_item(self):
        from dataloaders.synthetic_shapes_dataset import SyntheticShapesItem
        item = SyntheticShapesItem(
            input=torch.zeros(3, 64, 64),
            t=torch.tensor([0.5]),
            target=torch.zeros(3, 64, 64),
            unified_input=torch.zeros(4, 64, 64),
            raw_source=torch.zeros(3, 64, 64),
            raw_target=torch.zeros(3, 64, 64),
        )
        assert item.raw_source is not None
        assert item.raw_target is not None


class TestDatasetGetItem:
    """Verify all datasets return valid items with raw_source/raw_target."""

    def test_deterministic_dataset(self):
        from dataloaders.deterministic_dataset import DeterministicFlowDataset
        ds = DeterministicFlowDataset(num_classes=2, samples_per_class=2)
        item = ds[0]
        assert hasattr(item, 'raw_source')
        assert hasattr(item, 'raw_target')
        assert item.raw_source is not None
        assert item.raw_target is not None

    def test_synthetic_shapes_dataset(self):
        from dataloaders.synthetic_shapes_dataset import SyntheticShapesFlowDataset
        ds = SyntheticShapesFlowDataset(num_samples=2)
        item = ds[0]
        assert hasattr(item, 'raw_source')
        assert hasattr(item, 'raw_target')
        assert item.raw_source is not None
        assert item.raw_target is not None

    def test_kmeans_dataset(self):
        from dataloaders.kmeans_dataset import GaussianKMeansDataset
        ds = GaussianKMeansDataset(centroids=[[0, 0], [1, 1]], length=10)
        item = ds[0]
        assert hasattr(item, 'raw_source')
        assert hasattr(item, 'raw_target')
        assert item.raw_source is not None
        assert item.raw_target is not None


class TestToDict:
    """Verify to_dict() includes raw_source and raw_target."""

    def test_deterministic_to_dict(self):
        from dataloaders.deterministic_dataset import DeterministicFlowDataset
        ds = DeterministicFlowDataset(num_classes=2, samples_per_class=2)
        item = ds[0]
        d = item.to_dict()
        assert 'raw_source' in d
        assert 'raw_target' in d
        assert d['raw_source'] is not None
        assert d['raw_target'] is not None

    def test_synthetic_to_dict(self):
        from dataloaders.synthetic_shapes_dataset import SyntheticShapesFlowDataset
        ds = SyntheticShapesFlowDataset(num_samples=2)
        item = ds[0]
        d = item.to_dict()
        assert 'raw_source' in d
        assert 'raw_target' in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
