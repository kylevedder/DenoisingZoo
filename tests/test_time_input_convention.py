"""Tests for time input conventions."""

import torch

from dataloaders.base_dataloaders import (
    make_time_input,
    make_ununified_flow_matching_input,
)
from dataloaders.kmeans_dataset import GaussianKMeansDataset


def test_make_time_input_flowmatching_sets_end_time():
    t = torch.rand(6, 1)
    time_input = make_time_input(t)

    assert time_input.shape == (6, 2)
    assert torch.allclose(time_input[:, 0:1], t)
    assert torch.allclose(time_input[:, 1:2], torch.ones_like(t))


def test_make_time_input_meanflow_pairs_r_t():
    r = torch.rand(4, 1)
    t = torch.rand(4, 1)
    time_input = make_time_input(t, r=r, time_channels=2)

    assert torch.allclose(time_input[:, 0:1], r)
    assert torch.allclose(time_input[:, 1:2], t)


def test_dataset_unified_input_has_end_time_one():
    ds = GaussianKMeansDataset(centroids=[[0.0, 0.0], [1.0, 1.0]], length=4)
    item = ds[0]

    unified = item.unified_input.unsqueeze(0)
    ununified = make_ununified_flow_matching_input(unified, num_time_channels=2)
    t = ununified.t

    assert torch.allclose(t[:, 1:2], torch.ones_like(t[:, 1:2]))
