import torch
import os
import pandas as pd
import numpy as np
from typing import Union, Callable, List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PSIRNetSample:
    ir_kspace: torch.Tensor
    pd_kspace: torch.Tensor
    mask: torch.Tensor
    sens_maps: torch.Tensor
    target: torch.Tensor
    min_target: torch.Tensor
    max_target: torch.Tensor


class SliceDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset
    """
    def __init__(
        self,
        csv_path: Union[str, Path, os.PathLike],
        transform: Callable,
    ) -> None:
        self.transform = transform
        self.csv = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, idx: int) -> PSIRNetSample:
        """Load NumPy arrays and return PyTorch tensors"""
        row = self.csv.iloc[idx]
        with np.load(row['npz_path'], mmap_mode='r') as npz:
            ir_kspace = npz['ir_kspace']
            pd_kspace = npz['pd_kspace']
            sens_maps = npz['sens_maps']
            target = npz['moco_psir']
        return self.transform(
            ir_kspace, pd_kspace, sens_maps, target
        )


class PSIRNetDataTransform:
    """
    A callable class to convert NumPy arrays to Torch tensors
    """     
    def __call__(
        self,
        ir_kspace: np.ndarray,
        pd_kspace: np.ndarray,
        sens_maps: np.ndarray,
        target: np.ndarray,
    ) -> PSIRNetSample:
        ir_kspace = torch.tensor(ir_kspace)
        pd_kspace = torch.tensor(pd_kspace)
        sens_maps = torch.tensor(sens_maps)
        target = torch.tensor(target)

        # We store the minimum and maximum of the target
        # to use it in the loss calculation (SSIM requires these)
        min_target, max_target = target.min(), target.max()

        # Extract the sampling mask with no further subsampling
        # Note that all coils and sets have the same mask
        mask = (ir_kspace != 0)[0:1, ...]
        return PSIRNetSample(
            ir_kspace, pd_kspace, sens_maps, target,
            mask, min_target, max_target
        )


def collate_fn(batch: List[PSIRNetSample]) -> PSIRNetSample:
    """
    Collate function for dataloader to batch PSIRNetSamples
    """
    # Number of coils is uniform across all samples, so we can stack them
    ir_kspace = torch.stack([s.ir_kspace for s in batch])
    pd_kspace = torch.stack([s.pd_kspace for s in batch])
    mask = torch.stack([s.mask for s in batch])
    sens_maps = torch.stack([s.sens_maps for s in batch])
    target = torch.stack([s.target for s in batch])
    mins = torch.stack([s.min_target for s in batch])
    maxs = torch.stack([s.max_target for s in batch])
    return PSIRNetSample(
        ir_kspace, pd_kspace, mask, 
        sens_maps, target, mins, maxs
    )
