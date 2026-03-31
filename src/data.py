import torch
import os
import subprocess
import pandas as pd
import numpy as np
from typing import Union, Callable, List
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


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

    def __getitem__(self, idx: int) -> dict:
        """Return only the metadata needed for loading"""
        row = self.csv.iloc[idx]
        return {
            'buffer_id': row['buffer'],
            'transform': self.transform,
            'idx': idx
        }


class PSIRNetDataTransform:
    """
    A callable class to convert numpy arrays to torch tensors
    """     
    def __call__(
        self,
        ir_kspace: np.ndarray,
        pd_kspace: np.ndarray,
        sens_maps: np.ndarray,
        target: np.ndarray,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, 
        torch.Tensor, torch.Tensor
    ]:
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
        return (
            ir_kspace, pd_kspace, sens_maps, target,
            mask, min_target, max_target
        )  # type: ignore


def read_buffer(buffer_id) -> np.ndarray:
    """
    Function to read a buffer in single pass
    Args:
        buffer_id (str): The ID of the buffer to read.
    Returns:
        np.ndarray: IR & PD k-space, sens_maps and the target.
    """
    with subprocess.Popen(
        [f"tyger buffer read {buffer_id}"],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    ) as proc:
        try:
            data = np.frombuffer(
                proc.stdout.read(), dtype=np.complex64
            ).reshape(91, 256, 192)  # 30 coils x 3 + 1
            proc.wait()
            if proc.returncode != 0:
                raise ValueError(f"Error in {buffer_id}")
            return data
        except Exception:
            proc.kill()
            raise


def process_single_item(item_metadata) -> PSIRNetSample:
    """Process a single item - used for parallel execution"""
    buffer_id = item_metadata['buffer_id']
    transform = item_metadata['transform']

    data = read_buffer(buffer_id)
    ir_kspace = data[:30, ...]
    pd_kspace = data[30:60, ...]
    sens_maps = data[60:90, ...]
    target = data[90:91, ...].real
    (ir_kspace, pd_kspace, sens_maps, target,
        mask, min_target, max_target) = transform(
         ir_kspace, pd_kspace, sens_maps, target
    )
    return PSIRNetSample(
        ir_kspace, pd_kspace, mask, sens_maps,
        target, min_target, max_target
    )


def parallel_collate_fn(batch: List[dict]) -> PSIRNetSample:
    """
    Collate function that reads all buffers in parallel
    """
    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
        samples = list(executor.map(process_single_item, batch))

    # Number of coils is uniform across all samples, so we can stack them
    ir_kspace = torch.stack([s.ir_kspace for s in samples])
    pd_kspace = torch.stack([s.pd_kspace for s in samples])
    mask = torch.stack([s.mask for s in samples])
    sens_maps = torch.stack([s.sens_maps for s in samples])
    target = torch.stack([s.target for s in samples])
    mins = torch.stack([s.min_target for s in samples])
    maxs = torch.stack([s.max_target for s in samples])
    return PSIRNetSample(
        ir_kspace, pd_kspace, mask, 
        sens_maps, target, mins, maxs
    )
