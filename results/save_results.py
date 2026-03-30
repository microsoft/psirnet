import pathlib
import sys
import subprocess
import torch
import argparse
from torch import nn
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List
# Add the parent directory to the system path to import ismrmrd_utils
sys.path.append(str(pathlib.Path("/home/t-aatalik/psirnet/src").resolve()))
from math_utils import compute_scc
from ismrmrd_utils import Message, IsmrmrdSource


def load_model(pt_path: str) -> nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = torch.jit.load(pt_path, map_location=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    return model


def read_buffer(buffer_id) -> List[Message]:
    """
    Function to read a buffer in single pass
    Args:
        buffer_id (str): The ID of the buffer to read.
    Returns:
        list: A list of messages read from the buffer.
    """
    with subprocess.Popen(
        [f"tyger buffer read {buffer_id}"],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    ) as proc:
        try:
            msg_list = list(
                IsmrmrdSource(proc.stdout).source()  # type: ignore
            )
            proc.wait()
            if proc.returncode != 0:
                raise ValueError(f"Error in {buffer_id}")
            return msg_list
        except Exception:
            proc.kill()
            raise


def process_single_row(row_data: pd.Series, model: nn.Module) -> None:
    """Process a single row from the dataframe"""
    try:
        # Extract row data
        num_slices = row_data.num_slices
        header_id = row_data.header
        buffer_id = row_data.recon
        kspace_id = row_data.kspace
        sens_maps_id = row_data.sens_maps
        session_id = row_data.session_id
        measurement_id = row_data.measurement_id
        
        # Read buffers once per row
        encoding_dict = vars(vars(read_buffer(header_id)[0])['encoding'][0])
        encoding_dict = {k: v for k, v in encoding_dict.items() if v is not None}
        encoding_dict = np.array(encoding_dict, dtype=object)
        messages = read_buffer(buffer_id)[1:]
        kspace = read_buffer(kspace_id)[0].data
        sens_maps = read_buffer(sens_maps_id)
        
        chunk_size = len(messages) // num_slices
        psir_target = []
        psir_single_target = []
        ir_kspace = []
        pd_kspace = []
        sens = []
        vmins = []
        vmaxs = []
        # Process each slice
        for s in range(num_slices):
            message_chunk = messages[s * chunk_size:(s + 1) * chunk_size]
            ir_candidates = []
            pd_candidates = []
            sens.append(sens_maps[s].data.squeeze().transpose(0, 2, 1))
            for msg in message_chunk:
                hdr = msg.getHead()
                series, st = hdr.image_series_index, hdr.set
                if series == 100:
                    if st == 0:
                        ir_candidates.append(msg.data.squeeze().transpose())
                    elif st == 1:
                        pd_candidates.append(msg.data.squeeze().transpose())
                if series == 108:
                    if st == 0:
                        ir_recon = msg.data.squeeze().transpose()
                    elif st == 1:
                        pd_recon = msg.data.squeeze().transpose()
                if series == 111:
                    wc = int(msg.meta['GADGETRON_WindowCenter'])
                    ww = int(msg.meta['GADGETRON_WindowWidth'])
                    vmin, vmax = wc - ww / 2, wc + ww / 2
                    vmin = (vmin - 4096) / 1000
                    vmax = (vmax - 4096) / 1000
            vmins.append(vmin)
            vmaxs.append(vmax)  
            ir_candidates = np.stack(ir_candidates, axis=0)
            pd_candidates = np.stack(pd_candidates, axis=0)
            ir_idx = int(np.sum(np.abs(ir_candidates - ir_recon), axis=(1, 2)).argmin())
            pd_idx = int(np.sum(np.abs(pd_candidates - pd_recon), axis=(1, 2)).argmin())
            ir_kspace.append(kspace[:, :, 0, :, ir_idx, 0, s].transpose(2, 0, 1))
            pd_kspace.append(kspace[:, :, 0, :, pd_idx, 1, s].transpose(2, 0, 1))
            ir_recon /= 10
            ir_single_recon = ir_candidates[ir_idx] / 10
            pd_recon /= 10
            pd_single_recon = pd_candidates[pd_idx] / 10
            psir_target.append((ir_recon * pd_recon.conj()).real * compute_scc(pd_recon) / \
                np.sqrt(np.clip(np.abs(pd_recon)**2, a_min=1e-6, a_max=None)))
            psir_single_target.append((ir_single_recon * pd_single_recon.conj()).real * compute_scc(pd_single_recon) / \
                np.sqrt(np.clip(np.abs(pd_single_recon)**2, a_min=1e-6, a_max=None)))
        ir_kspace = np.stack(ir_kspace, axis=0)
        ir_kspace = torch.from_numpy(ir_kspace).to(dtype=torch.complex64, device='cuda')
        pd_kspace = np.stack(pd_kspace, axis=0)
        pd_kspace = torch.from_numpy(pd_kspace).to(dtype=torch.complex64, device='cuda')
        sens = np.stack(sens, axis=0)
        sens = torch.from_numpy(sens).to(dtype=torch.complex64, device='cuda')
        mask = (ir_kspace != 0)[:, 0:1, ...]
        reconstruction = model(ir_kspace, pd_kspace, mask, sens).squeeze(1).cpu().numpy()

        # Explicitly free memory
        del ir_kspace, pd_kspace, sens, mask
        torch.cuda.empty_cache()

        psir_target = np.stack(psir_target, axis=0)
        psir_single_target = np.stack(psir_single_target, axis=0)
        vmins = np.stack(vmins, axis=0)
        vmaxs = np.stack(vmaxs, axis=0)
        stacked = np.stack([psir_single_target, psir_target, reconstruction], axis=1)
        np.savez(
            f"/home/t-aatalik/psirnet/results/npz_files/{session_id}_{measurement_id}.npz",
            stacked=stacked, vmins=vmins, vmaxs=vmaxs, encoding_dict=encoding_dict,
        )
    except Exception as e:
        raise e
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a specific split of the test dataset')
    parser.add_argument('split_idx', type=int, help='Split index to process (0-7)')
    args = parser.parse_args()
    
    # Validate split index
    if args.split_idx < 0 or args.split_idx > 7:
        raise ValueError(f"Split index must be between 0 and 7, got {args.split_idx}")

    # Load the split
    df_test = pd.read_csv(f"/home/t-aatalik/psirnet/results/temp_csv_files/df_test_split_{args.split_idx}.csv")
    model = load_model('/home/t-aatalik/psirnet/checkpoints/PSIRNet.pt')
    print(f"Processing split_idx: {args.split_idx}")
    for row in tqdm(df_test.itertuples(), total=len(df_test)):
        try:
            process_single_row(row, model)
        except Exception as e:
            print(f"Error processing row {row.Index}: {e}")
            continue