import multiprocessing as mp
import numpy as np
import pandas as pd
import subprocess
from typing import List
from ismrmrd_utils import Message, IsmrmrdSource
from tqdm import tqdm


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


def process_buffer_header(args):
    """Process a single buffer and return its header info"""
    ind, recon_id = args
    try:
        recon = read_buffer(recon_id)[0]
        enc_limits = recon.encoding[0].encodingLimits
        rec_space = recon.encoding[0].reconSpace
        enc_space = recon.encoding[0].encodedSpace
        num_avg = enc_limits.average.maximum + 1
        num_slices = enc_limits.slice.maximum + 1
        num_coils = recon.acquisitionSystemInformation.receiverChannels
        recon_matrix_x = rec_space.matrixSize.x
        recon_matrix_y = rec_space.matrixSize.y
        enc_matrix_x = enc_space.matrixSize.x
        enc_matrix_y = enc_space.matrixSize.y
        header_info = (
            num_avg, num_slices, num_coils, recon_matrix_x,
            recon_matrix_y, enc_matrix_x, enc_matrix_y
        )
        return ind, header_info
    except Exception as e:
        return ind, f"Error: {e}"

lge_all_buffers = pd.read_csv('lge_all_buffers.csv')
buffer_args = [(ind, row.recon) for ind, row in enumerate(lge_all_buffers.itertuples())]

print(f"Processing {len(buffer_args)} buffers in parallel...")

# Process in parallel with tqdm progress tracking
with mp.Pool(processes=72) as pool:
    results = []
    for result in tqdm(
        pool.imap(process_buffer_header, buffer_args), 
        total=len(buffer_args),
        desc="Processing buffer headers"
    ):
        results.append(result)

# Sort results by index to maintain order
results = sorted(results)

# Add the new column to the dataframe
header_info_list = [result[1] for result in results]
lge_all_buffers['dim'] = header_info_list

# Save the expanded dataframe
lge_all_buffers.to_csv('lge_all_buffers.csv', index=False)
print(f"Saved expanded dataframe with {len(lge_all_buffers)} rows to 'lge_all_buffers.csv'")