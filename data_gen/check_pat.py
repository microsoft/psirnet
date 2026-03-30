import subprocess
import sys
import pathlib
from typing import List
from tqdm import tqdm
from collections import Counter
import pandas as pd
sys.path.append(str(pathlib.Path("/home/t-aatalik/psirnet/src").resolve()))
from ismrmrd_utils import Message, IsmrmrdSource

def read_raw_buffer(buffer_id) -> List[Message]:
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

if __name__ == "__main__":
    df = pd.read_csv("/home/t-aatalik/psirnet/csv_files/large_files/lge_filtered_most_common.csv")
    df_test = df[238425:]
    df_test = df_test.reset_index(drop=True)
    header_ids = df_test['header'].to_list()
    pats = []
    pbar = tqdm(header_ids, desc="Processing headers")
    for header_id in pbar:
        encoding_dict = vars(vars(read_raw_buffer(header_id)[0])['encoding'][0])
        pats.append(encoding_dict['parallelImaging'].accelerationFactor.kspace_encoding_step_1)
        counter = Counter(pats)
        pbar.set_postfix_str(f"PATs: {dict(counter)}")