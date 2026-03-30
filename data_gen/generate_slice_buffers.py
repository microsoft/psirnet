import multiprocessing as mp
import subprocess
from tqdm import tqdm
import numpy as np
import pandas as pd
from math_utils import compute_scc
from ismrmrd_utils import Message, IsmrmrdSource
from typing import List


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

def process_single_row(args):
    """Process a single row from the dataframe"""
    row_idx, row_data = args
    
    try:
        # Extract row data
        num_slices = row_data['num_slices']
        buffer_id = row_data['recon']
        kspace_id = row_data['kspace']
        sens_maps_id = row_data['sens_maps']
        patient_id = row_data['patient_id']
        study_id = row_data['study_id']
        session_id = row_data['session_id']
        measurement_id = row_data['measurement_id']
        
        # Read buffers once per row
        messages = read_buffer(buffer_id)[1:]
        kspace = read_buffer(kspace_id)[0].data
        sens_maps = read_buffer(sens_maps_id)
        
        chunk_size = len(messages) // num_slices
        new_buffer_ids = []
        
        # Process each slice
        for s in range(num_slices):
            try:
                # Your existing slice processing logic
                message_chunk = messages[s * chunk_size:(s + 1) * chunk_size]
                ir_candidates = []
                pd_candidates = []
                sens = sens_maps[s].data.squeeze().transpose(0, 2, 1)
                
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
                
                ir_candidates = np.stack(ir_candidates, axis=0)
                pd_candidates = np.stack(pd_candidates, axis=0)
                ir_idx = int(np.sum(np.abs(ir_candidates - ir_recon), axis=(1, 2)).argmin())
                pd_idx = int(np.sum(np.abs(pd_candidates - pd_recon), axis=(1, 2)).argmin())
                ir_kspace = kspace[:, :, 0, :, ir_idx, 0, s].transpose(2, 0, 1)
                pd_kspace = kspace[:, :, 0, :, pd_idx, 1, s].transpose(2, 0, 1)
                ir_recon /= 10
                pd_recon /= 10
                psir_target = (ir_recon * pd_recon.conj()).real * compute_scc(pd_recon) / \
                    np.sqrt(np.clip(np.abs(pd_recon)**2, a_min=1e-6, a_max=None))
                
                combined = np.concatenate([
                    ir_kspace, pd_kspace, sens,
                    np.expand_dims(psir_target, axis=0)
                ], axis=0).tobytes()
                
                # Create and write buffer
                new_buffer_id = subprocess.run([
                    f"tyger buffer create "
                    f"--tag description=slicebyslice "
                    f"--tag patient_id={patient_id} "
                    f"--tag study_id={study_id} "
                    f"--tag session_id={session_id} "
                    f"--tag measurement_id={measurement_id} "
                    f"--tag slice_idx={s}"
                ], shell=True, capture_output=True, text=True).stdout.strip()
                
                buffer_write = subprocess.run([
                    f"tyger buffer write {new_buffer_id}"
                ], shell=True, input=combined, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if buffer_write.returncode == 0:
                    new_buffer_ids.append(new_buffer_id)
                else:
                    new_buffer_ids.append(None)
                    
            except Exception as e:
                print(f"Error processing slice {s} for row {row_idx}: {e}")
                new_buffer_ids.append(None)
        
        return row_idx, new_buffer_ids, None
        
    except Exception as e:
        return row_idx, None, str(e)

# Main execution
def parallelize_buffer_processing():
    buffers = pd.read_csv("lge_filtered_most_common.csv")
    buffers['combined_buffer'] = None
    
    # Prepare arguments for parallel processing
    buffer_args = []
    for row_idx, row in enumerate(buffers.itertuples()):
        row_data = {
            'num_slices': row.num_slices,
            'recon': row.recon,
            'kspace': row.kspace,
            'sens_maps': row.sens_maps,
            'patient_id': row.patient_id,
            'study_id': row.study_id,
            'session_id': row.session_id,
            'measurement_id': row.measurement_id
        }
        buffer_args.append((row_idx, row_data))
    
    # Determine number of processes
    num_processes = 72
    print(f"Processing {len(buffer_args)} rows using {num_processes} processes...")
    
    # Process in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_row, buffer_args),
                total=len(buffer_args), desc="Processing rows"
            )
        )

    # Update the dataframe with results
    for row_idx, new_buffer_ids, error in results:
        if error:
            print(f"Row {row_idx} failed: {error}")
            buffers.at[row_idx, 'combined_buffer'] = None
        else:
            buffers.at[row_idx, 'combined_buffer'] = new_buffer_ids
    
    # Save the updated dataframe
    buffers.to_csv('lge_filtered_most_common_slice.csv', index=False)
    print(f"Saved updated dataframe to 'lge_filtered_most_common_slice.csv'")

    # Print summary
    successful_rows = sum(1 for _, buf_ids, err in results if err is None and buf_ids is not None)
    total_slices = sum(len(buf_ids) for _, buf_ids, err in results if buf_ids is not None)
    successful_slices = sum(sum(1 for bid in buf_ids if bid is not None) for _, buf_ids, err in results if buf_ids is not None)
    
    print(f"Summary:")
    print(f"  Successful rows: {successful_rows}/{len(results)}")
    print(f"  Successful slices: {successful_slices}/{total_slices}")
    return buffers

# Execute
if __name__ == "__main__":
    updated_buffers = parallelize_buffer_processing()