import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


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


def check_single_buffer(args):
    """Check a single buffer and return error info"""
    ind, buffer_id = args
    try:
        _ = read_buffer(buffer_id)
        return ind, None  # No error
    except Exception as e:
        return ind, str(e)  # Return error message


train_df = pd.read_csv('/home/t-aatalik/psirnet/csv_files/train.csv')
val_df = pd.read_csv('/home/t-aatalik/psirnet/csv_files/val.csv')
test_df = pd.read_csv('/home/t-aatalik/psirnet/csv_files/test.csv')
combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Prepare arguments for parallel processing
buffer_args = [(ind, buffer) for ind, buffer in enumerate(combined['buffer'])]

# Process in parallel
with ThreadPoolExecutor(max_workers=80) as executor:
    results = list(tqdm(
        executor.map(check_single_buffer, buffer_args),
        total=len(buffer_args),
        desc="Checking buffers"
    ))

# Collect error information
error_data = []
for ind, error in results:
    if error is not None:
        error_data.append({
            'index': ind,
            'buffer_id': combined.iloc[ind]['buffer'],
            'error_message': error
        })

# Create DataFrame with error information
error_df = pd.DataFrame(error_data)

# Print summary
total_buffers = len(buffer_args)
error_count = len(error_df)
success_count = total_buffers - error_count

print("Summary:")
print(f"  Total buffers checked: {total_buffers}")
print(f"  Successful reads: {success_count}")
print(f"  Failed reads: {error_count}")

if not error_df.empty:
    print(error_df.head())
    error_df.to_csv('buffer_read_errors.csv', index=False)
    print("Error details saved to 'buffer_read_errors.csv'")
else:
    print("No errors found!")

# Display the error DataFrame
error_df
