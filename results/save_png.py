import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_file(fname):
    h, w = 256, 192
    dpi = plt.rcParams['figure.dpi']
    
    file = np.load(f"/home/t-aatalik/psirnet/results/npz_files/{fname}", allow_pickle=True)
    stacked, vmin, vmax, encoding_size = file['stacked'], file['vmins'], file['vmaxs'], file['encoding_dict'].item()
    pat_factor = encoding_size['parallelImaging'].accelerationFactor.kspace_encoding_step_1
    enc_min = encoding_size['encodingLimits'].kspace_encoding_step_1.minimum
    enc_max = encoding_size['encodingLimits'].kspace_encoding_step_1.maximum
    num_averages = encoding_size['encodingLimits'].average.maximum - \
        encoding_size['encodingLimits'].average.minimum + 1
    session_id = fname.split('_')[0]
    measurement_id = fname.split('_')[1].split('.')[0]
    save_fname = f"{session_id}_{measurement_id}_{num_averages}_{enc_min}_{pat_factor}_{enc_max}"
    
    for slice in range(len(stacked)):
        fig = plt.figure(figsize=(3 * w/dpi, h/dpi), dpi=dpi, frameon=False)
        gs = fig.add_gridspec(1, 3, wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax1.imshow(stacked[slice, 0], vmin=vmin[slice], vmax=vmax[slice], cmap='gray', interpolation='none')
        ax1.axis('off')
        ax2.imshow(stacked[slice, 1], vmin=vmin[slice], vmax=vmax[slice], cmap='gray', interpolation='none')
        ax2.axis('off')
        ax3.imshow(stacked[slice, 2], vmin=vmin[slice], vmax=vmax[slice], cmap='gray', interpolation='none')
        ax3.axis('off')
        save_slice_name = save_fname + f"_{slice}"
        plt.savefig(f'../results/png_files/{save_slice_name}.png', bbox_inches='tight', pad_inches=0)
        plt.close('all')  # Extra safety to close all figures

if __name__ == "__main__":
    files = os.listdir('../results/npz_files/')

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_file, fname) for fname in files]
        for future in tqdm(as_completed(futures), total=len(files)):
            future.result()