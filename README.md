
# PSIRNet

## Project Repository Overview

This repository contains code and scripts for training and validating **PSIRNet**. The codebase was built on PyTorch with Lightning wrapper. It adapts the U-Net implementation from [fastMRI](https://github.com/facebookresearch/fastMRI), and its unrolled network architecture is similar to an [end-to-end variational network](https://rdcu.be/fbhjA).

For convenience, a devcontainer that supports CUDA acceleration (CUDA 12.8) was added. For other CUDA versions or ROCm, the Dockerfile can be modified. Before building the container, you might want to update the `devcontainer.json` to access the data inside the container. You can do so by uncommenting the `mounts` key and adding the data paths.

The repo follows the standard layout, and the devcontainer installs the PSIRNet package automatically with `pip install -e .`. If you prefer to install the requirements with pip or conda instead of using a Docker container, install the packages listed in lines 27–41 of the `Dockerfile` and then run `pip install -e .` from the repo root. 

## Trained Model Weights
We provide the [model weights](https://huggingface.co/microsoft/psirnet) for a PSIRNet with 845M parameters, trained on 640,000 slices of bright blood, dark blood, and wideband LGE data from 42,822 patients. The model configuration is detailed in the **`configs/psirnet.yaml`** file.

## Dataset Structure

The configuration file specifies a `.csv` file for training and validation. Each row of this `.csv` points to an
`.npz` file containing single-slice data with the following keys: `ir_kspace`, `pd_kspace`, `sens_maps`, and `moco_psir`.

 For each slice:
- The *k*-space and coil sensitivity data are of type numpy.complex64 with shape (1, coils, readout, phase encode) 
- The reference standard MOCO PSIR reconstruction with surface coil correction, `moco_psir`, has a numpy.float32 data type with shape (1, 1, readout, phase encode).

## Trainining, Validation, and Test

The main script performs training and validation of PSIRNet. The model capacity can be modified by adjusting the `num_cascades`, `sens_chans`, and `chans` variables in the configuration file.  

### Core Code Files

- **`scripts/main.py`**: Main script for training and evaluating PSIRNet.
- **`src/data.py`**: Contains data loading and preprocessing logic.
- **`src/loss.py`**: Implements SSIM loss function for training models.
- **`src/math_utils.py`**: Implements core mathematical operations.
- **`src/models.py`**: Defines the PSIRNet architecture.
- **`src/pl_data_module.py`**: Lightning data wrapper.
- **`inference/phantom.ipynb`**: Illustrates PSIRNet inference with phantom data.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
