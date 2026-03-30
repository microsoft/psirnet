import torch
from torch import nn
from typing import Tuple
from torch.nn import functional as F
from .math_utils import (
    norm_tensor, unnorm_tensor, expand, reduce,
    complex_to_chan_dim, chan_dim_to_complex,
    pad, unpad, tfftc, itfftc, compute_scc_torch
)


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model from the fastMRI library.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList(
            [ConvBlock(in_chans, chans, drop_prob)]
        )
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(
                ConvBlock(ch, ch * 2, drop_prob)
            )
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/bottom if needed to handle odd input dim
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two conv layers followed by
    instance normalization, LeakyReLU activation and dropout (optional).
    If use_deform is True, the first conv layer is deformable.
    """

    def __init__(
        self, in_chans: int, out_chans: int, drop_prob: float
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
            use_deform: Whether to use deformable conv2d in the first layer.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_chans, out_chans, kernel_size=3, padding=1, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(
                out_chans, out_chans, kernel_size=3, padding=1, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)


class NormUnet(nn.Module):
    """
    A modified version of fastMRI's NormUnet that supports
    complex-valued inputs and outputs, although the U-Net
    itself operates on real-valued tensors.
    """
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def _norm_tensor(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return norm_tensor(x)

    def _unnorm_tensor(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return unnorm_tensor(x, mean, std)

    def _complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return complex_to_chan_dim(x)

    def _chan_dim_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input tensor must be 4D (b, 2 * c, h, w)"
        return chan_dim_to_complex(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get shapes for unet and normalize
        x, mean, std = self._norm_tensor(x)
        x = self._complex_to_chan_dim(x)
        x, pad_sizes = pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = unpad(x, *pad_sizes)
        x = self._chan_dim_to_complex(x)
        x = self._unnorm_tensor(x, mean, std)
        return x


class NormUnet2(nn.Module):
    """
    A modified version of fastMRI's NormUnet that supports
    complex-valued inputs and outputs, although the U-Net
    itself operates on real-valued tensors. This design
    is for PD-IR joint processing in the refinement blocks.
    """
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 4,
        out_chans: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def _norm_tensor(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return norm_tensor(x)

    def _unnorm_tensor(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return unnorm_tensor(x, mean, std)

    def _complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return complex_to_chan_dim(x)

    def _chan_dim_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input tensor must be 4D (b, 2 * c, h, w)"
        return chan_dim_to_complex(x)

    def forward(
            self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get shapes for unet and normalize
        x, mean_x, std_x = self._norm_tensor(x)
        x = self._complex_to_chan_dim(x)
        x, pad_sizes_x = pad(x)

        y, mean_y, std_y = self._norm_tensor(y)
        y = self._complex_to_chan_dim(y)
        y, pad_sizes_y = pad(y)

        xy = torch.cat([x, y], dim=1)
        xy = self.unet(xy)
        x, y = torch.split(xy, xy.shape[1] // 2, dim=1)

        x = unpad(x, *pad_sizes_x)
        x = self._chan_dim_to_complex(x)
        x = self._unnorm_tensor(x, mean_x, std_x)

        y = unpad(y, *pad_sizes_y)
        y = self._chan_dim_to_complex(y)
        y = self._unnorm_tensor(y, mean_y, std_y)
        return x, y


class SensitivityModel(nn.Module):
    """
    Model for finetuning the estimated sensitivity maps.
    This model uses a NormUnet to enhance the coil
    sensitivity maps which were computed from the ref
    scan using standard techniques, e.g., ESPIRiT.
    """
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def _chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w = x.shape
        return x.view(b * c, 1, h, w), b

    def _batch_chans_to_chan_dim(
        self, x: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        bc, _, h, w = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w)

    def _divide_by_rss(self, x: torch.Tensor) -> torch.Tensor:
        x_rss = torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=1, keepdim=True))
        return x / x_rss

    def forward(
        self,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        images, batches = self._chans_to_batch_dim(sens_maps)
        return self._divide_by_rss(
            self._batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network
    for SCC+PSIR+MOCO+Avg reconstruction.
    Used in the baseline model which does not
    utilize patient-specific side information.
    """
    def __init__(self, model: nn.Module):
        """
        Args:
            model: Refinement network, e.g., NormUnet2
        """
        super().__init__()
        self.model = model
        self.ir_dc_weight = nn.Parameter(torch.ones(1))
        self.pd_dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        ir_current_image: torch.Tensor,
        pd_current_image: torch.Tensor,
        ir_ref_kspace: torch.Tensor,
        pd_ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ir_current_image: IR image estimate.
            pd_current_image: PD image estimate.
            ir_ref_kspace: IR ref k-space data.
            pd_ref_kspace: PD ref k-space data.
            mask: Undersampling mask.
            sens_maps: Coil sensitivity maps.

        Returns:
            Updated image estimates after one DC + refinement step.
        """
        ir_current_kspace = tfftc(expand(ir_current_image, sens_maps))
        pd_current_kspace = tfftc(expand(pd_current_image, sens_maps))

        zero = torch.zeros(1, 1, 1, 1).to(ir_current_kspace)

        ir_soft_dc = torch.where(mask, ir_current_kspace - ir_ref_kspace, zero)
        ir_soft_dc = reduce(itfftc(ir_soft_dc), sens_maps)
        ir_soft_dc = ir_soft_dc * torch.abs(self.ir_dc_weight)

        pd_soft_dc = torch.where(mask, pd_current_kspace - pd_ref_kspace, zero)
        pd_soft_dc = reduce(itfftc(pd_soft_dc), sens_maps)
        pd_soft_dc = pd_soft_dc * torch.abs(self.pd_dc_weight)

        ir_model_term, pd_model_term = self.model(
            ir_current_image, pd_current_image
        )
        out_ir = ir_current_image - ir_soft_dc - ir_model_term
        out_pd = pd_current_image - pd_soft_dc - pd_model_term
        return out_ir, out_pd


class PSIRNet(nn.Module):
    """
    A full PSIRNet implemented in image domain.
    We take 1 repetition of IR and PD k-space as well as
    the sensitivity maps as input to the model.
    Directly outputs the SCC PSIR image.
    """
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 16,
        sens_pools: int = 4,
        chans: int = 32,
        pools: int = 4,
        eps: float = 1e-6
    ):
        """
        Args:
            num_cascades: Number of DC + refinement steps.
            sens_chans: Number of channels in the sensitivity model.
            sens_pools: Number of pooling layers in the sensitivity model.
            chans: Number of channels in the image refinement model.
            pools: Number of pooling layers in the image refinement model.
            use_deform: Whether to use deformable conv2d in the models.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        if sens_chans > 0:
            self.sens_net = SensitivityModel(sens_chans, sens_pools)
        else:
            self.sens_net = nn.Identity()
        self.cascades = nn.ModuleList(
            [
                VarNetBlock(
                    NormUnet2(chans, pools)
                ) for _ in range(num_cascades)
            ]
        )

    def forward(
        self,
        ir_masked_kspace: torch.Tensor,
        pd_masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        sens_maps = self.sens_net(sens_maps)
        ir_image_pred = reduce(itfftc(ir_masked_kspace), sens_maps)
        pd_image_pred = reduce(itfftc(pd_masked_kspace), sens_maps)
        for cascade in self.cascades:
            ir_image_pred, pd_image_pred = cascade(
                ir_image_pred, pd_image_pred,
                ir_masked_kspace, pd_masked_kspace,
                mask, sens_maps
            )
        # At this point we have IR and PD MOCO Avg reconstructions
        # Finally, compute the PSIR SCC image
        return ((ir_image_pred * pd_image_pred.conj()).real
                * compute_scc_torch(pd_image_pred)
                / torch.sqrt((pd_image_pred.abs()**2).clamp(min=self.eps)))
