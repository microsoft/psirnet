import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional import structural_similarity_index_measure as ssim


class SobelFilter(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Sobel kernels for grayscale images
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [0.,  0.,  0.],
            [1.,  2.,  1.]
        ]).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(input, self.sobel_x, padding=1)  # type: ignore
        grad_y = F.conv2d(input, self.sobel_y, padding=1)  # type: ignore

        # Compute magnitude with epsilon stability
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + self.eps)


class EdgeAwareLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.sobel = SobelFilter(eps=eps)

    def forward(self, pred, target):
        """
        Computes edge-aware loss between prediction and target.
        Args:
            pred (torch.Tensor): Prediction of shape (B, 1, H, W)
            target (torch.Tensor): Target of shape (B, 1, H, W)
        """
        pred_edges = self.sobel(pred)
        target_edges = self.sobel(target)
        return F.l1_loss(pred_edges, target_edges)


class SSIMLoss(nn.Module):
    def __init__(
        self, eps: float = 1e-6, kernel_size: int = 7,
    ):
        super().__init__()
        self.eps = eps
        self.kernel_size = kernel_size

    def forward(self, pred, target, min_target, max_target):
        """
        Computes the SSIM loss between prediction and target.
        Args:
            pred (torch.Tensor): Prediction.
            target (torch.Tensor): Target.
        """
        B = pred.shape[0]
        # Ensure min_target and max_target have correct shape for broadcasting
        min_target = min_target.view(B, 1, 1, 1)
        max_target = max_target.view(B, 1, 1, 1)

        # Compute data range per slice
        data_ranges = torch.clamp(max_target - min_target, min=self.eps)

        # Normalize each slice to [0, 1] range using broadcasting
        pred_normalized = (pred - min_target) / data_ranges
        target_normalized = (target - min_target) / data_ranges

        return 1 - ssim(
            pred_normalized, target_normalized,
            gaussian_kernel=False,
            kernel_size=self.kernel_size,
            data_range=(0.0, 1.0)
        )  # type: ignore


class ValMetrics(nn.Module):
    """
    We use this class to log useful metrics in validation and test
    Returns mean SSIM, L1 loss, and edge-aware loss.
    """
    def __init__(self, eps: float = 1e-6,  kernel_size: int = 7):
        super().__init__()
        self.ssim_loss = SSIMLoss(eps=eps, kernel_size=kernel_size)
        self.L1_loss = nn.L1Loss()
        self.edge_loss = EdgeAwareLoss(eps=eps)

    def forward(self, pred, target, min_target, max_target):
        """
        Computes each loss component between prediction and target.
        Args:
            pred (torch.Tensor): Prediction of shape (B, 1, H, W)
            target (torch.Tensor): Target of shape (B, 1, H, W)
        """
        ssim_loss = self.ssim_loss(pred, target, min_target, max_target)
        l1_loss = self.L1_loss(pred, target)
        edge_loss = self.edge_loss(pred, target)
        return 1 - ssim_loss, l1_loss, edge_loss
