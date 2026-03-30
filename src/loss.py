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


class EASSIMLoss(nn.Module):
    def __init__(
            self, edge_weight: float = 0.5,
            eps: float = 1e-6, kernel_size: int = 7
    ):
        super().__init__()
        self.edge_weight = edge_weight
        self.ssim_loss = SSIMLoss(eps=eps, kernel_size=kernel_size)
        self.edge_loss = EdgeAwareLoss(eps=eps)

    def forward(self, pred, target, min_target, max_target):
        """
        Computes combined SSIM + λ * EdgeAware loss.
        Args:
            pred (torch.Tensor): Prediction of shape (B, 1, H, W)
            target (torch.Tensor): Target of shape (B, 1, H, W)
            min_target, max_target: Normalization parameters
        """
        ssim_loss = self.ssim_loss(pred, target, min_target, max_target)
        edge_loss = self.edge_loss(pred, target)
        total_loss = ssim_loss + self.edge_weight * edge_loss
        return total_loss, ssim_loss, edge_loss


class EAL1Loss(nn.Module):
    def __init__(self, edge_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.edge_weight = edge_weight
        self.L1_loss = nn.L1Loss()
        self.edge_loss = EdgeAwareLoss(eps=eps)

    def forward(self, pred, target, min_target, max_target):
        """
        Computes edge-aware L1 loss between prediction and target.
        Args:
            pred (torch.Tensor): Prediction of shape (B, 1, H, W)
            target (torch.Tensor): Target of shape (B, 1, H, W)
        """
        l1_loss = self.L1_loss(pred, target)
        edge_loss = self.edge_loss(pred, target)
        total_loss = l1_loss + self.edge_weight * edge_loss
        return total_loss, l1_loss, edge_loss


class ValMetrics(nn.Module):
    """
    We use this class to log useful metrics in validation and test
    Returns mean SSIM, GSSIM (not loss), L1 loss, and edge-aware loss.
    """
    def __init__(self, eps: float = 1e-6,  kernel_size: int = 7):
        super().__init__()
        self.ssim_loss = SSIMLoss(eps=eps, kernel_size=kernel_size)
        self.gssim_loss = GSSIMLoss(eps=eps, kernel_size=kernel_size)
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
        gssim_loss = self.gssim_loss(pred, target, min_target, max_target)
        l1_loss = self.L1_loss(pred, target)
        edge_loss = self.edge_loss(pred, target)
        return 1 - ssim_loss, 1 - gssim_loss, l1_loss, edge_loss


class GMSDLoss(nn.Module):
    def __init__(self, c: float = 170 / 255 ** 2, eps: float = 1e-6):
        super().__init__()
        self.c = c
        self.eps = eps

        # Define Sobel filters for gradient computation
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

    def gradient_magnitude(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)  # type: ignore
        gy = F.conv2d(x, self.sobel_y, padding=1)  # type: ignore
        return torch.sqrt(gx ** 2 + gy ** 2 + self.eps)

    def forward(self, pred, target, min_target, max_target):
        # Assumes pred and target are (B, 1, H, W)
        B = pred.shape[0]
        # Ensure min_target and max_target have correct shape for broadcasting
        min_target = min_target.view(B, 1, 1, 1)
        max_target = max_target.view(B, 1, 1, 1)

        # Compute data range per slice
        data_ranges = torch.clamp(max_target - min_target, min=self.eps)

        # Normalize each slice to [0, 1] range using broadcasting
        pred_normalized = (pred - min_target) / data_ranges
        target_normalized = (target - min_target) / data_ranges
        grad_pred = self.gradient_magnitude(pred_normalized)
        grad_target = self.gradient_magnitude(target_normalized)

        gms_map = (
            (2 * grad_pred * grad_target + self.c)
            /
            (grad_pred ** 2 + grad_target ** 2 + self.c)
        )
        return torch.std(gms_map, dim=[2, 3]).mean()


class GSSIMLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, kernel_size: int = 7):
        super().__init__()
        self.eps = eps
        self.kernel_size = kernel_size

        # Define Sobel filters for gradient computation
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

    def gradient_magnitude(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)  # type: ignore
        gy = F.conv2d(x, self.sobel_y, padding=1)  # type: ignore
        return torch.sqrt(gx ** 2 + gy ** 2 + self.eps)

    def forward(self, pred, target, min_target, max_target):
        B = pred.shape[0]
        min_target = min_target.view(B, 1, 1, 1)
        max_target = max_target.view(B, 1, 1, 1)
        data_ranges = torch.clamp(max_target - min_target, min=self.eps)
        pred = (pred - min_target) / data_ranges
        target = (target - min_target) / data_ranges

        grad_pred = self.gradient_magnitude(pred)
        grad_target = self.gradient_magnitude(target)

        mu_pred = F.avg_pool2d(
            pred, self.kernel_size, stride=1, padding=self.kernel_size // 2
        )
        mu_target = F.avg_pool2d(
            target, self.kernel_size, stride=1, padding=self.kernel_size // 2
        )

        # Luminance component (from original images)
        C1 = 0.01 ** 2
        luminance = (2 * mu_pred * mu_target + C1) / (
            mu_pred ** 2 + mu_target ** 2 + C1
        )
        # Contrast and structure components (from gradient maps)
        mu_grad_pred = F.avg_pool2d(
            grad_pred, self.kernel_size,
            stride=1, padding=self.kernel_size // 2
        )
        mu_grad_target = F.avg_pool2d(
            grad_target, self.kernel_size,
            stride=1, padding=self.kernel_size // 2
        )

        sigma_pred_sq = F.avg_pool2d(
            grad_pred ** 2, self.kernel_size,
            stride=1, padding=self.kernel_size // 2
        ) - mu_grad_pred ** 2
        sigma_target_sq = F.avg_pool2d(
            grad_target ** 2, self.kernel_size,
            stride=1, padding=self.kernel_size // 2
        ) - mu_grad_target ** 2
        sigma_cross = F.avg_pool2d(
            grad_pred * grad_target, self.kernel_size,
            stride=1, padding=self.kernel_size // 2
        ) - mu_grad_pred * mu_grad_target

        C2 = 0.03 ** 2
        contrast_structure = (2 * sigma_cross + C2) / (
            sigma_pred_sq + sigma_target_sq + C2
        )
        return 1 - (luminance * contrast_structure).mean()


class MS_SSIM_L1Loss(nn.Module):
    def __init__(
        self,
        gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
        K=(0.01, 0.03),
        alpha=0.025,
        compensation=20.0,
        eps=1e-6
    ):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.compensation = compensation
        self.eps = eps
        self.pad = int(2 * gaussian_sigmas[-1])
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros(
            (
                len(gaussian_sigmas), 1,
                filter_size, filter_size
            )
        )
        for idx, sigma in enumerate(gaussian_sigmas):
            g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(
                filter_size, sigma
            )

        # Register g_masks as a buffer
        self.register_buffer('g_masks', g_masks)

    def _fspecial_gauss_1d(self, size, sigma):
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, pred, target, min_target, max_target):
        B = pred.shape[0]
        min_target = min_target.view(B, 1, 1, 1)
        max_target = max_target.view(B, 1, 1, 1)
        data_ranges = torch.clamp(max_target - min_target, min=self.eps)

        # Normalize each slice to [0, 1]
        pred_normalized = (pred - min_target) / data_ranges
        target_normalized = (target - min_target) / data_ranges

        C1 = self.K[0] ** 2
        C2 = self.K[1] ** 2

        mux = F.conv2d(
            pred_normalized, self.g_masks, padding=self.pad  # type: ignore
        )
        muy = F.conv2d(
            target_normalized, self.g_masks, padding=self.pad  # type: ignore
        )

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = (
            F.conv2d(
                pred_normalized * pred_normalized,
                self.g_masks, padding=self.pad  # type: ignore
            ) - mux2
        )
        sigmay2 = (
            F.conv2d(
                target_normalized * target_normalized,
                self.g_masks, padding=self.pad  # type: ignore
            ) - muy2
        )
        sigmaxy = (
            F.conv2d(
                pred_normalized * target_normalized,
                self.g_masks, padding=self.pad  # type: ignore
            ) - muxy
        )

        luminance = (2 * muxy + C1) / (mux2 + muy2 + C1)
        cs = (2 * sigmaxy + C2) / (sigmax2 + sigmay2 + C2)

        lM = luminance[:, -1, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs

        loss_l1 = F.l1_loss(
            pred_normalized, target_normalized, reduction='none'
        )
        gaussian_l1 = F.conv2d(
            loss_l1,
            self.g_masks.narrow(dim=0, start=-1, length=1),   # type: ignore
            padding=self.pad
        ).mean(1)

        loss_mix = (
            self.alpha * loss_ms_ssim
            + (1 - self.alpha) * gaussian_l1
        )
        loss_mix = self.compensation * loss_mix
        return loss_mix.mean()
