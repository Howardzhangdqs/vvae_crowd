import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from pytorch_msssim import SSIM
import random


class VideoReconstructionLoss(nn.Module):
    def __init__(self,
                 device,
                 use_perceptual=True,
                 num_perceptual_frames=8):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.ssim = SSIM(data_range=1.0, channel=3, spatial_dims=2)

        # 感知损失相关配置
        self.use_perceptual = use_perceptual
        self.num_perceptual_frames = num_perceptual_frames

        if self.use_perceptual:
            # 初始化特征提取器
            self.resnet = resnet18(pretrained=True).to(device).eval()
            self.feature_extractor = nn.Sequential(
                *list(self.resnet.children())[:-2]  # 取到layer3
            )
            for param in self.feature_extractor.parameters():
                param.requires_grad_(False)

            # 时空下采样
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 64, 64))  # 空间下采样到64x64

    def _temporal_subsample(self, x):
        """时间维度子采样"""
        total_frames = x.shape[2]

        if total_frames <= self.num_perceptual_frames:
            return x

        # 均匀间隔采样
        step = total_frames // self.num_perceptual_frames
        indices = torch.linspace(0, total_frames-1, self.num_perceptual_frames).long()
        return x[:, :, indices]

    def perceptual_loss(self, x, recon_x):
        """
        改进的时空感知损失
        输入形状: [B, C, T, H, W]
        """
        # 1. 子采样时间维度
        x_sampled = self._temporal_subsample(x)
        recon_sampled = self._temporal_subsample(recon_x)

        # 2. 空间下采样
        x_sampled = self.spatial_pool(x_sampled)  # [B, C, T, 64, 64]
        recon_sampled = self.spatial_pool(recon_sampled)

        # 3. 合并批次和时间维度
        B, C, T = x_sampled.shape[:3]
        x_combined = x_sampled.permute(0, 2, 1, 3, 4).reshape(-1, C, 64, 64)
        recon_combined = recon_sampled.permute(0, 2, 1, 3, 4).reshape(-1, C, 64, 64)

        # 4. 提取特征
        with torch.no_grad():
            x_feats = self.feature_extractor(x_combined)
            recon_feats = self.feature_extractor(recon_combined)

        # 5. 计算特征差异
        return F.l1_loss(x_feats, recon_feats)

    def forward(self, x, recon_x, mu, logvar, epoch, max_epoch):
        # 多尺度结构相似性（逐帧计算）
        ssim_loss = 0
        for t in range(x.shape[2]):
            ssim_loss += 1 - self.ssim(x[:, :, t], recon_x[:, :, t])
        ssim_loss /= x.shape[2]

        # L1损失
        l1_loss = self.l1(recon_x, x).mean()

        # 感知损失
        percep_loss = 0.0
        if self.use_perceptual:
            percep_loss = self.perceptual_loss(x, recon_x)

        # KL退火策略
        kl_weight = min(epoch/(0.3*max_epoch), 1.0)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 损失组合
        total_loss = l1_loss + 0.8*ssim_loss
        if self.use_perceptual:
            total_loss += 0.5*percep_loss
        total_loss += kl_weight*kl_loss

        return {
            "total": total_loss,
            "l1": l1_loss,
            "ssim": ssim_loss,
            "perceptual": percep_loss if self.use_perceptual else torch.tensor(0.0),
            "kl": kl_loss,
            "kl_weight": torch.tensor(kl_weight)
        }
