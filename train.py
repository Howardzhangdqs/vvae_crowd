import torch
import wandb
import os
from pathlib import Path
from model import VVAE
from losses import VideoReconstructionLoss
from dataset import FDSTVideoDataset
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR


# 训练配置
config = {
    "lr": 3e-4,
    "batch_size": 8,
    "epochs": 1000,
    "grad_clip": 1.0,
    "weight_decay": 0.01,
    "seed": 42,
}


def init_weights(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')


def main():
    # 初始化wandb
    wandb.init(project="vae-crowd", config=config)
    cfg = wandb.config

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    # 初始化模型
    model = VVAE().to(device)
    model.apply(init_weights)

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(device=device)

    # 损失函数
    loss_fn = VideoReconstructionLoss(device)

    # 加载数据集
    train_dataset = FDSTVideoDataset(root_dir='/root/autodl-fs/FDST', mode='train')
    val_dataset = FDSTVideoDataset(root_dir='/root/autodl-fs/FDST', mode='val')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    best_val_loss = float('inf')
    Path(f"runs/{wandb.run.name}").mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        # 训练阶段
        model.train()
        train_losses = {}

        for batch_idx, videos in enumerate(train_loader):  # videos: (B, C, T, H, W)
            videos = videos.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device):
                recon_videos, mu, logvar = model(videos)
                loss_dict = loss_fn(videos, recon_videos, mu, logvar, epoch, cfg.epochs)

            scaler.scale(loss_dict["total"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # 累积损失
            for k, v in loss_dict.items():
                train_losses[k] = train_losses.get(k, 0) + v.item()/len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for videos in val_loader:
                videos = videos.to(device)
                recon_videos, mu, logvar = model(videos)
                loss_dict = loss_fn(videos, recon_videos, mu, logvar, epoch, cfg.epochs)
                val_loss += loss_dict["total"].item()/len(val_loader)

        # 学习率调度
        scheduler.step()

        # 记录到wandb
        log = {"epoch": epoch, "lr": optimizer.param_groups[0]['lr']}
        log.update({f"train/{k}": v for k, v in train_losses.items()})
        log["val/loss"] = val_loss

        # 保存样本可视化
        if epoch % 10 == 0:
            # 转换视频格式为wandb.Video期望的格式 (T, C, H, W)
            sample_vid = videos[0].permute(2, 0, 1, 3, 4).squeeze(1).cpu().numpy()
            recon_vid = recon_videos[0].permute(2, 0, 1, 3, 4).squeeze(1).cpu().numpy()

            log.update({
                "input_video": wandb.Video(sample_vid, fps=12),
                "recon_video": wandb.Video(recon_vid, fps=12)
            })

        wandb.log(log)

        # 保存最佳checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = f"runs/{wandb.run.name}/epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_val_loss,
            }, ckpt_path)

            # 上传checkpoint到wandb
            wandb.save(ckpt_path)


if __name__ == "__main__":
    main()
