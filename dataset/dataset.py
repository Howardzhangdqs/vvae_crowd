import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from tqdm import tqdm


class FDSTVideoDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        """
        Args:
            root_dir (str): 数据集的根目录。
            mode (str): 数据集模式，'train' 或 'test'。
        """
        self.root_dir = root_dir
        self.mode = mode
        self.video_dir = os.path.join(root_dir, f'{mode}_videos')
        self.folders = sorted(os.listdir(self.video_dir), key=lambda x: int(x))

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        folder_path = os.path.join(self.video_dir, folder)

        # 读取视频
        video_path = os.path.join(folder_path, 'video.mp4')
        density_path = os.path.join(folder_path, 'density.mp4')

        video = self.read_video(video_path)
        density = self.read_video(density_path, grayscale=True)

        return video, density

    def read_video(self, video_path, grayscale=False):
        """
        读取视频并返回帧张量。

        Args:
            video_path (str): 视频路径。
            grayscale (bool): 是否以灰度模式读取视频。

        Returns:
            torch.Tensor: 视频帧张量，形状为 (C, T, H, W)。
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.expand_dims(frame, axis=-1)  # 增加通道维度

            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # (C, H, W)
            frames.append(frame)

        cap.release()

        # 将列表转换为张量并调整形状为 (C, T, H, W)
        frames = torch.stack(frames, dim=1)  # (C, T, H, W)
        return frames


# 使用示例
if __name__ == "__main__":
    root_dir = '/root/autodl-fs/FDST'
    mode = 'train'  # 或 'test'

    # 创建数据集实例
    dataset = FDSTVideoDataset(root_dir=root_dir, mode=mode)

    # 获取第一个样本
    video, density = dataset[0]
    print(f"Video shape: {video.shape}")
    print(f"Density map shape: {density.shape}")
