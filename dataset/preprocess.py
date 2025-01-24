import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def preprocess(root_dir, output_size=(150, 1080, 1920), mode='train'):
    """
    将图片和密度图整合为视频。

    Args:
        root_dir (str): 数据集的根目录。
        output_size (tuple): 输出的视频大小，格式为 (frames, height, width)。
        mode (str): 数据集模式，'train' 或 'test'。
    """
    data_dir = os.path.join(root_dir, f'{mode}_data')
    folders = sorted(os.listdir(data_dir), key=lambda x: int(x))

    # 计算帧的采样间隔
    frame_interval = max(1, 150 // output_size[0])

    # 定义图像转换
    def resize_image(image, size):
        return cv2.resize(image, (size[2], size[1]), interpolation=cv2.INTER_LINEAR)

    output_folder = os.path.join(root_dir, f'{mode}_videos')

    for folder in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(data_dir, folder)
        output_folder_path = os.path.join(output_folder, folder)
        output_video_path = os.path.join(output_folder_path, 'video.mp4')
        output_density_path = os.path.join(output_folder_path, 'density.mp4')

        os.makedirs(output_folder_path, exist_ok=True)

        # 初始化视频写入器
        height, width = output_size[1], output_size[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
        density_writer = cv2.VideoWriter(output_density_path, fourcc, 30, (width, height))

        for i in range(1, 151):
            if i % frame_interval != 0:
                continue

            # 读取图像
            img_path = os.path.join(folder_path, f'{i:03d}.jpg')
            img = cv2.imread(img_path)
            img = resize_image(img, output_size)

            # 读取密度图
            density_path = os.path.join(folder_path, f'{i:03d}.png')
            density = cv2.imread(density_path)
            density = cv2.cvtColor(density, cv2.COLOR_BGR2GRAY)
            density = resize_image(density, output_size)

            # 写入视频
            video_writer.write(img)
            density_writer.write(cv2.cvtColor(density, cv2.COLOR_GRAY2BGR))

        # 释放视频写入器
        video_writer.release()
        density_writer.release()
        tqdm.write(f'Processed folder: {folder_path}')


if __name__ == "__main__":
    preprocess(root_dir='/root/autodl-fs/FDST', mode='train')
    preprocess(root_dir='/root/autodl-fs/FDST', mode='test')
