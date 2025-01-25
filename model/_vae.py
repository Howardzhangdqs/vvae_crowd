import torch
import torch.nn as nn
from vae import AutoencoderKLCausal3D

from typing import Tuple, List, Dict


class VVAE(nn.Module):
    def __init__(self, **kwargs):
        super(VVAE, self).__init__()
        self.vae = AutoencoderKLCausal3D(
            down_block_types=("DownEncoderBlockCausal3D", "DownEncoderBlockCausal3D", "DownEncoderBlockCausal3D"),
            up_block_types=("UpDecoderBlockCausal3D", "UpDecoderBlockCausal3D", "UpDecoderBlockCausal3D"),
            block_out_channels=(64, 32, 64),
            out_channels=4,
            **kwargs
        )
        self.align = nn.Conv3d(
            in_channels=4,
            out_channels=3,
            kernel_size=(4, 1, 1),
            stride=1,
            padding=(3, 0, 0)
        )

    def encode(self, x):
        return self.vae.encode(x).latent_dist.mode()

    def decode(self, z):
        x = self.vae.decode(z).sample
        x = self.align(x)
        return x

    def forward(self, x):
        x = self.vae(x).sample
        x = self.align(x)
        return x


if __name__ == '__main__':
    model = VVAE()
    x = torch.randn(1, 3, 32, 64, 64)
    z = model.encode(x)
    x_recon = model.decode(z)
    print(x.shape, z.shape, x_recon.shape)
    print(model(x).sample.shape)
