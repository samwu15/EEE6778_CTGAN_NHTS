
# src/cyclegan_min.py
# Minimal CycleGAN components (Generator & Discriminator) for 256x256 images.
# This is a light, instructional reference implementation suitable for toy training.
# NOTE: Designed for clarity, not for speed. For production use, consider a well-tested library.
import torch
import torch.nn as nn

# --- building blocks ---
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
        )
    def forward(self, x):
        return x + self.main(x)

class GeneratorResnet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_res=6, ngf=64):
        super().__init__()
        layers = []
        # c7s1-64
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        ]
        # d128, d256
        mult = 1
        for _ in range(2):
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True)
            ]
            mult *= 2
        # R256 x n_res
        for _ in range(n_res):
            layers += [ResidualBlock(ngf * mult)]
        # u128, u64
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult // 2, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True)
            ]
            mult //= 2
        # c7s1-3
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_c, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DiscriminatorPatchGAN(nn.Module):
    def __init__(self, in_c=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4; padw = 1
        sequence = [
            nn.Conv2d(in_c, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult, affine=False, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        # last
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
