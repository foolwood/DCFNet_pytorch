import torch  # pytorch 0.4.0! fft
import torch.nn as nn
import numpy as np


def conj(input):
    input[:, 1] *= -1
    return input


def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)


class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        self.model_alphaf = []
        self.model_xf = []
        self.config = config
        self.numel_zf = 1

    def forward(self, z, x):
        z = self.feature(z)
        zf = torch.rfft(z, signal_ndim=2, normalized=False, onesided=False)
        self.numel_zf = float(np.prod(z[0].shape).astype(np.float32))
        kf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True) / self.numel_zf
        alphaf = self.config.yf / (kf + self.config.lambda0)

        x = self.feature(x)
        xf = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False)
        kzf = torch.sum(complex_mulconj(xf, zf), dim=1, keepdim=True) / self.numel_zf
        response = torch.irfft(complex_mul(kzf, alphaf), signal_ndim=2, onesided=False)
        return response


if __name__ == '__main__':

    # network test
    net = DCFNet()
    net.eval()



