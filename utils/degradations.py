"""Degradation operators and mask helpers for inverse problems.

Includes deblurring (Gaussian / motion), super-resolution, decolorization,
and inpainting mask generators.
"""

from functools import partial

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F

from motionblur.motionblur import Kernel
from torch_utils.resizer import Resizer

from utils.dct import LinearDCT, apply_linear_2d

from torch.fft  import fftshift, ifftshift

def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.
    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    
    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(torch.fft.fftn(torch.view_as_complex(data), dim=(-2, -1), norm=norm))
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.
    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def fft2_m(x):
  """ FFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
  """ IFFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))



class BlurKernel(torch.nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=45, std=3.0, channels=3):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.channels = channels
        self.seq = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(kernel_size // 2),
            torch.nn.Conv2d(
                channels, channels, kernel_size,
                stride=1, padding=0, bias=False, groups=channels,
            ),
        )
        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def transpose(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            self.filter = k.unsqueeze(0).unsqueeze(0)
            for _, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            self.filter = k.unsqueeze(0).unsqueeze(0)
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)


class SuperResolutionOperator(torch.nn.Module):
    def __init__(self, in_shape, scale_factor, device):
        super().__init__()
        self.scale_factor = scale_factor
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1 / scale_factor).to(device)

    def forward(self, data):
        return self.down_sample(data)

    def transpose(self, data):
        return self.up_sample(data)


class MotionBlurOperator(torch.nn.Module):
    def __init__(self, kernel_size, intensity, device):
        super().__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.conv = BlurKernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity).to(device)

        self.kernel_obj = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        self.k = torch.tensor(self.kernel_obj.kernelMatrix, dtype=torch.float32, device=self.device)
        self.filter = self.k.unsqueeze(0).unsqueeze(0)
        self.conv.update_weights(self.k)

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.k.view(1, 1, self.kernel_size, self.kernel_size)

    def get_fourier(self, img_dim):
        kernel_padded = torch.zeros((img_dim, img_dim))
        half = self.kernel_size // 2
        kernel_padded[:half+1, :half+1] = self.k[half:, half:]
        kernel_padded[-half:, :half+1] = self.k[:half, half:]
        kernel_padded[:half+1, -half:] = self.k[half:, :half]
        kernel_padded[-half:, -half:] = self.k[:half, :half]
        k_fourier = torch.fft.fft2(kernel_padded, dim=[-1, -2])
        return k_fourier

    def get_fourier_shift(self, img_dim):
        kernel_padded = torch.zeros((img_dim, img_dim))
        kernel_padded[:self.kernel_size, :self.kernel_size] = self.k
        k_fourier = torch.fft.fftshift(torch.fft.fft2(kernel_padded), dim=[-1, -2])
        return k_fourier

    def get_inverse_fourier(self, img_dim):
        f_kernel = self.get_fourier(img_dim)
        return f_kernel ** -1

    def get_inverse_reg_fourier(self, img_dim, epsilon=1e-8):
        f_kernel = self.get_fourier(img_dim)
        f_kernel_mag_sq = torch.abs(f_kernel) ** 2
        stabilized_denominator = f_kernel_mag_sq + epsilon
        inverse_filter_fourier = torch.conj(f_kernel) / stabilized_denominator
        return inverse_filter_fourier


class Decolorize(torch.nn.Module):
    def __init__(self, channels=3, srf='rec601', device='cpu'):
        super().__init__()
        self.device = device
        if srf is None or srf == 'rec601':
            srf = torch.tensor([0.4472 * 0.66851, 0.8781 * 0.66851, 0.1706 * 0.66851])
        elif srf in ('average', 'flat'):
            srf = torch.tensor([1 / channels] * channels)
        elif srf == 'random':
            srf = torch.rand(channels)
            srf = srf / srf.sum()
        elif isinstance(srf, (tuple, list)):
            srf = torch.tensor(srf)
        else:
            raise ValueError('Invalid srf')

        if srf.size(0) < channels:
            srf = torch.cat([srf, torch.zeros(channels - srf.size(0))])
        elif srf.size(0) > channels:
            raise ValueError('srf should be of length equal to or less than channels.')

        assert torch.allclose(srf.sum(), torch.tensor(1.0), rtol=1e-4)

        self.register_buffer('srf', srf.to(device).view(1, channels, 1, 1))
        self.register_buffer('mask', torch.linalg.vector_norm(self.srf, dim=1, keepdim=True))
        self.to(device)

    def forward(self, data, **kwargs):
        if data.shape[1] != self.srf.shape[1]:
            raise ValueError('data should have same number of channels as SRF.')
        return torch.sum(data * self.srf, dim=1, keepdim=True)

    def transpose(self, data, **kwargs):
        if data.shape[1] != 1:
            raise ValueError('data should be grayscale i.e. have length 1 in the 1st dimension.')
        return data.expand(data.shape[0], self.srf.shape[1], *data.shape[2:]) * self.srf

    def H_pinv(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != 1:
            raise ValueError('x should be grayscale i.e. have length 1 in the 1st dimension.')
        return x.expand(x.shape[0], self.srf.shape[1], *x.shape[2:]) * self.srf / (self.mask ** 2)


def generate_box_mask(batch_size, channels, height, width, box_size, device):
    mask = torch.ones(batch_size, channels, height, width, device=device)
    start = (height - box_size) // 2
    mask[:, :, start:start + box_size, start:start + box_size] = 0
    return mask


def generate_random_mask(batch_size, channels, height, width, mask_ratio=0.5, device='cuda'):
    """
    Generate random pixel masks for inpainting.

    Args:
        mask_ratio: proportion of pixels to mask

    Returns:
        mask: [B, C, H, W], 1 = observed, 0 = masked
    """
    mask = torch.rand(batch_size, channels, height, width, device=device)
    mask = (mask > mask_ratio).float()
    return mask


#@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(torch.nn.Module):
    # def __init__(self, oversample, device, resolution=128):
    #     super().__init__()
    #     self.pad = int((oversample / 8.0) * resolution)
    #     self.device = device
        
    # def forward(self, data, **kwargs):
    #     data = 0.5 * data + 0.5
    #     padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
    #     # print(data)
    #     amplitude = fft2_m(padded).abs()
    #     return amplitude
    def __init__(self, oversample, device, resolution=128):
        super().__init__()
        self.pad = int((oversample / 8.0) * resolution)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude
    
    def H_pinv(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = ifft2_m(x).abs()
        x = self.undo_padding(x, self.pad, self.pad, self.pad, self.pad)
        return x
    
    def undo_padding(self, tensor, pad_left, pad_right, pad_top, pad_bottom):
        # Assuming 'tensor' is the 4D tensor
        # 'pad_left', 'pad_right', 'pad_top', 'pad_bottom' are the padding values
        if tensor.dim() != 4:
            raise ValueError("Input tensor should have 4 dimensions.")
        return tensor[:, :, pad_top : -pad_bottom, pad_left : -pad_right]
        

# ---------------------------------------------------------------------------
# JPEG compression
# ---------------------------------------------------------------------------

def torch_rgb2ycbcr(x):
    v = torch.tensor(
        [[0.299, 0.587, 0.114],
         [-0.1687, -0.3313, 0.5],
         [0.5, -0.4187, -0.0813]]
    ).to(x.device)
    ycbcr = torch.tensordot(x, v, dims=([1], [1])).transpose(3, 2).transpose(2, 1)
    ycbcr[:, 1:] += 128
    return ycbcr


def torch_ycbcr2rgb(x):
    v = torch.tensor(
        [[1.00000000e+00, -3.68199903e-05, 1.40198758e+00],
         [1.00000000e+00, -3.44113281e-01, -7.14103821e-01],
         [1.00000000e+00, 1.77197812e+00, -1.34583413e-04]]
    ).to(x.device)
    x[:, 1:] -= 128
    rgb = torch.tensordot(x, v, dims=([1], [1])).transpose(3, 2).transpose(2, 1)
    return rgb


def chroma_subsample(x):
    return x[:, 0:1, :, :], x[:, 1:, ::2, ::2]


def general_quant_matrix(qf=10):
    q1 = torch.tensor([
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ])
    q2 = torch.tensor([
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    ])
    s = (5000 / qf) if qf < 50 else (200 - 2 * qf)
    q1 = torch.floor((s * q1 + 50) / 100)
    q1[q1 <= 0] = 1
    q1[q1 > 255] = 255
    q2 = torch.floor((s * q2 + 50) / 100)
    q2[q2 <= 0] = 1
    q2[q2 > 255] = 255
    return q1, q2


def quantization_matrix(qf):
    return general_quant_matrix(qf)


def jpeg_encode(x, qf):
    """Encode a batch of images in [-1, 1] (N, 3, H, W) into JPEG luma/chroma blocks."""
    x = (x + 1) / 2 * 255
    n_batch, _, n_size, _ = x.shape

    x = torch_rgb2ycbcr(x)
    x_luma, x_chroma = chroma_subsample(x)
    unfold = torch.nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    x_luma = unfold(x_luma).transpose(2, 1)
    x_chroma = unfold(x_chroma).transpose(2, 1)

    x_luma = x_luma.reshape(-1, 8, 8) - 128
    x_chroma = x_chroma.reshape(-1, 8, 8) - 128

    dct_layer = LinearDCT(8, 'dct', norm='ortho').to(x_luma.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)

    x_luma = x_luma.view(-1, 1, 8, 8)
    x_chroma = x_chroma.view(-1, 2, 8, 8)

    q1, q2 = quantization_matrix(qf)
    q1 = q1.to(x_luma.device)
    q2 = q2.to(x_luma.device)
    x_luma /= q1.view(1, 8, 8)
    x_chroma /= q2.view(1, 8, 8)

    x_luma = x_luma.round()
    x_chroma = x_chroma.round()

    x_luma = x_luma.reshape(n_batch, (n_size // 8) ** 2, 64).transpose(2, 1)
    x_chroma = x_chroma.reshape(n_batch, (n_size // 16) ** 2, 64 * 2).transpose(2, 1)

    fold = torch.nn.Fold(output_size=(n_size, n_size), kernel_size=(8, 8), stride=(8, 8))
    x_luma = fold(x_luma)
    fold = torch.nn.Fold(output_size=(n_size // 2, n_size // 2), kernel_size=(8, 8), stride=(8, 8))
    x_chroma = fold(x_chroma)

    return [x_luma, x_chroma]


def jpeg_decode(x, qf):
    x_luma, x_chroma = x
    n_batch, _, n_size, _ = x_luma.shape
    unfold = torch.nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    x_luma = unfold(x_luma).transpose(2, 1)
    x_luma = x_luma.reshape(-1, 1, 8, 8)
    x_chroma = unfold(x_chroma).transpose(2, 1)
    x_chroma = x_chroma.reshape(-1, 2, 8, 8)

    q1, q2 = quantization_matrix(qf)
    q1 = q1.to(x_luma.device)
    q2 = q2.to(x_luma.device)
    x_luma *= q1.view(1, 8, 8)
    x_chroma *= q2.view(1, 8, 8)

    x_luma = x_luma.reshape(-1, 8, 8)
    x_chroma = x_chroma.reshape(-1, 8, 8)

    dct_layer = LinearDCT(8, 'idct', norm='ortho').to(x_luma.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)

    x_luma = (x_luma + 128).reshape(n_batch, (n_size // 8) ** 2, 64).transpose(2, 1)
    x_chroma = (x_chroma + 128).reshape(n_batch, (n_size // 16) ** 2, 64 * 2).transpose(2, 1)

    fold = torch.nn.Fold(output_size=(n_size, n_size), kernel_size=(8, 8), stride=(8, 8))
    x_luma = fold(x_luma)
    fold = torch.nn.Fold(output_size=(n_size // 2, n_size // 2), kernel_size=(8, 8), stride=(8, 8))
    x_chroma = fold(x_chroma)

    x_chroma_repeated = torch.zeros(n_batch, 2, n_size, n_size, device=x_luma.device)
    x_chroma_repeated[:, :, 0::2, 0::2] = x_chroma
    x_chroma_repeated[:, :, 0::2, 1::2] = x_chroma
    x_chroma_repeated[:, :, 1::2, 0::2] = x_chroma
    x_chroma_repeated[:, :, 1::2, 1::2] = x_chroma

    x = torch.cat([x_luma, x_chroma_repeated], dim=1)
    x = torch_ycbcr2rgb(x)
    x = x / 255 * 2 - 1
    return x


class JPEGOperator(torch.nn.Module):
    """Differentiable JPEG compress + decompress; treated as a self-mapping degradation.

    Image dimension must be divisible by 16 (chroma subsampling + 8x8 blocks).
    Treated as approximately self-adjoint.
    """

    def __init__(self, qf=10, device='cpu'):
        super().__init__()
        self.qf = qf
        self.device = device
        self.to(device)

    def forward(self, data, **kwargs):
        encoded = jpeg_encode(data, self.qf)
        return jpeg_decode(encoded, self.qf)

    def transpose(self, data, **kwargs):
        return self.forward(data)


class DiffJPEGOperator(torch.nn.Module):
    """Differentiable JPEG (Reich et al. 2024, necla-ml/Diff-JPEG).

    Wraps `diff_jpeg.diff_jpeg_coding` with [-1, 1] ↔ [0, 255] conversion.
    Use this inside autograd-driven loops (e.g. PnP data fidelity).

    Args:
        qf: JPEG quality factor in [1, 99] (lower = more compression).
        ste: if True, use the straight-through-estimator variant. Default
             False uses the polynomial soft-rounding (more principled).
    """

    def __init__(self, qf=10, ste=False, device='cpu'):
        super().__init__()
        from utils.diff_jpeg import DiffJPEGCoding
        assert 1 <= qf <= 99, "Diff-JPEG requires qf in [1, 99]"
        self.qf = qf
        self.ste = ste
        self.device = device
        self.coder = DiffJPEGCoding(ste=ste)
        self.to(device)

    def forward(self, data, **kwargs):
        x = (data + 1) / 2 * 255
        jq = torch.full((x.shape[0],), float(self.qf), device=x.device, dtype=x.dtype)
        y = self.coder(x, jq)
        return y / 255 * 2 - 1

    def transpose(self, data, **kwargs):
        return self.forward(data)
