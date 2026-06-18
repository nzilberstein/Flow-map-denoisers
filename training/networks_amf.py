# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu, gelu

import pdb
#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.
def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x
    
#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.
# class AttentionOp(torch.autograd.Function):
#     @staticmethod
#     def forward(q, k):
#         w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
#         return w
    
#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         q, k = inputs
#         w = output
#         ctx.save_for_backward(q, k, w)
    
#     @staticmethod
#     def backward(ctx, dw):
#         q, k, w = ctx.saved_tensors
#         db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
#         dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
#         dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
#         return dq, dk
    
#     @staticmethod
#     def jvp(ctx, dq, dk):
#         """
#         Forward-mode autodiff (JVP).
        
#         Args:
#             ctx: context with saved tensors
#             dq: tangent vector for q
#             dk: tangent vector for k
        
#         Returns:
#             dw: tangent vector for output w
#         """
#         q, k, w = ctx.saved_tensors
        
#         # Compute JVP for the attention operation
#         # d/dt [softmax(q @ k^T / sqrt(d))] where q(t) = q + t*dq, k(t) = k + t*dk
        
#         # First compute derivative of the logits (before softmax)
#         # logits = q @ k^T / sqrt(d)
#         # d_logits = (dq @ k^T + q @ dk^T) / sqrt(d)
#         scale = 1.0 / np.sqrt(k.shape[1])
#         d_logits = torch.einsum('ncq,nck->nqk', dq.to(torch.float32), k.to(torch.float32)) * scale
#         d_logits += torch.einsum('ncq,nck->nqk', q.to(torch.float32), dk.to(torch.float32)) * scale
        
#         # Then compute derivative through softmax
#         # d_softmax = softmax * (d_logits - sum(softmax * d_logits, keepdim=True))
#         w_float = w.to(torch.float32)
#         dw = w_float * (d_logits - (w_float * d_logits).sum(dim=2, keepdim=True))
        
#         return dw.to(q.dtype)

def attention_op(q, k):
    """Regular function version - PyTorch handles all autodiff automatically"""
    w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
    return w

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
        anisotropic_noise=False, bin_spectrum=False, homogeneous_norm=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.anisotropic_noise = anisotropic_noise
        self.bin_spectrum = bin_spectrum
        self.homogeneous_norm = homogeneous_norm

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        if self.anisotropic_noise:
            self.affine = torch.nn.Conv2d(in_channels=emb_channels, out_channels=out_channels*(2 if adaptive_scale else 1), kernel_size=1)
        else:
            if self.bin_spectrum:
                self.affine = Linear(in_features=emb_channels * 15, out_features=out_channels*(2 if adaptive_scale else 1), **init)    
            else:
                self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)

        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))
        params = self.affine(emb)

        if params.ndim == 2:
            # Scalar embedding (Linear affine): reshape to [B, C, 1, 1] for broadcasting
            params = params.unsqueeze(-1).unsqueeze(-1)
        elif params.shape[-2:] != x.shape[-2:]:
            params = torch.nn.functional.interpolate(
                params,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        # -- #
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x) * (1 + params))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            # q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            # w = AttentionOp.apply(q, k)
            # a = torch.einsum('nqk,nck->ncq', w, v)
            # x = self.proj(a.reshape(*x.shape)).add_(x)
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = attention_op(q, k)  # Use regular function instead of AttentionOp.apply
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

@persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128, #64 or #128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2], #[1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 2, #1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,3,3,1], #[1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        anisotropic_noise   = True,         # True: per-pixel spatial embedding; False: scalar embedding via PositionalEmbedding
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        self.anisotropic_noise = anisotropic_noise
        self.bin_spectrum = False #TODO: this is hardcoded. Need to be changed!
        self.homogeneous_norm = False
        self.adaptive_scale = False # The original is false
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=self.adaptive_scale,
            init=init, init_zero=init_zero, init_attn=init_attn, anisotropic_noise = self.anisotropic_noise,
            bin_spectrum = self.bin_spectrum, homogeneous_norm = self.homogeneous_norm
        )  
        # Mapping.
        if self.anisotropic_noise:
            self.map_noise_t = AnisotropicGammaEmbeddingNet(fourier_dim=noise_channels, t_min=1e-9, t_max=1e3,
                                                          base_channels=model_channels, num_res_blocks=4, #4 for large
                                                          out_embedding_channels=emb_channels, embedding_type=embedding_type)
            self.map_noise_r = AnisotropicGammaEmbeddingNet(fourier_dim=noise_channels, t_min=1e-9, t_max=1e3,
                                                          base_channels=model_channels, num_res_blocks=4, #4 for large
                                                          out_embedding_channels=emb_channels, embedding_type=embedding_type)
        else:
            self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        if self.bin_spectrum:
            self.map_layer0 = Linear(in_features=noise_channels * 15, out_features=emb_channels * 15, **init)
            self.map_layer1 = Linear(in_features=emb_channels * 15, out_features=emb_channels * 15, **init)
        else:
            self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels_t, noise_labels_r, covariance=None, class_labels=None, augment_labels=None):
        # Mapping.
        if self.anisotropic_noise:
            emb_t = self.map_noise_t(noise_labels_t)
            emb_r = self.map_noise_r(noise_labels_r)
            emb = silu(emb_t + emb_r)
        else:
            # Scalar conditioning: collapse to [B] for PositionalEmbedding, then apply MLP
            t_scalar = noise_labels_t.reshape(x.shape[0], -1).mean(dim=1)
            r_scalar = noise_labels_r.reshape(x.shape[0], -1).mean(dim=1)
            # emb_t = self.map_noise(t_scalar)
            emb_t = self.map_noise(t_scalar - r_scalar) 
            emb_r = self.map_noise(r_scalar)
            emb = silu(self.map_layer0(emb_t + emb_r))
            emb = silu(self.map_layer1(emb))
        
        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux
    

#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * 2
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero, anisotropic_noise=True)
        self.anisotropic_noise = True #TODO: this is hardcoded. Need to be changed!

        # Mapping.
    
        if self.anisotropic_noise:
            self.map_noise = AnisotropicGammaEmbeddingNet(fourier_dim=noise_channels, t_min=1e-9, t_max=1e3,
                                                          base_channels=model_channels, num_res_blocks=3,
                                                          out_embedding_channels=emb_channels, embedding_type=embedding_type)
        else:
            self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
            self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
            self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        # Mapping.
        if self.anisotropic_noise:
            emb = self.map_noise(noise_labels)
        else:
            if self.map_augment is not None and augment_labels is not None:
                emb = emb + self.map_augment(augment_labels)
            emb = silu(self.map_layer0(emb))
            emb = self.map_layer1(emb)
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
                emb = emb + self.map_label(tmp)
            emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x



#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VEPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        sigma_min       = 0.02,         # Minimum supported noise level.
        sigma_max       = 100,          # Maximum supported noise level.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".

@persistence.persistent_class
class iDDPMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels*2, label_dim=label_dim, **model_kwargs)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, :self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

# @persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# @persistence.persistent_class
# class AMFPrecond(torch.nn.Module):
#     """Anisotropic MeanFlow preconditioning wrapper.

#     Thin identity wrapper around networks_afm.SongUNet that exposes the
#     same interface (img_resolution, img_channels, label_dim) expected by
#     the training loop, while passing (x, alpha_t, alpha_r) straight
#     through to the underlying model without any preconditioning.
#     """
#     def __init__(self,
#         img_resolution,
#         img_channels,
#         label_dim       = 0,
#         use_fp16        = False,
#         model_type      = 'SongUNet',
#         **model_kwargs,
#     ):
#         super().__init__()
#         self.img_resolution = img_resolution
#         self.img_channels = img_channels
#         self.label_dim = label_dim
#         self.use_fp16 = use_fp16

#         self.model = SongUNet(
#             img_resolution=img_resolution,
#             in_channels=img_channels,
#             out_channels=img_channels,
#             label_dim=label_dim,
#             **model_kwargs,
#         )

#     def forward(self, x, alpha_t, alpha_r, **kwargs):
#         dtype = torch.float16 if (self.use_fp16 and x.device.type == 'cuda') else torch.float32
#         return self.model(x.to(dtype), alpha_t.to(dtype), alpha_r.to(dtype), **kwargs).to(torch.float32)

@persistence.persistent_class
class AMFPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,
        img_channels,
        label_dim       = 0,
        use_fp16        = False,
        use_weight      = False,
        model_type      = 'SongUNet',
        model_channels      = 128, #64 or #128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2], #[1,2,2,2],    # Per-resolution multipliers for the number of channels.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability.
        **model_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.use_weight = use_weight

        self.model = SongUNet(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            **model_kwargs,
        )

        if use_weight:
            self.weight_net = AlphaWeightNet()

    def calc_weight(self, alpha_r, alpha_t):
        if not self.use_weight:
            return torch.zeros(alpha_r.shape[0], 1, 1, 1, device=alpha_r.device)
        return self.weight_net(alpha_r, alpha_t)  # logvar [B, 1, 1, 1]

    def forward(self, x, alpha_r, alpha_t, return_weight=False, **kwargs):
        dtype = torch.float16 if (self.use_fp16 and x.device.type == 'cuda') else torch.float32
        out = self.model(x.to(dtype), alpha_t.to(dtype), alpha_r.to(dtype), **kwargs).to(torch.float32)

        if return_weight:
            logvar = self.calc_weight(alpha_r, alpha_t)
            return out, logvar

        return out


#----------------------------------------------------------------------------
# AMF preconditioning using ImageTransformerDenoiserModelV2 as backbone.
#
# Interface is identical to AMFPrecond: forward(x, alpha_r, alpha_t)
#   alpha_t → sigma  (target time, shape [B,1,1,1] → [B])
#   alpha_r → mapping_cond  (source time, shape [B,1,1,1] → [B,1])
#            injected via mapping_cond_dim=1 in the mapping network.
#
# Default architecture (edit levels/mapping to resize the model):
#   Level 0 : depth=2, width=384, ShiftedWindow(window=8, d_head=64)
#   Level 1 : depth=4, width=768, ShiftedWindow(window=8, d_head=64)
#   Level 2  (mid): depth=4, width=768, Global(d_head=64)
#   Mapping : depth=2, width=768
#
# patch_size=(2,2) halves spatial dims on input/output, so for 192×192
# images the transformer sees 96×96 tokens at level 0.

# @persistence.persistent_class
# class ITv2AMFPrecond(torch.nn.Module):
#     def __init__(
#         self,
#         img_resolution,
#         img_channels,
#         label_dim       = 0,
#         use_fp16        = False,
#         use_weight      = False,
#         patch_size      = (4, 4),
#         levels          = None,   # list of LevelSpec; None → default below
#         mapping         = None,   # MappingSpec;       None → default below
#         **model_kwargs,
#     ):
#         super().__init__()
#         self.img_resolution = img_resolution
#         self.img_channels   = img_channels
#         self.label_dim      = label_dim
#         self.use_fp16       = use_fp16
#         self.use_weight     = use_weight

#         from training.k_diffusion.image_transformer_v2 import (
#             ImageTransformerDenoiserModelV2,
#             LevelSpec, MappingSpec,
#             GlobalAttentionSpec, ShiftedWindowAttentionSpec,
#         )
#         # try:
#         #     from training.k_diffusion.image_transformer_v2 import NeighborhoodAttentionSpec
#         #     _has_natten = True
#         # except (ImportError, ModuleNotFoundError):
#         #     _has_natten = False

#         # if levels is None:
#         #     levels = [
#         #         LevelSpec(depth=2, width=384, d_ff=768,
#         #                   self_attn=ShiftedWindowAttentionSpec(d_head=64, window_size=8),
#         #                   dropout=0.0),
#         #         LevelSpec(depth=4, width=768, d_ff=1536,
#         #                   self_attn=ShiftedWindowAttentionSpec(d_head=64, window_size=8),
#         #                   dropout=0.0),
#         #         LevelSpec(depth=4, width=768, d_ff=1536,
#         #                   self_attn=GlobalAttentionSpec(d_head=64),
#         #                   dropout=0.0),
#         #     ]
#         # if mapping is None:
#         #     mapping = MappingSpec(depth=2, width=768, d_ff=1536, dropout=0.0)

#         if levels is None:
#             # ImageNet-256 config from Table 1 of the HDiT paper:
#             #   Levels: 1 local (NeighborhoodAttn, kernel=7) + 1 global
#             #   Depth:  [2, 11], Widths: [128, 256], d_head=64
#             #   Mapping: depth=1, width=768
#             # Falls back to ShiftedWindowAttentionSpec if natten is not installed.
#             # local_attn = (NeighborhoodAttentionSpec(d_head=64, kernel_size=7) if _has_natten
#             #               else ShiftedWindowAttentionSpec(d_head=64, window_size=7))
#             # window_size=7 from the paper requires natten (no divisibility constraint).
#             # ShiftedWindowAttentionSpec needs spatial_dim % window_size == 0.
#             # With patch_size=4: 128px → 32 tokens, 256px → 64 tokens — both divisible by 8.
#             local_attn = ShiftedWindowAttentionSpec(d_head=64, window_size=8)
            
#             levels = [
#                 LevelSpec(depth=2, width=128, d_ff=256,
#                         self_attn=local_attn, dropout=0.0),
#                 LevelSpec(depth=11, width=256, d_ff=512,
#                         self_attn=GlobalAttentionSpec(d_head=64), dropout=0.0),
#             ]
#         if mapping is None:
#             mapping = MappingSpec(depth=1, width=768, d_ff=1536, dropout=0.0)

#         self.model = ImageTransformerDenoiserModelV2(
#             levels=levels,
#             mapping=mapping,
#             in_channels=img_channels,
#             out_channels=img_channels,
#             patch_size=patch_size,
#             num_classes=label_dim if label_dim > 0 else 0,
#         )

#     def forward(self, x, alpha_r, alpha_t, return_weight=False, **kwargs):
#         dtype = torch.float16 if (self.use_fp16 and x.device.type == 'cuda') else torch.float32
#         x = x.to(dtype)
#         B = x.shape[0]

#         # Both alpha_t and alpha_r go through identical log-scale Fourier embeddings
#         # inside ImageTransformerDenoiserModelV2 (time_emb/time_emb_r paths).
#         sigma   = alpha_t.reshape(B).clamp(min=1e-6)
#         sigma_r = alpha_r.reshape(B).clamp(min=1e-6)

#         out = self.model(x, sigma, sigma_r=sigma_r).to(torch.float32)
#         return out


#----------------------------------------------------------------------------
# Embedding network for anisotropic noise

@persistence.persistent_class
class ConvResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == 64:
            group_norm_channels = 8
        elif in_channels == 128:
            group_norm_channels = 16
        self.norm1 = torch.nn.GroupNorm(group_norm_channels, out_channels) # Use GroupNorm for stability
        self.silu = torch.nn.SiLU()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(group_norm_channels, out_channels)

        self.shortcut = torch.nn.Identity()
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.silu(h + self.shortcut(x))


# --- 3. Anisotropic Gamma Embedding Network ---
@persistence.persistent_class
class AnisotropicGammaEmbeddingNet(torch.nn.Module):
    def __init__(self, fourier_dim, t_min, t_max, base_channels, num_res_blocks, out_embedding_channels, embedding_type=None):
        super().__init__()
        self.fourier_dim = fourier_dim
        # self.t_min = 1e-9 #t_min
        # self.t_max = 1e3 #t_max

        self.initial_conv = torch.nn.Conv2d(self.fourier_dim, base_channels, kernel_size=3, padding=1)

        self.res_blocks = torch.nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ConvResBlock(base_channels, base_channels))

        self.final_conv = torch.nn.Conv2d(base_channels, out_embedding_channels, kernel_size=1)
        self.embedding_type = embedding_type

        if self.embedding_type == 'positional':
            self.fourier_embedding = PositionalEmbedding(num_channels=self.fourier_dim, endpoint=True)
            # self.embedding = PositionalEmbedding(num_channels=self.fourier_dim, endpoint=True)
        elif self.embedding_type == 'fourier':
            self.embedding = FourierEmbedding(num_channels=self.fourier_dim)

        # self.res_blocks = UNetBlock_embedding(in_channels=self.fourier_dim, out_channels=out_embedding_channels, attention=True, emb_channels=None, num_heads=1, dropout=0.0, skip_scale=1.0, eps=1e-6,
        #                             resample_filter=[1,3,3,1], resample_proj=False, adaptive_scale=False,
        #                             init=dict(init_mode='xavier_uniform'), init_zero=dict(init_mode='xavier_uniform', init_weight=1e-5), init_attn=None, anisotropic_noise=False, bin_spectrum=False)

    def forward(self, t_map):
        # pe_gamma_map: [B, D, H, W] (D is dim from positional encoding)
        b, c, h, w = t_map.shape

        # --- Step 1: Pixel-wise Sinusoidal Embedding ---
        # pdb.set_trace()
        if self.embedding_type == 'fourier':
            t_map_flat = t_map.reshape(-1) #[B*H*W]
            sin_embedding = self.embedding(t_map_flat) # [B*H*W, 64]
        elif self.embedding_type == 'positional':
            # a. Reshape the input tensor to combine batch and spatial dimensions.
            t_map_flat = t_map.reshape(-1)
            sin_embedding = self.fourier_embedding(t_map_flat)
        # d. Reshape the result back into a spatial map.
        sin_map = sin_embedding.view(b, h, w, self.fourier_dim).permute(0, 3, 1, 2) # [B, 64, H, W]

        x = self.initial_conv(sin_map)
        for block in self.res_blocks:
            x = block(x)
        gamma_embedding = self.final_conv(x) # [B, out_embedding_channels, H, W]

        # x = self.res_blocks(sin_map)
        # gamma_embedding = x

        return gamma_embedding



@persistence.persistent_class
class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = normalize(self.weight.to(torch.float32))
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

@persistence.persistent_class
class AlphaWeightNet(torch.nn.Module):
    """Magnitude-preserving weight network for anisotropic flow map."""
    def __init__(self, hidden_channels=128):
        super().__init__()
        # alpha_r and alpha_t are each [B, 1, H, W], concatenated → 2 channels
        self.conv1 = MPConv(2, hidden_channels, kernel=(3, 3))
        self.conv2 = MPConv(hidden_channels, hidden_channels, kernel=(3, 3))
        self.conv3 = MPConv(hidden_channels, hidden_channels, kernel=(3, 3))
        self.linear = MPConv(hidden_channels, 1, kernel=())  # kernel=() → linear

    def forward(self, alpha_r, alpha_t):
        x = torch.cat([alpha_r, alpha_t], dim=1)  # [B, 2, H, W]
        x = torch.nn.functional.silu(self.conv1(x))
        x = torch.nn.functional.silu(self.conv2(x, gain=2**0.5))  # stride via pool
        x = torch.nn.functional.adaptive_avg_pool2d(x, 4)
        x = torch.nn.functional.silu(self.conv3(x, gain=2**0.5))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)  # [B, hidden, 1, 1]
        x = x.flatten(1)                                    # [B, hidden]
        return self.linear(x).reshape(-1, 1, 1, 1)         # [B, 1, 1, 1]
    
# @persistence.persistent_class
# class AlphaWeightNet(torch.nn.Module):
#     """Convolutional network to compute per-sample scalar weight from alpha maps."""
#     def __init__(self, hidden_channels=128):
#         super().__init__()
#         self.net = torch.nn.Sequential(
#             # Process alpha_r and alpha_t jointly
#             torch.nn.Conv2d(2, hidden_channels, 3, padding=1),
#             torch.nn.SiLU(),
#             torch.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=2),
#             torch.nn.SiLU(),
#             torch.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=2),
#             torch.nn.SiLU(),
#             torch.nn.AdaptiveAvgPool2d(1),  # [B, hidden, 1, 1]
#             torch.nn.Flatten(),             # [B, hidden]
#             torch.nn.Linear(hidden_channels, 1),  # [B, 1] scalar logvar
#         )

#     def forward(self, alpha_r, alpha_t):
#         # Concatenate along channel dim
#         x = torch.cat([alpha_r, alpha_t], dim=1)  # [B, 2C, H, W]
#         return self.net(x).reshape(-1, 1, 1, 1)   # [B, 1, 1, 1]
