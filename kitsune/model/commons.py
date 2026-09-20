import math
import torch
from torch import nn
from torch.nn import functional as F

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def kl_divergence(m_p, logs_p, m_q, logs_q, z_mask):
    """KL divergence between two diagonal Gaussians."""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + (m_p - m_q) ** 2) * torch.exp(-2.0 * logs_q)
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss

def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        max_starts = torch.full(
            (b,), max(t - segment_size + 1, 1), device=x.device
        )
    else:
        max_starts = torch.clamp(x_lengths - segment_size + 1, min=1)

    starts = (torch.rand([b], device=x.device) * max_starts).to(dtype=torch.long)
    return _gather_segments(x, starts, segment_size), starts

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path

def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

def _gather_segments(x, starts, segment_size):
    b, d, t = x.size()
    if t == 0:
        return x.new_zeros((b, d, segment_size))

    offsets = torch.arange(segment_size, device=x.device).view(1, 1, -1)
    indices = starts.to(device=x.device, dtype=torch.long).view(b, 1, 1) + offsets
    valid = (indices >= 0) & (indices < t)
    indices = indices.clamp(min=0, max=t - 1).expand(b, d, segment_size)
    segments = torch.gather(x, 2, indices)
    return segments * valid.expand_as(segments).to(dtype=x.dtype)


def slice_segments(x, ids_str, segment_size=4):
    return _gather_segments(x, ids_str, segment_size)
