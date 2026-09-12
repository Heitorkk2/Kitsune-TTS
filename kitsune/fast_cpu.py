"""Optional FP32 CPU inference layout; not a training or ONNX architecture."""
import copy

import torch
from torch import nn


class ChannelsLastVocoder(nn.Module):
    """Copy a materialized vocoder into height-one channels-last convolutions."""

    def __init__(self, original):
        super().__init__()
        self.decoder = copy.deepcopy(original)
        # Replacement layers initialize randomly before their weights are copied.
        # Do not change the caller's sampling RNG stream during conversion.
        with torch.random.fork_rng(devices=[]):
            self._convert(self.decoder)
        self.eval()

    @staticmethod
    def _convert(module):
        for name, child in list(module.named_children()):
            if not isinstance(child, (nn.Conv1d, nn.ConvTranspose1d)):
                ChannelsLastVocoder._convert(child)
                continue
            transpose = isinstance(child, nn.ConvTranspose1d)
            cls = nn.ConvTranspose2d if transpose else nn.Conv2d
            kwargs = {"output_padding": (0, child.output_padding[0])} if transpose else {}
            replacement = cls(
                child.in_channels, child.out_channels, (1, child.kernel_size[0]),
                stride=(1, child.stride[0]), padding=(0, child.padding[0]),
                dilation=(1, child.dilation[0]), groups=child.groups,
                bias=child.bias is not None, padding_mode=child.padding_mode,
                device=child.weight.device, dtype=child.weight.dtype, **kwargs,
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight.unsqueeze(2))
                if child.bias is not None:
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement.to(memory_format=torch.channels_last))

    def forward(self, latent, g=None):
        conditioning = None if g is None else g.unsqueeze(2)
        latent = latent.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        return self.decoder(latent, g=conditioning).squeeze(2)
