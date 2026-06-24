import math
import torch
from torch import nn
from torch.nn import functional as F
from .commons import sequence_mask, convert_pad_shape

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional relative positional encoding
    (VITS-style, window_size=4). Without relative position the attention is
    permutation-equivariant (position-blind); the relative embeddings give it a
    local sense of *where* each phoneme sits, which matters for prosody.

    Conv names (convq/convk/convv/convout) and the emb_rel_k/emb_rel_v shapes
    match the upstream VCTK checkpoint so a transplant can transfer them.
    """
    def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=4, heads_share=True):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.k_channels = channels // n_heads

        self.convq = nn.Conv1d(channels, channels, 1)
        self.convk = nn.Conv1d(channels, channels, 1)
        self.convv = nn.Conv1d(channels, channels, 1)
        self.convout = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

    def forward(self, x, c, attn_mask=None):
        q = self.convq(x)
        k = self.convk(c)
        v = self.convv(c)

        b, _, t_t = q.size()
        t_s = k.size(2)
        q = q.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        k = k.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        v = v.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(q / math.sqrt(self.k_channels), k.transpose(-2, -1))

        if self.window_size is not None:
            assert t_s == t_t, "Relative attention only supports self-attention."
            key_rel = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = torch.matmul(q / math.sqrt(self.k_channels), key_rel.unsqueeze(0).transpose(-2, -1))
            scores = scores + self._relative_position_to_absolute_position(rel_logits)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)

        output = torch.matmul(p_attn, v)

        if self.window_size is not None:
            rel_weights = self._absolute_position_to_relative_position(p_attn)
            value_rel = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + torch.matmul(rel_weights, value_rel.unsqueeze(0))

        output = output.transpose(2, 3).contiguous().view(b, self.channels, t_t)
        return self.convout(output)

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            relative_embeddings = F.pad(
                relative_embeddings, convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]])
            )
        return relative_embeddings[:, slice_start:slice_end]

    def _relative_position_to_absolute_position(self, x):
        # x: [b, h, l, 2*l-1]  ->  [b, h, l, l]
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        return x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]

    def _absolute_position_to_relative_position(self, x):
        # x: [b, h, l, l]  ->  [b, h, l, 2*l-1]
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length * length + length * (length - 1)])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask

class EncoderBlock(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, kernel_size, p_dropout=0.):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.ffn = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        # Pre-norm style
        x_norm = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x + self.drop(self.attn(x_norm, x_norm, attn_mask))
        
        x_norm = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop(self.ffn(x_norm, x_mask))
        return x * x_mask

class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout)
            for _ in range(n_layers)
        ])
        
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
        x = x.transpose(1, 2) # [b, h, t]
        
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        
        for block in self.encoder_blocks:
            x = block(x, x_mask)
            
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        logs = torch.clamp(logs, min=-15.0, max=5.0)
        return x, m, logs, x_mask
