import math
import torch
from torch import nn

from .text_encoder import TextEncoder
from .posterior_encoder import PosteriorEncoder
from .flow import ResidualCouplingBlock
from .duration_predictor import StochasticDurationPredictor, DeterministicDurationPredictor
from .generator import Generator
from .commons import rand_slice_segments, generate_path, sequence_mask
from .monotonic_align import maximum_path

class SynthesizerTrn(nn.Module):
    """
    VITS2-Slim Synthesizer.
    Combines TextEncoder, PosteriorEncoder, Flow, Duration Predictor, and Generator.
    """
    def __init__(self, n_vocab, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, n_speakers=0, gin_channels=0,
                 use_sdp=True, **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.dec = Generator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = DeterministicDurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def remove_weight_norm(self):
        """Materialize all normalized weights for deployment inference."""
        removed = 0
        for module in self.modules():
            parametrizations = getattr(module, "parametrizations", None)
            if parametrizations is not None and hasattr(parametrizations, "weight"):
                torch.nn.utils.parametrize.remove_parametrizations(
                    module, "weight", leave_parametrized=True
                )
                removed += 1
        return removed

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        # Training forward pass
        # 1. Text Encoder
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        
        # 2. Speaker Embedding
        g = None
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)
            
        # 3. Posterior Encoder
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        
        # 4. Flow Decoder
        z_p = self.flow(z, y_mask, g=g, reverse=False)
        
        # 5. Monotonic Alignment Search (MAS)
        # Find the optimal monotonic text<->frame alignment under the prior,
        # instead of assuming a fixed diagonal. Runs under no_grad.
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)                                    # [b, d, t_x]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_x]
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)            # [b, t_y, t_x]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))                  # [b, t_y, t_x]
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)           # [b, 1, t_x]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4                          # [b, t_y, t_x]

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)             # [b, 1, t_y, t_x]

            # maximum_path expects [b, t_x, t_y] (text tokens x audio frames).
            mask_tx_ty = attn_mask.squeeze(1).transpose(1, 2)                                # [b, t_x, t_y]
            path = maximum_path(neg_cent.transpose(1, 2), mask_tx_ty)                        # [b, t_x, t_y]
            attn = path.transpose(1, 2).unsqueeze(1) * attn_mask                             # [b, 1, t_y, t_x]

        # Project prior from text resolution to audio resolution using alignment
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
            
        # 6. Duration Predictor
        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g, reverse=False)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask)

        # 7. Generator (Vocoder)
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer_latent(self, x, x_lengths, sid=None, noise_scale=0.667, length_scale=1.0):
        """Run the acoustic inference graph and return the decoder inputs.

        Keeping this boundary explicit allows deployment to export and profile
        the acoustic model and vocoder independently without changing weights.
        """
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        
        g = None
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale)
        else:
            logw = self.dp(x, x_mask, g=g)
            
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).unsqueeze(1).to(x_mask.dtype)
        
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        return z * y_mask, g, attn, y_mask, (z, z_p, m_p, logs_p)

    def infer(self, x, x_lengths, sid=None, noise_scale=0.667, length_scale=1.0, max_len=None):
        # Inference forward pass (Exportable to ONNX)
        decoder_input, g, attn, y_mask, latent_stats = self.infer_latent(
            x,
            x_lengths,
            sid=sid,
            noise_scale=noise_scale,
            length_scale=length_scale,
        )
        o = self.dec(decoder_input, g=g)
        return o, attn, y_mask, latent_stats

    def duration_loss(self, x, x_lengths, y, y_lengths, sid=None):
        """Compute validation duration loss without running the vocoder.

        Duration validation still requires the posterior encoder, flow and MAS,
        but decoding the aligned latent into audio is unrelated and accounts for
        most of the model's convolutional work.
        """
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        g = None
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)

        z, _, _, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g, reverse=False)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )
            neg_cent2 = torch.matmul(
                -0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r
            )
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), m_p * s_p_sq_r
            )
            neg_cent4 = torch.sum(
                -0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True
            )
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            path = maximum_path(
                neg_cent.transpose(1, 2),
                attn_mask.squeeze(1).transpose(1, 2),
            )
            attn = path.transpose(1, 2).unsqueeze(1) * attn_mask

        durations = attn.sum(2)
        if self.use_sdp:
            loss = self.dp(x, x_mask, durations, g=g, reverse=False)
            return loss / torch.sum(x_mask)

        target_log_duration = torch.log(durations + 1e-6) * x_mask
        predicted_log_duration = self.dp(x, x_mask, g=g)
        return torch.sum(
            (predicted_log_duration - target_log_duration) ** 2, [1, 2]
        ) / torch.sum(x_mask)

    def voice_walk(self, x, x_lengths, sid_a, sid_b, alpha=0.5, noise_scale=0.667, length_scale=1.0):
        """
        Interpolates between two speaker embeddings (sid_a and sid_b) by a factor of alpha.
        alpha=0.0 -> 100% speaker A
        alpha=1.0 -> 100% speaker B
        """
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        
        g_a = self.emb_g(sid_a).unsqueeze(-1)
        g_b = self.emb_g(sid_b).unsqueeze(-1)
        g = (1.0 - alpha) * g_a + alpha * g_b

        logw = self.dp(x, x_mask, g=g) if not self.use_sdp else self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale)
        
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).unsqueeze(1).to(x_mask.dtype)
        
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask), g=g)
        return o
