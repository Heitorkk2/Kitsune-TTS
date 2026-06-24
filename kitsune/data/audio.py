import torch
import numpy as np
from scipy.io.wavfile import read
import librosa

MAX_WAV_VALUE = 32768.0

_hann_window_cache = {}
_mel_basis_cache = {}


def _device_key(device):
    return device.type, device.index

def load_wav(full_path):
    """Loads a 16-bit PCM wav file and returns audio data and sample rate."""
    sampling_rate, data = read(full_path)
    # Ensure it is float32 between -1.0 and 1.0
    if data.dtype == np.int16:
        data = data.astype(np.float32) / MAX_WAV_VALUE
    elif data.dtype == np.float32:
        pass
    else:
        raise ValueError(f"Unsupported WAV format: {data.dtype}")
    return data, sampling_rate

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    """Extracts linear spectrogram for the posterior encoder."""
    if torch.min(y) < -1.1 or torch.max(y) > 1.1:
        print(f"Warning: Audio might be clipped (min={torch.min(y)}, max={torch.max(y)})")

    window_key = (win_size, _device_key(y.device), y.dtype)
    hann_window = _hann_window_cache.get(window_key)
    if hann_window is None:
        hann_window = torch.hann_window(win_size, device=y.device, dtype=y.dtype)
        _hann_window_cache[window_key] = hann_window
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    return spec

def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """Extracts Mel spectrogram for the training reconstruction loss."""
    # Compute linear spec
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)
    
    basis_key = (
        sampling_rate, n_fft, num_mels, float(fmin), float(fmax),
        _device_key(spec.device), spec.dtype,
    )
    mel_basis = _mel_basis_cache.get(basis_key)
    if mel_basis is None:
        mel_array = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        mel_basis = torch.from_numpy(mel_array).to(device=spec.device, dtype=spec.dtype)
        _mel_basis_cache[basis_key] = mel_basis
    
    # Apply Mel basis
    mel = torch.matmul(mel_basis, spec)
    mel = spectral_normalize_torch(mel)
    return mel
