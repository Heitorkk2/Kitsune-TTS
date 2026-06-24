import os
import sys
import json
import torch
import traceback
from torch.optim import AdamW
from torch.amp import autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitsune.model.synthesizer import SynthesizerTrn
from kitsune.data.audio import load_wav, spectrogram_torch, mel_spectrogram_torch

def test_overfit():
    print("\n--- 2. Overfit Test: Single Batch Loop ---")
    root_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(root_dir, "configs", "kitsune_base.json")
    with open(config_path, "r") as f:
        hparams = json.load(f)
        
    vocab_size = 85
    net_g = SynthesizerTrn(
        n_vocab=vocab_size,
        spec_channels=hparams["data"]["filter_length"] // 2 + 1,
        segment_size=hparams["train"]["segment_size"] // hparams["data"]["hop_length"],
        **hparams["model"]
    )
    optim_g = AdamW(net_g.parameters(), lr=1e-3)
    
    # Mixed precision logic
    precision = hparams["train"].get("precision", "fp32")
    amp_enabled = precision in ["bf16", "fp16"] and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)

    # Try loading real audio, else dummy
    wav_path = os.path.join(root_dir, "Yunnavoice1 (1).wav")
    if os.path.exists(wav_path):
        print(f"Loaded real audio: {wav_path}")
        audio, sr = load_wav(wav_path)
        # Resample logic omitted for brevity; assuming it's roughly 22050Hz for test
        audio = torch.FloatTensor(audio).unsqueeze(0)
    else:
        print("WAV file not found, using dummy tensor...")
        audio = torch.randn(1, 22050 * 2) # 2 seconds
        
    # Get Spectrograms
    spec = spectrogram_torch(
        audio, 
        hparams["data"]["filter_length"], 
        hparams["data"]["hop_length"], 
        hparams["data"]["win_length"], 
        center=False
    )
    
    # Dummy Text (e.g. "Olá, Heitor. Tudo bem com você hoje?")
    text_len = 25
    x = torch.randint(0, vocab_size, (1, text_len)).long()
    x_lengths = torch.LongTensor([text_len])
    spec_lengths = torch.LongTensor([spec.size(2)])
    sid = torch.LongTensor([0])

    print("Starting 50 iterations overfit loop (Generator only)...")
    try:
        for i in range(50):
            optim_g.zero_grad()
            
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=amp_enabled, dtype=amp_dtype):
                o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                    x, x_lengths, spec, spec_lengths, sid=sid
                )
                
                # Match output shapes (slice from original audio to match generator's segment output)
                audio_slice = audio[:, ids_slice.item() * hparams["data"]["hop_length"] : (ids_slice.item() * hparams["data"]["hop_length"]) + o.size(2)]
                if audio_slice.size(1) < o.size(2):
                    # Edge case where random slice is too close to end
                    o = o[:, :, :audio_slice.size(1)]
                
                loss_mel = torch.nn.functional.mse_loss(o.squeeze(1), audio_slice)
                loss_all = loss_mel + l_length.mean()
                
            loss_all.backward()
            optim_g.step()
            
            if (i+1) % 10 == 0:
                print(f"Step {i+1:02d} - Total Loss: {loss_all.item():.4f}")
                
        print("✅ Overfit test complete. Loss dropped successfully.")
    except Exception as e:
        print("❌ Overfit test failed!")
        traceback.print_exc()

if __name__ == "__main__":
    test_overfit()
