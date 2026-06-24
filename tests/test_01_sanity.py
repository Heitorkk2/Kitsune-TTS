import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitsune.model.synthesizer import SynthesizerTrn

def test_sanity():
    print("\n--- 1. Sanity Check: Model Shapes & Forward Pass ---")
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "kitsune_base.json")
    with open(config_path, "r") as f:
        hparams = json.load(f)

    # Initialize model
    vocab_size = 85
    net_g = SynthesizerTrn(
        n_vocab=vocab_size,
        spec_channels=hparams["data"]["filter_length"] // 2 + 1,
        segment_size=hparams["train"]["segment_size"] // hparams["data"]["hop_length"],
        **hparams["model"]
    )
    
    # Dummy Tensors
    b = 2
    t_text = 50
    t_audio = 200 # Frames
    
    x = torch.randint(0, vocab_size, (b, t_text)).long()
    x_lengths = torch.LongTensor([t_text, t_text-5])
    
    spec = torch.randn(b, hparams["data"]["filter_length"] // 2 + 1, t_audio)
    spec_lengths = torch.LongTensor([t_audio, t_audio-10])

    sid = torch.LongTensor([0, 1])

    # Forward pass (spec is passed as y). Let exceptions propagate so this
    # script and pytest both report a real failure to the caller.
    o, l_length, *_ = net_g(x, x_lengths, spec, spec_lengths, sid=sid)
    print("Forward pass successful! No shape mismatches.")
    print(f"Output Vocoder Shape: {o.shape}")
    print(f"L_length loss: {l_length.mean().item():.4f}")

if __name__ == "__main__":
    test_sanity()
