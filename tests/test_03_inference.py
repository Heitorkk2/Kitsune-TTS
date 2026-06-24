import os
import sys
import json
import time
import torch
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitsune.model.synthesizer import SynthesizerTrn

def test_inference():
    print("\n--- 3. Inference RTF Test ---")
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "kitsune_base.json")
    with open(config_path, "r") as f:
        hparams = json.load(f)
        
    vocab_size = 85
    net_g = SynthesizerTrn(
        n_vocab=vocab_size,
        spec_channels=hparams["data"]["filter_length"] // 2 + 1,
        segment_size=hparams["train"]["segment_size"] // hparams["data"]["hop_length"],
        **hparams["model"]
    )
    net_g.eval()
    
    # Text tensor (simulate ~10 seconds of speech)
    text_len = 60 
    x = torch.randint(0, vocab_size, (1, text_len)).long()
    x_lengths = torch.LongTensor([text_len])
    sid = torch.LongTensor([0])

    print("Warming up model...")
    try:
        with torch.no_grad():
            net_g.infer(x, x_lengths, sid=sid)
            
        print("Running timed inference...")
        start_time = time.time()
        with torch.no_grad():
            # Inference automatically uses Deterministic DP if use_sdp config is bypassed, 
            # or runs the SDP in reverse.
            audio, attn, y_mask, _ = net_g.infer(x, x_lengths, sid=sid)
        end_time = time.time()
        
        gen_time = end_time - start_time
        audio_duration = audio.size(2) / hparams["data"]["sampling_rate"]
        rtf = gen_time / max(0.001, audio_duration)
        
        print(f"Generated {audio_duration:.2f} seconds of audio in {gen_time:.2f} seconds.")
        print(f"✅ Real-Time Factor (RTF): {rtf:.4f} (Lower is better)")
        if rtf < 1.0:
            print("Speed is excellent for CPU!")
        else:
            print("Speed is acceptable, but could be improved via ONNX.")
    except Exception as e:
        print("❌ Inference test failed!")
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
