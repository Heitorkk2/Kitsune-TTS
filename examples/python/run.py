#!/usr/bin/env python3
import os
import sys
import urllib.request
import scipy.io.wavfile as wavf

# Ensure kitsune package is importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kitsune.api import KitsuneSynthesizer

def download_file(url, dest):
    """Downloads a file if it doesn't already exist locally."""
    if not os.path.exists(dest):
        print(f"Downloading {os.path.basename(dest)} from HuggingFace...")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print("Download complete!")

def main():
    print("🦊 Starting Kitsune-TTS Python Inference...")
    
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    checkpoint_path = os.path.join(model_dir, "latest_model_fp16.pth")
    config_path = os.path.join(model_dir, "model_config.json")

    # Auto-download FP16 weights and config from HuggingFace
    base_url = "https://huggingface.co/Heitorkk2/Kitsune-TTS-V1/resolve/main"
    download_file(f"{base_url}/latest_model_fp16.pth", checkpoint_path)
    download_file(f"{base_url}/model_config.json", config_path)

    synth = KitsuneSynthesizer(checkpoint=checkpoint_path, config=config_path)
    
    print(f"Available speakers: {synth.list_speakers()}")
    
    text = "Olá! Este é um teste rodando em Python, baixando os pesos fp16 automaticamente."
    speaker = "frieren"
    
    print(f'Synthesizing text: "{text}" (Speaker: {speaker})')
    audio = synth.synthesize(text, speaker=speaker)
    
    output_path = os.path.join(os.path.dirname(__file__), "output.wav")
    wavf.write(output_path, 22050, audio)
    print(f"✅ Audio saved to {output_path}")

if __name__ == "__main__":
    main()
