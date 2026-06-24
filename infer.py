import argparse
import sys
import wave

import numpy as np

from kitsune import KitsuneSynthesizer


def write_wav(path, sample_rate, audio):
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm16.tobytes())


def main():
    parser = argparse.ArgumentParser()
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument('-m', '--model', type=str, help='PyTorch checkpoint path')
    backend.add_argument('--onnx', type=str, help='ONNX model path')
    parser.add_argument('-c', '--config', type=str, default=None, help='JSON config (optional, defaults to config in model dir)')
    parser.add_argument('-t', '--text', type=str, required=True, help='Text to synthesize')
    parser.add_argument('-s', '--speaker', type=str, default='frieren', help='Speaker persona name')
    parser.add_argument('-l', '--lang', type=str, default='pt-br', help='Language code')
    parser.add_argument('-o', '--output', type=str, default='output.wav', help='Output wav path')
    args = parser.parse_args()

    print("Loading Kitsune-TTS VITS2-Slim...")
    
    try:
        voz = KitsuneSynthesizer(
            checkpoint=args.model,
            onnx_path=args.onnx,
            config=args.config,
        )
        
        print(f"Synthesizing text: {args.text}")
        audio = voz.synthesize(args.text, speaker=args.speaker, lang=args.lang)
        
        write_wav(args.output, voz.sample_rate, audio)
        print(f"Audio saved to {args.output}")
        return 0
        
    except Exception as e:
        print(f"Failed to synthesize audio: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
