#!/usr/bin/env python3
import torch
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python export_fp16.py <input_ckpt> <output_dir>")
        sys.exit(1)
        
    in_path = sys.argv[1]
    out_dir = sys.argv[2]
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading {in_path}...")
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    
    # Extract only the generator weights (discarding optimizer states, epoch counts, etc.)
    model_state = ckpt.get("generator", ckpt.get("model", ckpt))
    
    # Strip the torch.compile prefix if the model was compiled
    new_state = {}
    for k, v in model_state.items():
        if k.startswith("_orig_mod."):
            new_state[k.replace("_orig_mod.", "")] = v
        else:
            new_state[k] = v
            
    # Save a pruned FP32 version
    fp32_path = os.path.join(out_dir, "latest_model_fp32.pth")
    print(f"Saving pruned model (FP32) to {fp32_path}...")
    torch.save({"model": new_state}, fp32_path)
    
    # Convert all floating-point tensors to FP16 (halves the file size)
    fp16_state = {k: v.half() if torch.is_floating_point(v) else v for k, v in new_state.items()}
    fp16_path = os.path.join(out_dir, "latest_model_fp16.pth")
    print(f"Saving compacted model (FP16) to {fp16_path}...")
    torch.save({"model": fp16_state}, fp16_path)
    
    print("Success! Checkpoint successfully extracted and converted.")

if __name__ == "__main__":
    main()
