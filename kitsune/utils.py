import os
import json
import torch
import logging

def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Avoid adding duplicate handlers on re-init
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    if log_file:
        # Check if a file handler for this path already exists
        existing = [h for h in logger.handlers if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)]
        if not existing:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger

def save_checkpoint(model_g, model_d, optim_g, optim_d, step, epoch, model_dir, tag=None):
    """
    Saves a unified checkpoint containing both Generator and Discriminator.
    Also updates train_state.json with the latest training progress.
    
    Args:
        model_g: Generator model
        model_d: Discriminator model
        optim_g: Generator optimizer
        optim_d: Discriminator optimizer
        step: Current global step
        epoch: Current epoch
        model_dir: Output directory
        tag: Optional tag (e.g. "latest"). If None, uses step number.
    """
    if tag:
        filename = f"checkpoint_{tag}.pth"
    else:
        filename = f"checkpoint_step{step}.pth"
    
    filepath = os.path.join(model_dir, filename)
    
    torch.save({
        'step': step,
        'epoch': epoch,
        'generator': model_g.state_dict(),
        'discriminator': model_d.state_dict(),
        'optimizer_g': optim_g.state_dict(),
        'optimizer_d': optim_d.state_dict(),
    }, filepath)
    
    logging.info(f"Saved checkpoint: {filename} (step {step}, epoch {epoch})")
    return filepath

def update_train_state(model_dir, step, epoch, losses=None):
    """
    Updates train_state.json with the latest training progress.
    This file is human-readable and lets you know exactly where training left off.
    """
    state_path = os.path.join(model_dir, "train_state.json")
    
    state = {
        "last_step": step,
        "last_epoch": epoch,
    }
    if losses:
        state["last_losses"] = losses
    
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def load_checkpoint(filepath, model_g, model_d=None, optim_g=None, optim_d=None):
    """
    Loads a unified checkpoint. Supports both the new format (generator/discriminator)
    and the legacy format (model_state_dict) for backwards compatibility.
    
    Returns:
        step, epoch
    """
    assert os.path.isfile(filepath), f"Checkpoint not found: {filepath}"
    logging.info(f"Loading checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    # Handle torch.compile wrapper (_orig_mod) so keys match correctly
    if hasattr(model_g, "_orig_mod"):
        model_g = model_g._orig_mod
    if model_d is not None and hasattr(model_d, "_orig_mod"):
        model_d = model_d._orig_mod
        
    def strip_orig_mod(state_dict):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # New unified format
    if 'generator' in checkpoint:
        model_g.load_state_dict(strip_orig_mod(checkpoint['generator']), strict=False)
        if model_d is not None and 'discriminator' in checkpoint:
            model_d.load_state_dict(strip_orig_mod(checkpoint['discriminator']), strict=False)
        if optim_g is not None and 'optimizer_g' in checkpoint:
            optim_g.load_state_dict(checkpoint['optimizer_g'])
        if optim_d is not None and 'optimizer_d' in checkpoint:
            optim_d.load_state_dict(checkpoint['optimizer_d'])
        return checkpoint.get('step', 0), checkpoint.get('epoch', 1)
    
    # VITS-style external / transplant format ({'model': state, 'iteration': ...}).
    # strict=False: the transplant intentionally omits enc_p.emb, emb_g and dp.*.
    elif 'model' in checkpoint:
        missing, unexpected = model_g.load_state_dict(strip_orig_mod(checkpoint['model']), strict=False)
        logging.info(f"Loaded 'model' checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys (expected for transplant).")
        return checkpoint.get('iteration', checkpoint.get('step', 0)), checkpoint.get('epoch', 1)

    # Legacy format (G_*.pth from old code)
    elif 'model_state_dict' in checkpoint:
        model_g.load_state_dict(strip_orig_mod(checkpoint['model_state_dict']), strict=False)
        if model_d is not None and 'discriminator_state_dict' in checkpoint:
            model_d.load_state_dict(strip_orig_mod(checkpoint['discriminator_state_dict']), strict=False)
        return checkpoint.get('iteration', 0), 1
    
    else:
        # Raw state dict
        model_g.load_state_dict(checkpoint)
        return 0, 1

def latest_checkpoint_path(dir_path):
    """Finds the latest checkpoint file in a directory."""
    import glob
    # Look for new format first
    f_list = glob.glob(os.path.join(dir_path, "checkpoint_step*.pth"))
    if not f_list:
        # Fallback to legacy format
        f_list = glob.glob(os.path.join(dir_path, "G_*.pth"))
    
    if not f_list:
        return None
        
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f)))))
    return f_list[-1]
