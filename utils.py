import torch

def save_checkpoint(state, path):
    torch.save(state, path)

def load_checkpoint(path, map_location=None):
    import torch
    return torch.load(path, map_location=map_location)
