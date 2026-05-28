# Get cuda device
import torch

def get_gpu():
    return "cuda" if torch.cuda.is_available() else "cpu"