import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# How to use in every code to chose GPU/CPU:

# from core.device import get_device

# device = get_device()
# print(device)