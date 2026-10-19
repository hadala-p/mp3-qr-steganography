import torch

def ber_from_logits(logits: torch.Tensor, target_bits: torch.Tensor, thr: float) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    return (pred != target_bits).float().mean().item()
