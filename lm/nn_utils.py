import torch
import math

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # prevents overflow in exp
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def cross_entropy(logits, targets):
    shifted = logits - logits.max(dim=-1, keepdim=True).values

    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logit = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logit).mean()

def lr_cosine(step, lr_max, lr_min, warmup_steps, total_steps):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    if step > total_steps:
        return lr_min
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (1 + math.cos(math.pi * progress)) * (lr_max - lr_min)


def clip_grad_norm_(params, max_norm, eps=1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    with torch.no_grad():
        total_norm = torch.stack([g.detach().pow(2).sum() for g in grads]).sum().sqrt()
        scale = max_norm / (total_norm + eps)
        if scale < 1:
            for g in grads:
                g.mul_(scale)

    return total_norm