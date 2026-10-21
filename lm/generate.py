import torch
from lm.nn_utils import softmax


@torch.no_grad()
def generate_ids(model, prompt_ids, max_new_tokens, temperature, top_p,
                 eos_id=None, generator=None):
    model.eval()
    device = next(model.parameters()).device
    ids = list(prompt_ids)
    new_ids = []

    for _ in range(max_new_tokens):
        window = ids[-model.context_length:]
        x = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)

        logits = model(x)[0, -1]

        probs = softmax(logits / temperature, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        remove = (cumsum - sorted_probs) > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()

        sample = torch.multinomial(sorted_probs, 1, generator=generator)
        next_id = sorted_idx[sample].item()

        ids.append(next_id)
        new_ids.append(next_id)

        if eos_id is not None and next_id == eos_id:
            break

    return new_ids