def _matmul_params_per_block(d_model, num_heads, d_ff, num_kv_heads):
    kv = num_heads if num_kv_heads is None else num_kv_heads
    d_head = d_model // num_heads
    attn = d_model * d_model   
    attn += 2 * d_model * kv * d_head  
    attn += d_model * d_model    
    ffn = 3 * d_model * d_ff 
    return attn + ffn


def count_params(vocab_size, context_length, d_model, num_layers, num_heads, d_ff,
                 num_kv_heads=None, tied=False):
    block = _matmul_params_per_block(d_model, num_heads, d_ff, num_kv_heads)
    block += 2 * d_model               
    total = num_layers * block
    total += vocab_size * d_model       
    total += d_model       
    if not tied:
        total += vocab_size * d_model  
    return total


def count_flops(vocab_size, context_length, d_model, num_layers, num_heads, d_ff,
                seq_len, num_kv_heads=None):
    matmul_params = num_layers * _matmul_params_per_block(d_model, num_heads, d_ff, num_kv_heads)
    matmul_params += vocab_size * d_model
    forward = 2 * matmul_params
    forward += 4 * num_layers * seq_len * d_model  # q·kᵀ and weights·v
    return forward, 3 * forward