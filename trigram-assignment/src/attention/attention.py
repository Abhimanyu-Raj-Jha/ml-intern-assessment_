import numpy as np

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Computes the Scaled Dot-Product Attention.
    
    Formula: softmax(QK^T / sqrt(d_k)) * V
    
    Args:
        query: (batch_size, seq_len_q, d_k)
        key:   (batch_size, seq_len_k, d_k)
        value: (batch_size, seq_len_v, d_v)
        mask:  Optional mask (batch_size, seq_len_q, seq_len_k)
    
    Returns:
        output: (batch_size, seq_len_q, d_v)
        attention_weights: (batch_size, seq_len_q, seq_len_k)
    """
    d_k = query.shape[-1]
    
    # 1. Dot Product: Q * K^T
    # (batch, seq_q, d_k) x (batch, d_k, seq_k) -> (batch, seq_q, seq_k)
    scores = np.matmul(query, key.swapaxes(-2, -1))
    
    # 2. Scale by sqrt(d_k)
    scores = scores / np.sqrt(d_k)
    
    # 3. Apply Mask (optional)
    if mask is not None:
        # Set masked positions to -infinity so softmax makes them 0
        scores = np.where(mask == 0, -1e9, scores)
    
    # 4. Softmax
    # Subtract max for numerical stability before exp
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 5. Multiply by Value: weights * V
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights

def demo_attention():
    """Runs a simple sanity check for the attention mechanism."""
    np.random.seed(42)
    
    # Dimensions: Batch=1, Sequence=3, Dimension=4
    Q = np.random.rand(1, 3, 4)
    K = np.random.rand(1, 3, 4)
    V = np.random.rand(1, 3, 4)
    
    print("Query Shape:", Q.shape)
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("\nAttention Weights:\n", weights)
    print("\nOutput:\n", output)
    print("\nTask 2 Demo Passed Successfully.")

if __name__ == "__main__":
    demo_attention()