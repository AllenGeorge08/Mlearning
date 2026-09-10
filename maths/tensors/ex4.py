import torch 

# Multi-head attention shape tracker
def attention_shape_tracker(batch_size,seq_len,embed_dim,num_heads):
    assert embed_dim %num_heads == 0,\
        "embed dim must be divisible by num heads"

    
    head_dim = embed_dim//num_heads
    print(f"Input : ({batch_size},{seq_len},{embed_dim})")

    # Input (B,T,D+) 
    x = torch.randn(batch_size,seq_len,embed_dim)

    # Q,K,V
    W_q = torch.randn(embed_dim,embed_dim)
    W_k = torch.randn(embed_dim,embed_dim)
    W_v = torch.randn(embed_dim,embed_dim)

    Q  = x@W_q
    K = x@W_k
    V = x@W_v 

    print(f"Q Projection : {tuple(Q.shape)}")
    print(f"K Projection : {tuple(K.shape)}")
    print(f"V Projection : {tuple(V.shape)}")

    # Split into heads
    # (B,T,E)
    # (B,T,H,D)
    # (B,H,T,D)

    Q = Q.reshape(batch_size,seq_len,num_heads,head_dim)
    K = K.reshape(batch_size,seq_len,num_heads,head_dim)
    V =  V.reshape(batch_size,seq_len,num_heads,head_dim)

    Q = Q.permute(0,2,1,3)
    K = K.permute(0,2,1,3)
    V = V.permute(0,2,1,3)

    print(f"Head split Q: {tuple(Q.shape)}") 
    print(f"Head split K: {tuple(K.shape)}") 
    print(f"Head split V: {tuple(V.shape)}")

    # Attention scores
    # Q: (B,H,T,D)
    # K: (B,H,T,D)
    # Scores: (B,H,T,T)
    scores = torch.einsum("bhtd,bhsd->bhts",Q,K)
    print(f"Attention Scores: {tuple(scores.shape)}")

    # Softmax weights 
    weights = torch.softmax(scores/(head_dim**0.5),dim=1)
    print(f"Softmax weights: {tuple(weights.shape)}")


    # Weighted sum
    # weights: (B, H, T, T) 
    # # V: (B, H, T, D) 
    # # output: (B, H, T, D)
    context = torch.einsum("bhts,bhsd->bhtd",weights,V)
    print(f"Weighted Sum: {tuple(context.shape)}")

    # -------------------------------------------------- # 7. Merge heads # --------------------------------------------------
    #  # (B, H, T, D) 
    # # ↓ # (B, T, H, D)
    #  # ↓ # (B, T, E)
    context = context.permute(0,2,1,3)
    context = context.reshape(
        batch_size,
        seq_len,
        embed_dim
    )
    print(f"Head merge: {tuple(context.shape)}")

    # Output projection
    W_o = torch.randn(embed_dim,embed_dim)
    output= context@W_o
    print(f"Output projection: {tuple(output.shape)}")
    return output




output = attention_shape_tracker(
    batch_size=2,
    seq_len=5,
    embed_dim=12,
    num_heads=3
)

print(output)
