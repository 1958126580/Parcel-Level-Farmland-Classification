"""
    Attention mechanisms for Parcel-Level Farmland Classification

This module provides attention layers:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Self-Attention with mask support
"""

using Flux
using NNlib

# ============================================================================
# Scaled Dot-Product Attention
# ============================================================================

"""
    scaled_dot_product_attention(Q, K, V; mask=nothing, dropout=0.0)

Compute scaled dot-product attention.

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

# Arguments
- `Q`: Query tensor (batch_size × seq_len × d_k)
- `K`: Key tensor (batch_size × seq_len × d_k)
- `V`: Value tensor (batch_size × seq_len × d_v)
- `mask`: Optional attention mask
- `dropout`: Dropout probability for attention weights

# Returns
- Tuple of (output, attention_weights)
"""
function scaled_dot_product_attention(Q, K, V; mask=nothing, dropout_rate=0.0)
    d_k = Float32(size(Q, 3))

    # Compute attention scores: QK^T / sqrt(d_k)
    # Q: (B, T, d_k) -> transpose K for matrix multiplication
    # scores: (B, T, T)
    scores = batched_mul(Q, batched_transpose(K)) ./ sqrt(d_k)

    # Apply mask if provided
    if mask !== nothing
        # mask: (B, T) -> (B, T, T)
        # Where mask is false, set scores to -inf
        mask_expanded = reshape(mask, size(mask, 1), 1, size(mask, 2))
        scores = ifelse.(mask_expanded, scores, Float32(-1e9))
    end

    # Apply softmax along last dimension
    attention_weights = softmax(scores, dims=3)

    # Apply dropout
    if dropout_rate > 0 && Flux.istraining()
        attention_weights = Flux.dropout(attention_weights, dropout_rate)
    end

    # Apply attention to values
    output = batched_mul(attention_weights, V)

    return output, attention_weights
end

"""
    batched_mul(A, B)

Batched matrix multiplication for 3D tensors.

# Arguments
- `A`: Tensor (batch_size × m × k)
- `B`: Tensor (batch_size × k × n)

# Returns
- Tensor (batch_size × m × n)
"""
function batched_mul(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    B_size, M, K = size(A)
    _, _, N = size(B)

    result = zeros(T, B_size, M, N)

    for b in 1:B_size
        result[b, :, :] = A[b, :, :] * B[b, :, :]
    end

    return result
end

"""
    batched_transpose(A)

Transpose last two dimensions of a 3D tensor.

# Arguments
- `A`: Tensor (batch_size × m × n)

# Returns
- Tensor (batch_size × n × m)
"""
function batched_transpose(A::AbstractArray{T, 3}) where T
    return permutedims(A, (1, 3, 2))
end

# ============================================================================
# Multi-Head Attention
# ============================================================================

"""
    MultiHeadAttention

Multi-head attention layer as described in "Attention is All You Need".

MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

# Fields
- `n_heads::Int`: Number of attention heads
- `d_model::Int`: Model dimension
- `d_k::Int`: Key dimension per head
- `d_v::Int`: Value dimension per head
- `W_Q::Dense`: Query projection
- `W_K::Dense`: Key projection
- `W_V::Dense`: Value projection
- `W_O::Dense`: Output projection
- `dropout::Dropout`: Dropout layer
"""
struct MultiHeadAttention
    n_heads::Int
    d_model::Int
    d_k::Int
    d_v::Int
    W_Q::Dense
    W_K::Dense
    W_V::Dense
    W_O::Dense
    dropout::Dropout
end

Flux.@functor MultiHeadAttention

"""
    MultiHeadAttention(d_model::Int, n_heads::Int; dropout::Float64=0.1)

Create multi-head attention layer.

# Arguments
- `d_model::Int`: Model dimension
- `n_heads::Int`: Number of attention heads
- `dropout::Float64`: Dropout probability
"""
function MultiHeadAttention(d_model::Int, n_heads::Int; dropout::Float64=0.1)
    @assert d_model % n_heads == 0 "d_model ($d_model) must be divisible by n_heads ($n_heads)"

    d_k = d_model ÷ n_heads
    d_v = d_model ÷ n_heads

    W_Q = Dense(d_model, d_model)
    W_K = Dense(d_model, d_model)
    W_V = Dense(d_model, d_model)
    W_O = Dense(d_model, d_model)

    return MultiHeadAttention(n_heads, d_model, d_k, d_v, W_Q, W_K, W_V, W_O, Dropout(dropout))
end

"""
    (mha::MultiHeadAttention)(Q, K, V; mask=nothing)

Apply multi-head attention.

# Arguments
- `Q`: Query tensor (batch_size × seq_len × d_model)
- `K`: Key tensor (batch_size × seq_len × d_model)
- `V`: Value tensor (batch_size × seq_len × d_model)
- `mask`: Optional attention mask

# Returns
- Tuple of (output, attention_weights)
"""
function (mha::MultiHeadAttention)(Q, K, V; mask=nothing)
    B, T_q, _ = size(Q)
    _, T_k, _ = size(K)

    # Project Q, K, V
    Q_proj = apply_dense_3d(mha.W_Q, Q)  # (B, T_q, d_model)
    K_proj = apply_dense_3d(mha.W_K, K)  # (B, T_k, d_model)
    V_proj = apply_dense_3d(mha.W_V, V)  # (B, T_k, d_model)

    # Split into heads: (B, T, d_model) -> (B, T, n_heads, d_k) -> (B*n_heads, T, d_k)
    Q_heads = split_heads(Q_proj, mha.n_heads, mha.d_k)
    K_heads = split_heads(K_proj, mha.n_heads, mha.d_k)
    V_heads = split_heads(V_proj, mha.n_heads, mha.d_v)

    # Apply attention
    attn_output, attn_weights = scaled_dot_product_attention(Q_heads, K_heads, V_heads; mask=mask)

    # Combine heads: (B*n_heads, T, d_v) -> (B, T, d_model)
    combined = combine_heads(attn_output, mha.n_heads, B, T_q)

    # Final projection
    output = apply_dense_3d(mha.W_O, combined)
    output = mha.dropout(output)

    return output, attn_weights
end

"""
    apply_dense_3d(layer::Dense, x)

Apply Dense layer to 3D tensor.

# Arguments
- `layer::Dense`: Dense layer
- `x`: Tensor (batch_size × seq_len × features)

# Returns
- Tensor (batch_size × seq_len × out_features)
"""
function apply_dense_3d(layer::Dense, x)
    B, T, C = size(x)
    x_flat = reshape(x, B * T, C)
    out = layer(x_flat')  # (out_features, B*T)
    return permutedims(reshape(out, size(out, 1), B, T), (2, 3, 1))
end

"""
    split_heads(x, n_heads::Int, d_k::Int)

Split tensor into multiple heads.

# Arguments
- `x`: Tensor (batch_size × seq_len × d_model)
- `n_heads::Int`: Number of heads
- `d_k::Int`: Dimension per head

# Returns
- Tensor (batch_size × n_heads × seq_len × d_k) reshaped to (batch_size*n_heads × seq_len × d_k)
"""
function split_heads(x, n_heads::Int, d_k::Int)
    B, T, D = size(x)
    # Reshape: (B, T, D) -> (B, T, n_heads, d_k) -> (B, n_heads, T, d_k) -> (B*n_heads, T, d_k)
    reshaped = reshape(x, B, T, n_heads, d_k)
    transposed = permutedims(reshaped, (1, 3, 2, 4))
    return reshape(transposed, B * n_heads, T, d_k)
end

"""
    combine_heads(x, n_heads::Int, batch_size::Int, seq_len::Int)

Combine multiple heads back to single tensor.

# Arguments
- `x`: Tensor (batch_size*n_heads × seq_len × d_v)
- `n_heads::Int`: Number of heads
- `batch_size::Int`: Original batch size
- `seq_len::Int`: Sequence length

# Returns
- Tensor (batch_size × seq_len × d_model)
"""
function combine_heads(x, n_heads::Int, batch_size::Int, seq_len::Int)
    d_v = size(x, 3)
    # Reshape: (B*n_heads, T, d_v) -> (B, n_heads, T, d_v) -> (B, T, n_heads, d_v) -> (B, T, d_model)
    reshaped = reshape(x, batch_size, n_heads, seq_len, d_v)
    transposed = permutedims(reshaped, (1, 3, 2, 4))
    return reshape(transposed, batch_size, seq_len, n_heads * d_v)
end

# ============================================================================
# Self-Attention Layer
# ============================================================================

"""
    SelfAttention

Self-attention layer (Q = K = V = input).

# Fields
- `mha::MultiHeadAttention`: Multi-head attention module
- `norm::LayerNorm`: Layer normalization
"""
struct SelfAttention
    mha::MultiHeadAttention
    norm::LayerNorm
end

Flux.@functor SelfAttention

"""
    SelfAttention(d_model::Int, n_heads::Int; dropout::Float64=0.1)

Create self-attention layer.
"""
function SelfAttention(d_model::Int, n_heads::Int; dropout::Float64=0.1)
    mha = MultiHeadAttention(d_model, n_heads; dropout=dropout)
    norm = LayerNorm(d_model)
    return SelfAttention(mha, norm)
end

"""
    (sa::SelfAttention)(x; mask=nothing)

Apply self-attention with residual connection.
"""
function (sa::SelfAttention)(x; mask=nothing)
    attn_out, _ = sa.mha(x, x, x; mask=mask)
    return sa.norm(x .+ attn_out)
end

# ============================================================================
# Attention with Feed-Forward
# ============================================================================

"""
    TransformerBlock

Complete transformer block with attention and feed-forward layers.

# Fields
- `attention::MultiHeadAttention`: Multi-head attention
- `norm1::LayerNorm`: First layer normalization
- `ffn::Chain`: Feed-forward network
- `norm2::LayerNorm`: Second layer normalization
- `dropout::Dropout`: Dropout layer
"""
struct TransformerBlock
    attention::MultiHeadAttention
    norm1::LayerNorm
    ffn::Chain
    norm2::LayerNorm
    dropout::Dropout
end

Flux.@functor TransformerBlock

"""
    TransformerBlock(d_model::Int, n_heads::Int, d_ff::Int;
                     dropout::Float64=0.1, activation=gelu)

Create transformer block.

# Arguments
- `d_model::Int`: Model dimension
- `n_heads::Int`: Number of attention heads
- `d_ff::Int`: Feed-forward network dimension
- `dropout::Float64`: Dropout probability
- `activation`: Activation function
"""
function TransformerBlock(d_model::Int, n_heads::Int, d_ff::Int;
                          dropout::Float64=0.1, activation=gelu)
    attention = MultiHeadAttention(d_model, n_heads; dropout=dropout)
    norm1 = LayerNorm(d_model)
    ffn = Chain(
        Dense(d_model, d_ff, activation),
        Dropout(dropout),
        Dense(d_ff, d_model),
        Dropout(dropout)
    )
    norm2 = LayerNorm(d_model)

    return TransformerBlock(attention, norm1, ffn, norm2, Dropout(dropout))
end

"""
    (block::TransformerBlock)(x; mask=nothing)

Apply transformer block.

# Arguments
- `x`: Input tensor (batch_size × seq_len × d_model)
- `mask`: Optional attention mask

# Returns
- Output tensor (batch_size × seq_len × d_model)
"""
function (block::TransformerBlock)(x; mask=nothing)
    # Self-attention with residual connection and layer norm
    attn_out, _ = block.attention(x, x, x; mask=mask)
    x = block.norm1(x .+ attn_out)

    # Feed-forward with residual connection and layer norm
    ffn_out = apply_ffn_3d(block.ffn, x)
    x = block.norm2(x .+ ffn_out)

    return x
end

"""
    apply_ffn_3d(ffn::Chain, x)

Apply feed-forward network to 3D tensor.
"""
function apply_ffn_3d(ffn::Chain, x)
    B, T, D = size(x)
    x_flat = reshape(x, B * T, D)
    out = ffn(x_flat')  # (D, B*T)
    return permutedims(reshape(out, size(out, 1), B, T), (2, 3, 1))
end
