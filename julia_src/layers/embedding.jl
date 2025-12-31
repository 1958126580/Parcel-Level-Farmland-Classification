"""
    Embedding layers for Parcel-Level Farmland Classification

This module provides embedding layers for time series data:
- Token embedding (value projection)
- Positional encoding (sinusoidal and learnable)
- Temporal encoding (time features)
- Data embedding (combined embeddings)
"""

using Flux

# ============================================================================
# Token Embedding
# ============================================================================

"""
    TokenEmbedding

Linear projection of input features to model dimension.

# Fields
- `projection::Dense`: Linear projection layer
"""
struct TokenEmbedding
    projection::Dense
end

Flux.@functor TokenEmbedding

"""
    TokenEmbedding(input_dim::Int, d_model::Int)

Create token embedding layer.

# Arguments
- `input_dim::Int`: Number of input features
- `d_model::Int`: Model embedding dimension
"""
function TokenEmbedding(input_dim::Int, d_model::Int)
    projection = Dense(input_dim, d_model)
    return TokenEmbedding(projection)
end

"""
    (embed::TokenEmbedding)(x)

Apply token embedding.

# Arguments
- `x`: Input tensor (batch_size × seq_len × input_dim)

# Returns
- Embedded tensor (batch_size × seq_len × d_model)
"""
function (embed::TokenEmbedding)(x)
    # x: (B, T, C)
    B, T, C = size(x)

    # Reshape for Dense layer: (B*T, C)
    x_flat = reshape(x, B * T, C)

    # Apply projection
    out = embed.projection(x_flat')  # (d_model, B*T)

    # Reshape back: (B, T, d_model)
    return permutedims(reshape(out, size(out, 1), B, T), (2, 3, 1))
end

# ============================================================================
# Positional Encoding
# ============================================================================

"""
    PositionalEncoding

Sinusoidal positional encoding as described in "Attention is All You Need".

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# Fields
- `encoding::Array{Float32, 3}`: Pre-computed positional encodings
- `dropout::Dropout`: Dropout layer
"""
struct PositionalEncoding
    encoding::Array{Float32, 3}
    dropout::Dropout
    d_model::Int
    max_len::Int
end

Flux.@functor PositionalEncoding

# Only make dropout trainable, encoding is fixed
Flux.trainable(pe::PositionalEncoding) = (dropout = pe.dropout,)

"""
    PositionalEncoding(d_model::Int; max_len::Int=5000, dropout::Float64=0.1)

Create positional encoding layer.

# Arguments
- `d_model::Int`: Model dimension
- `max_len::Int`: Maximum sequence length
- `dropout::Float64`: Dropout probability
"""
function PositionalEncoding(d_model::Int; max_len::Int=5000, dropout::Float64=0.1)
    # Pre-compute positional encodings
    pe = zeros(Float32, 1, max_len, d_model)

    position = collect(0:max_len-1)
    div_term = exp.(-log(10000.0f0) .* collect(0:2:d_model-1) ./ d_model)

    for pos in 1:max_len
        for i in 1:2:d_model
            pe[1, pos, i] = sin(position[pos] * div_term[(i + 1) ÷ 2])
            if i + 1 <= d_model
                pe[1, pos, i+1] = cos(position[pos] * div_term[(i + 1) ÷ 2])
            end
        end
    end

    return PositionalEncoding(pe, Dropout(dropout), d_model, max_len)
end

"""
    (pe::PositionalEncoding)(x)

Add positional encoding to input.

# Arguments
- `x`: Input tensor (batch_size × seq_len × d_model)

# Returns
- Tensor with positional encoding added
"""
function (pe::PositionalEncoding)(x)
    B, T, D = size(x)
    @assert D == pe.d_model "Input dimension ($D) must match d_model ($(pe.d_model))"
    @assert T <= pe.max_len "Sequence length ($T) exceeds max_len ($(pe.max_len))"

    # Add positional encoding
    encoding = repeat(pe.encoding[:, 1:T, :], B, 1, 1)
    return pe.dropout(x .+ encoding)
end

# ============================================================================
# Learnable Positional Embedding
# ============================================================================

"""
    LearnablePositionalEmbedding

Learnable positional embedding.

# Fields
- `embedding::Array{Float32, 3}`: Learnable position embeddings
"""
struct LearnablePositionalEmbedding
    embedding::Array{Float32, 3}
end

Flux.@functor LearnablePositionalEmbedding

"""
    LearnablePositionalEmbedding(d_model::Int; max_len::Int=5000)

Create learnable positional embedding.
"""
function LearnablePositionalEmbedding(d_model::Int; max_len::Int=5000)
    # Initialize with small random values
    embedding = randn(Float32, 1, max_len, d_model) .* 0.02f0
    return LearnablePositionalEmbedding(embedding)
end

"""
    (lpe::LearnablePositionalEmbedding)(x)

Add learnable positional embedding to input.
"""
function (lpe::LearnablePositionalEmbedding)(x)
    B, T, D = size(x)
    embedding = repeat(lpe.embedding[:, 1:T, :], B, 1, 1)
    return x .+ embedding
end

# ============================================================================
# Temporal Embedding
# ============================================================================

"""
    TemporalEmbedding

Embedding for temporal features (day of year, month, etc.).

# Fields
- `month_embed::Embedding`: Month embedding
- `day_embed::Embedding`: Day of year embedding
"""
struct TemporalEmbedding
    month_embed::Flux.Embedding
    day_embed::Flux.Embedding
    d_model::Int
end

Flux.@functor TemporalEmbedding

"""
    TemporalEmbedding(d_model::Int)

Create temporal embedding for time features.
"""
function TemporalEmbedding(d_model::Int)
    month_embed = Flux.Embedding(12, d_model)  # 12 months
    day_embed = Flux.Embedding(366, d_model)   # 366 days
    return TemporalEmbedding(month_embed, day_embed, d_model)
end

"""
    (te::TemporalEmbedding)(months, days)

Apply temporal embedding.

# Arguments
- `months`: Month indices (1-12)
- `days`: Day of year indices (1-366)
"""
function (te::TemporalEmbedding)(months, days)
    month_emb = te.month_embed(months)
    day_emb = te.day_embed(days)
    return month_emb .+ day_emb
end

# ============================================================================
# Fixed Time Features Encoding
# ============================================================================

"""
    time_features_encoding(x::AbstractArray, embed_type::Symbol=:timeF)

Create time feature encodings.

# Arguments
- `x`: Time indices or timestamps
- `embed_type::Symbol`: Type of embedding (:timeF, :fixed)

# Returns
- Time feature tensor
"""
function time_features_encoding(x::AbstractArray; freq::String="d")
    # For now, return zeros as placeholder
    # In full implementation, would extract hour, day, month, etc.
    return nothing
end

# ============================================================================
# Data Embedding (Combined)
# ============================================================================

"""
    DataEmbedding

Combined embedding layer that includes:
- Token embedding (value projection)
- Positional encoding
- Optional temporal embedding

# Fields
- `token_embed::TokenEmbedding`: Token (value) embedding
- `pos_embed::Union{PositionalEncoding, LearnablePositionalEmbedding}`: Position encoding
- `dropout::Dropout`: Dropout layer
"""
struct DataEmbedding
    token_embed::TokenEmbedding
    pos_embed::PositionalEncoding
    dropout::Dropout
end

Flux.@functor DataEmbedding

"""
    DataEmbedding(input_dim::Int, d_model::Int; embed_type::Symbol=:positional,
                  dropout::Float64=0.1, max_len::Int=5000)

Create combined data embedding layer.

# Arguments
- `input_dim::Int`: Number of input features
- `d_model::Int`: Model embedding dimension
- `embed_type::Symbol`: Type of positional encoding (:positional, :learnable)
- `dropout::Float64`: Dropout probability
- `max_len::Int`: Maximum sequence length
"""
function DataEmbedding(input_dim::Int, d_model::Int;
                       embed_type::Symbol=:positional,
                       dropout::Float64=0.1,
                       max_len::Int=5000)
    token_embed = TokenEmbedding(input_dim, d_model)
    pos_embed = PositionalEncoding(d_model; max_len=max_len, dropout=0.0)
    dropout_layer = Dropout(dropout)

    return DataEmbedding(token_embed, pos_embed, dropout_layer)
end

"""
    (de::DataEmbedding)(x, x_mark=nothing)

Apply combined data embedding.

# Arguments
- `x`: Input values (batch_size × seq_len × input_dim)
- `x_mark`: Optional time marks/features

# Returns
- Embedded tensor (batch_size × seq_len × d_model)
"""
function (de::DataEmbedding)(x, x_mark=nothing)
    # Token embedding
    embedded = de.token_embed(x)

    # Add positional encoding
    embedded = de.pos_embed(embedded)

    # Apply dropout
    return de.dropout(embedded)
end

# ============================================================================
# Patch Embedding (for PatchTST-style models)
# ============================================================================

"""
    PatchEmbedding

Patch-based embedding for time series (inspired by Vision Transformer).

Divides the time series into non-overlapping patches and embeds each patch.

# Fields
- `patch_len::Int`: Length of each patch
- `stride::Int`: Stride between patches
- `projection::Dense`: Projection layer
- `pos_embed::LearnablePositionalEmbedding`: Position embedding for patches
"""
struct PatchEmbedding
    patch_len::Int
    stride::Int
    projection::Dense
    pos_embed::LearnablePositionalEmbedding
end

Flux.@functor PatchEmbedding

"""
    PatchEmbedding(input_dim::Int, d_model::Int;
                   patch_len::Int=16, stride::Int=8)

Create patch embedding layer.
"""
function PatchEmbedding(input_dim::Int, d_model::Int;
                        patch_len::Int=16, stride::Int=8)
    projection = Dense(patch_len * input_dim, d_model)

    # Max number of patches
    max_patches = 100
    pos_embed = LearnablePositionalEmbedding(d_model; max_len=max_patches)

    return PatchEmbedding(patch_len, stride, projection, pos_embed)
end

"""
    (pe::PatchEmbedding)(x)

Apply patch embedding.

# Arguments
- `x`: Input tensor (batch_size × seq_len × input_dim)

# Returns
- Patch embeddings (batch_size × num_patches × d_model)
"""
function (pe::PatchEmbedding)(x)
    B, T, C = size(x)

    # Calculate number of patches
    num_patches = (T - pe.patch_len) ÷ pe.stride + 1

    # Extract patches
    patches = zeros(Float32, B, num_patches, pe.patch_len * C)
    for p in 1:num_patches
        start_idx = (p - 1) * pe.stride + 1
        end_idx = start_idx + pe.patch_len - 1
        patches[:, p, :] = reshape(x[:, start_idx:end_idx, :], B, :)
    end

    # Project patches
    patches_flat = reshape(patches, B * num_patches, :)
    embedded = pe.projection(patches_flat')  # (d_model, B*num_patches)
    embedded = permutedims(reshape(embedded, size(embedded, 1), B, num_patches), (2, 3, 1))

    # Add position embedding
    return pe.pos_embed(embedded)
end
