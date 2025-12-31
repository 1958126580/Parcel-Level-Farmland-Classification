"""
    Transformer Model Implementation

This module implements a Transformer encoder model for time series classification,
following the architecture from "Attention is All You Need" (Vaswani et al., 2017).

Architecture:
1. Data Embedding: Token + Positional encoding
2. Encoder: Stack of TransformerEncoderLayers
   - Multi-head self-attention
   - Feed-forward network
   - Layer normalization and residual connections
3. Classification Head: Global pooling + projection
"""

using Flux
using Flux: @functor, Dense, Dropout, Chain
using NNlib
using Statistics

# ============================================================================
# Transformer Encoder Layer
# ============================================================================

"""
    TransformerEncoderLayer

Single layer of the Transformer encoder.

# Fields
- `self_attn::MultiHeadAttention`: Multi-head self-attention
- `ffn::Chain`: Feed-forward network
- `norm1::LayerNorm`: First layer normalization
- `norm2::LayerNorm`: Second layer normalization
- `dropout1::Dropout`: Dropout after attention
- `dropout2::Dropout`: Dropout after FFN
"""
struct TransformerEncoderLayer
    self_attn::MultiHeadAttention
    ffn::Chain
    norm1::LayerNorm
    norm2::LayerNorm
    dropout1::Dropout
    dropout2::Dropout
end

@functor TransformerEncoderLayer

"""
    TransformerEncoderLayer(d_model::Int, n_heads::Int, d_ff::Int;
                            dropout::Float64=0.1, activation=gelu)

Create Transformer encoder layer.

# Arguments
- `d_model::Int`: Model dimension
- `n_heads::Int`: Number of attention heads
- `d_ff::Int`: Feed-forward network dimension
- `dropout::Float64`: Dropout probability
- `activation`: Activation function for FFN
"""
function TransformerEncoderLayer(d_model::Int, n_heads::Int, d_ff::Int;
                                  dropout::Float64=0.1, activation=gelu)
    self_attn = MultiHeadAttention(d_model, n_heads; dropout=dropout)

    ffn = Chain(
        Dense(d_model, d_ff, activation),
        Dropout(dropout),
        Dense(d_ff, d_model),
        Dropout(dropout)
    )

    norm1 = LayerNorm(d_model)
    norm2 = LayerNorm(d_model)
    dropout1 = Dropout(dropout)
    dropout2 = Dropout(dropout)

    return TransformerEncoderLayer(self_attn, ffn, norm1, norm2, dropout1, dropout2)
end

"""
    (layer::TransformerEncoderLayer)(x; mask=nothing)

Apply encoder layer.

# Arguments
- `x`: Input tensor (batch_size × seq_len × d_model)
- `mask`: Optional attention mask

# Returns
- Output tensor (batch_size × seq_len × d_model)
"""
function (layer::TransformerEncoderLayer)(x; mask=nothing)
    # Self-attention with residual connection
    attn_out, _ = layer.self_attn(x, x, x; mask=mask)
    x = layer.norm1(x .+ layer.dropout1(attn_out))

    # Feed-forward with residual connection
    B, T, D = size(x)
    x_flat = reshape(x, B * T, D)
    ffn_out = layer.ffn(x_flat')  # (D, B*T)
    ffn_out = permutedims(reshape(ffn_out, D, B, T), (2, 3, 1))  # (B, T, D)
    x = layer.norm2(x .+ layer.dropout2(ffn_out))

    return x
end

# ============================================================================
# Transformer Encoder
# ============================================================================

"""
    TransformerEncoder

Stack of Transformer encoder layers.

# Fields
- `layers::Vector{TransformerEncoderLayer}`: Encoder layers
- `norm::LayerNorm`: Final layer normalization
"""
struct TransformerEncoder
    layers::Vector{TransformerEncoderLayer}
    norm::LayerNorm
end

@functor TransformerEncoder

"""
    TransformerEncoder(n_layers::Int, d_model::Int, n_heads::Int, d_ff::Int;
                       dropout::Float64=0.1)

Create Transformer encoder with multiple layers.
"""
function TransformerEncoder(n_layers::Int, d_model::Int, n_heads::Int, d_ff::Int;
                            dropout::Float64=0.1)
    layers = [TransformerEncoderLayer(d_model, n_heads, d_ff; dropout=dropout)
              for _ in 1:n_layers]
    norm = LayerNorm(d_model)
    return TransformerEncoder(layers, norm)
end

"""
    (encoder::TransformerEncoder)(x; mask=nothing)

Apply encoder.
"""
function (encoder::TransformerEncoder)(x; mask=nothing)
    for layer in encoder.layers
        x = layer(x; mask=mask)
    end
    return encoder.norm(x)
end

# ============================================================================
# Transformer Classifier
# ============================================================================

"""
    TransformerClassifier

Complete Transformer model for time series classification.

# Fields
- `embedding::DataEmbedding`: Input embedding layer
- `encoder::TransformerEncoder`: Transformer encoder
- `projection::Dense`: Classification head
- `dropout::Dropout`: Dropout layer
- `pooling::Symbol`: Pooling method (:cls, :mean, :flatten)
"""
struct TransformerClassifier
    embedding::DataEmbedding
    encoder::TransformerEncoder
    projection::Dense
    dropout::Dropout
    pooling::Symbol
    seq_len::Int
    d_model::Int
    num_class::Int
end

@functor TransformerClassifier

"""
    TransformerClassifier(config::Config; pooling::Symbol=:flatten)

Create Transformer classifier from configuration.

# Arguments
- `config::Config`: Model configuration
- `pooling::Symbol`: Pooling method
  - `:cls`: Use first token (requires CLS token prepending)
  - `:mean`: Mean pooling over sequence
  - `:flatten`: Flatten all tokens (default)
"""
function TransformerClassifier(config::Config; pooling::Symbol=:flatten)
    embedding = DataEmbedding(
        config.enc_in,
        config.d_model;
        dropout=config.dropout
    )

    encoder = TransformerEncoder(
        config.e_layers,
        config.d_model,
        config.n_heads,
        config.d_ff;
        dropout=config.dropout
    )

    # Projection layer size depends on pooling method
    if pooling == :flatten
        proj_input = config.d_model * config.seq_len
    else
        proj_input = config.d_model
    end

    projection = Dense(proj_input, config.num_class)
    dropout = Dropout(config.dropout)

    return TransformerClassifier(
        embedding, encoder, projection, dropout, pooling,
        config.seq_len, config.d_model, config.num_class
    )
end

"""
    TransformerClassifier(; kwargs...)

Create Transformer classifier with keyword arguments.
"""
function TransformerClassifier(; seq_len::Int=24, enc_in::Int=1, d_model::Int=128,
                                 n_heads::Int=8, d_ff::Int=256, e_layers::Int=3,
                                 dropout::Float64=0.1, num_class::Int=8,
                                 pooling::Symbol=:flatten)
    config = Config(
        seq_len=seq_len,
        enc_in=enc_in,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        e_layers=e_layers,
        dropout=dropout,
        num_class=num_class
    )
    return TransformerClassifier(config; pooling=pooling)
end

"""
    (model::TransformerClassifier)(x, mask=nothing)

Forward pass of Transformer classifier.

# Arguments
- `x`: Input tensor (batch_size × seq_len × enc_in)
- `mask`: Optional padding mask (batch_size × seq_len)

# Returns
- Logits tensor (batch_size × num_class)
"""
function (model::TransformerClassifier)(x, mask=nothing)
    B = size(x, 1)

    # 1. Embedding
    embedded = model.embedding(x, nothing)  # (B, T, d_model)

    # 2. Encode
    encoded = model.encoder(embedded; mask=mask)  # (B, T, d_model)

    # 3. Apply activation and dropout
    output = gelu.(encoded)
    output = model.dropout(output)

    # 4. Apply mask if provided
    if mask !== nothing
        mask_expanded = reshape(Float32.(mask), B, size(mask, 2), 1)
        output = output .* mask_expanded
    end

    # 5. Pool
    if model.pooling == :cls
        # Use first token
        pooled = output[:, 1, :]  # (B, d_model)
    elseif model.pooling == :mean
        # Mean pooling
        if mask !== nothing
            # Masked mean
            sum_output = sum(output, dims=2)
            count = sum(mask, dims=2)
            pooled = dropdims(sum_output ./ max.(count, 1), dims=2)
        else
            pooled = dropdims(mean(output, dims=2), dims=2)  # (B, d_model)
        end
    else  # :flatten
        # Flatten
        pooled = reshape(output, B, :)  # (B, T*d_model)
    end

    # 6. Project to classes
    logits = model.projection(pooled')  # (num_class, B)

    return permutedims(logits)  # (B, num_class)
end

# ============================================================================
# iTransformer (Inverted Transformer)
# ============================================================================

"""
    iTransformer

Inverted Transformer that treats each time step as a token and
each channel as a sequence position.

This is particularly effective when the number of channels/features
is large relative to the sequence length.

# Fields
- `embedding::Dense`: Channel embedding
- `encoder::TransformerEncoder`: Transformer encoder
- `projection::Dense`: Classification head
"""
struct iTransformer
    embedding::Dense
    encoder::TransformerEncoder
    projection::Dense
    dropout::Dropout
    seq_len::Int
    d_model::Int
    num_class::Int
end

@functor iTransformer

"""
    iTransformer(; kwargs...)

Create iTransformer classifier.
"""
function iTransformer(; seq_len::Int=24, enc_in::Int=1, d_model::Int=128,
                       n_heads::Int=8, d_ff::Int=256, e_layers::Int=3,
                       dropout::Float64=0.1, num_class::Int=8)
    # Embed each channel (treat seq_len as channel dimension)
    embedding = Dense(seq_len, d_model)

    encoder = TransformerEncoder(e_layers, d_model, n_heads, d_ff; dropout=dropout)

    # Project to classes
    projection = Dense(d_model * enc_in, num_class)
    dropout_layer = Dropout(dropout)

    return iTransformer(embedding, encoder, projection, dropout_layer,
                        seq_len, d_model, num_class)
end

"""
    (model::iTransformer)(x, mask=nothing)

Forward pass of iTransformer.

# Arguments
- `x`: Input tensor (batch_size × seq_len × num_channels)
- `mask`: Optional mask

# Returns
- Logits tensor (batch_size × num_class)
"""
function (model::iTransformer)(x, mask=nothing)
    B, T, C = size(x)

    # Transpose: (B, T, C) -> (B, C, T)
    x_t = permutedims(x, (1, 3, 2))  # (B, C, T)

    # Embed each channel: (B, C, T) -> (B, C, d_model)
    x_flat = reshape(x_t, B * C, T)
    embedded = model.embedding(x_flat')  # (d_model, B*C)
    embedded = permutedims(reshape(embedded, model.d_model, B, C), (2, 3, 1))  # (B, C, d_model)

    # Encode (treat channels as sequence)
    encoded = model.encoder(embedded)  # (B, C, d_model)

    # Apply dropout
    output = model.dropout(gelu.(encoded))

    # Flatten and project
    output_flat = reshape(output, B, :)  # (B, C*d_model)
    logits = model.projection(output_flat')

    return permutedims(logits)
end

# ============================================================================
# Model Utilities
# ============================================================================

"""
    count_parameters(model::TransformerClassifier)

Count total trainable parameters.
"""
function count_parameters(model::TransformerClassifier)
    total = 0
    for p in Flux.params(model)
        total += length(p)
    end
    return total
end

"""
    model_summary(model::TransformerClassifier)

Print model summary.
"""
function model_summary(model::TransformerClassifier)
    println("\n" * "="^50)
    println("Transformer Classifier Summary")
    println("="^50)
    println("Sequence Length:   $(model.seq_len)")
    println("Model Dimension:   $(model.d_model)")
    println("Number of Classes: $(model.num_class)")
    println("Pooling Method:    $(model.pooling)")
    println("Number of Layers:  $(length(model.encoder.layers))")
    println("Total Parameters:  $(count_parameters(model))")
    println("="^50 * "\n")
end
