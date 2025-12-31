"""
    TimesNet Model Implementation

This module implements the TimesNet architecture for time series classification
as described in:

"TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"
(ICLR 2023)

Key Innovation:
TimesNet transforms 1D time series into 2D representations based on
automatically detected periods, enabling 2D convolutions to capture
both intra-period and inter-period variations.

Architecture:
1. Data Embedding: Project input features to model dimension
2. TimesBlocks: Multiple blocks that:
   a. Detect periods using FFT
   b. Reshape to 2D based on periods
   c. Apply 2D Inception convolutions
   d. Reshape back and aggregate
3. Classification Head: Final projection to class probabilities
"""

using Flux
using Flux: @functor
using FFTW
using NNlib
using Statistics

# ============================================================================
# TimesBlock
# ============================================================================

"""
    TimesBlock

Core building block of TimesNet that performs:
1. FFT-based period detection
2. Reshape 1D → 2D based on detected periods
3. 2D Inception convolution for multi-scale feature extraction
4. Reshape 2D → 1D and aggregate across periods

# Fields
- `seq_len::Int`: Input sequence length
- `pred_len::Int`: Prediction length (0 for classification)
- `k::Int`: Number of top periods to consider
- `conv::Chain`: 2D convolution layers (Inception blocks)
- `d_model::Int`: Model dimension
- `d_ff::Int`: Feed-forward dimension
"""
struct TimesBlock
    seq_len::Int
    pred_len::Int
    k::Int
    conv::Chain
    d_model::Int
    d_ff::Int
end

@functor TimesBlock

"""
    TimesBlock(config)

Create a TimesBlock from configuration.

# Arguments
- `config`: Configuration with seq_len, pred_len, top_k, d_model, d_ff, num_kernels
"""
function TimesBlock(; seq_len::Int, pred_len::Int=0, top_k::Int=5,
                     d_model::Int=128, d_ff::Int=256, num_kernels::Int=6)
    # Create 2D convolution layers
    conv = Chain(
        # First Inception block: d_model -> d_ff
        Conv((3, 3), d_model => d_ff, pad=SamePad()),
        BatchNorm(d_ff),
        x -> gelu.(x),
        # Second Inception block: d_ff -> d_model
        Conv((3, 3), d_ff => d_model, pad=SamePad()),
        BatchNorm(d_model)
    )

    return TimesBlock(seq_len, pred_len, top_k, conv, d_model, d_ff)
end

"""
    (block::TimesBlock)(x)

Apply TimesBlock to input.

# Arguments
- `x`: Input tensor (batch_size × seq_len × d_model)

# Returns
- Output tensor (batch_size × seq_len × d_model)
"""
function (block::TimesBlock)(x)
    B, T, N = size(x)

    # 1. FFT for period detection
    periods, period_weights = fft_for_period(x; k=block.k)

    # 2. Process each period
    results = []

    for i in 1:block.k
        period = max(2, periods[i])

        # Padding to ensure divisibility
        total_len = block.seq_len + block.pred_len
        if total_len % period != 0
            pad_len = period - (total_len % period)
            x_padded = cat(x, zeros(eltype(x), B, pad_len, N), dims=2)
        else
            pad_len = 0
            x_padded = x
        end

        # Reshape to 2D: (B, T+pad, C) -> (W, H, C, B) for Flux Conv2D
        padded_len = size(x_padded, 2)
        num_periods = padded_len ÷ period

        # (B, num_periods, period, C)
        x_reshaped = reshape(x_padded, B, num_periods, period, N)
        # (period, num_periods, C, B) - Flux expects (W, H, C, B)
        x_2d = permutedims(x_reshaped, (3, 2, 4, 1))

        # Apply 2D convolution
        out_2d = block.conv(x_2d)

        # Reshape back to 1D
        # (W, H, C, B) -> (B, H, W, C) -> (B, T, C)
        out_perm = permutedims(out_2d, (4, 2, 1, 3))
        out_1d = reshape(out_perm, B, :, N)

        # Trim to original length
        out_1d = out_1d[:, 1:total_len, :]

        push!(results, out_1d)
    end

    # 3. Aggregate results with period weights
    # Apply softmax to period weights
    weights_softmax = softmax(period_weights, dims=2)  # (B, k)

    # Weighted sum
    output = zeros(eltype(x), B, T, N)
    for i in 1:block.k
        weight = reshape(weights_softmax[:, i], B, 1, 1)
        output .+= results[i] .* weight
    end

    # 4. Residual connection
    output = output .+ x

    return output
end

# ============================================================================
# TimesNet Model
# ============================================================================

"""
    TimesNet

Complete TimesNet model for time series classification.

# Fields
- `embedding::DataEmbedding`: Data embedding layer
- `blocks::Vector{TimesBlock}`: TimesNet blocks
- `layer_norm::LayerNorm`: Layer normalization
- `projection::Dense`: Classification head
- `dropout::Dropout`: Dropout layer
- `config`: Model configuration
"""
struct TimesNet
    embedding::DataEmbedding
    blocks::Vector{TimesBlock}
    layer_norm::LayerNorm
    projection::Dense
    dropout::Dropout
    seq_len::Int
    d_model::Int
    num_class::Int
end

@functor TimesNet

"""
    TimesNet(config::Config)

Create TimesNet model from configuration.

# Arguments
- `config::Config`: Model configuration

# Example
```julia
config = Config(
    seq_len=24,
    enc_in=1,
    d_model=128,
    d_ff=256,
    e_layers=3,
    top_k=5,
    num_kernels=6,
    dropout=0.1,
    num_class=8
)
model = TimesNet(config)
```
"""
function TimesNet(config::Config)
    # Data embedding
    embedding = DataEmbedding(
        config.enc_in,
        config.d_model;
        dropout=config.dropout
    )

    # TimesNet blocks
    blocks = TimesBlock[]
    for _ in 1:config.e_layers
        block = TimesBlock(
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            top_k=config.top_k,
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_kernels=config.num_kernels
        )
        push!(blocks, block)
    end

    # Layer normalization
    layer_norm = LayerNorm(config.d_model)

    # Classification projection
    # Flatten: seq_len * d_model -> num_class
    projection = Dense(config.d_model * config.seq_len, config.num_class)

    dropout = Dropout(config.dropout)

    return TimesNet(
        embedding, blocks, layer_norm, projection, dropout,
        config.seq_len, config.d_model, config.num_class
    )
end

"""
    TimesNet(; kwargs...)

Create TimesNet with keyword arguments.

# Keyword Arguments
- `seq_len::Int=24`: Sequence length
- `enc_in::Int=1`: Number of input features
- `d_model::Int=128`: Model dimension
- `d_ff::Int=256`: Feed-forward dimension
- `e_layers::Int=3`: Number of encoder layers
- `top_k::Int=5`: Number of top periods
- `num_kernels::Int=6`: Number of inception kernels
- `dropout::Float64=0.1`: Dropout probability
- `num_class::Int=8`: Number of output classes
"""
function TimesNet(; seq_len::Int=24, enc_in::Int=1, d_model::Int=128,
                   d_ff::Int=256, e_layers::Int=3, top_k::Int=5,
                   num_kernels::Int=6, dropout::Float64=0.1, num_class::Int=8)
    config = Config(
        seq_len=seq_len,
        enc_in=enc_in,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        top_k=top_k,
        num_kernels=num_kernels,
        dropout=dropout,
        num_class=num_class
    )
    return TimesNet(config)
end

"""
    (model::TimesNet)(x, mask=nothing)

Forward pass of TimesNet.

# Arguments
- `x`: Input tensor (batch_size × seq_len × enc_in)
- `mask`: Optional padding mask (batch_size × seq_len)

# Returns
- Output tensor (batch_size × num_class)
"""
function (model::TimesNet)(x, mask=nothing)
    # 1. Embedding
    enc_out = model.embedding(x, nothing)  # (B, T, d_model)

    # 2. TimesNet blocks
    for block in model.blocks
        enc_out = model.layer_norm(block(enc_out))
    end

    # 3. Classification
    # Apply GELU activation
    output = gelu.(enc_out)

    # Apply dropout
    output = model.dropout(output)

    # Apply mask if provided
    if mask !== nothing
        # Expand mask: (B, T) -> (B, T, d_model)
        mask_expanded = reshape(mask, size(mask, 1), size(mask, 2), 1)
        output = output .* Float32.(mask_expanded)
    end

    # Flatten: (B, T, d_model) -> (B, T*d_model)
    B = size(output, 1)
    output_flat = reshape(output, B, :)

    # Project to classes
    logits = model.projection(output_flat')  # (num_class, B)

    return permutedims(logits)  # (B, num_class)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    count_parameters(model::TimesNet)

Count total number of trainable parameters.
"""
function count_parameters(model::TimesNet)
    total = 0
    for p in Flux.params(model)
        total += length(p)
    end
    return total
end

"""
    model_summary(model::TimesNet)

Print model summary.
"""
function model_summary(model::TimesNet)
    println("\n" * "="^50)
    println("TimesNet Model Summary")
    println("="^50)
    println("Sequence Length:   $(model.seq_len)")
    println("Model Dimension:   $(model.d_model)")
    println("Number of Classes: $(model.num_class)")
    println("Number of Blocks:  $(length(model.blocks))")
    println("Total Parameters:  $(count_parameters(model))")
    println("="^50 * "\n")
end

# ============================================================================
# Simplified TimesNet for Testing
# ============================================================================

"""
    SimpleTimesNet

Simplified TimesNet without Inception blocks for faster execution.
Uses simple 2D convolutions instead.
"""
struct SimpleTimesNet
    embedding::Dense
    blocks::Vector{Chain}
    layer_norm::LayerNorm
    projection::Dense
    seq_len::Int
    d_model::Int
    num_class::Int
end

@functor SimpleTimesNet

"""
    SimpleTimesNet(; kwargs...)

Create simplified TimesNet.
"""
function SimpleTimesNet(; seq_len::Int=24, enc_in::Int=1, d_model::Int=64,
                         e_layers::Int=2, num_class::Int=8, dropout::Float64=0.1)
    embedding = Dense(enc_in, d_model)

    blocks = Chain[]
    for _ in 1:e_layers
        block = Chain(
            Dense(d_model, d_model * 2, gelu),
            Dropout(dropout),
            Dense(d_model * 2, d_model),
            Dropout(dropout)
        )
        push!(blocks, block)
    end

    layer_norm = LayerNorm(d_model)
    projection = Dense(d_model * seq_len, num_class)

    return SimpleTimesNet(embedding, blocks, layer_norm, projection,
                          seq_len, d_model, num_class)
end

"""
    (model::SimpleTimesNet)(x, mask=nothing)

Forward pass of SimpleTimesNet.
"""
function (model::SimpleTimesNet)(x, mask=nothing)
    B, T, C = size(x)

    # Embedding: apply to each time step
    x_flat = reshape(x, B * T, C)
    embedded = model.embedding(x_flat')  # (d_model, B*T)
    embedded = permutedims(reshape(embedded, model.d_model, B, T), (2, 3, 1))  # (B, T, d_model)

    # Apply blocks with residual connections
    output = embedded
    for block in model.blocks
        # Apply block to each time step
        out_flat = reshape(output, B * T, model.d_model)
        block_out = block(out_flat')  # (d_model, B*T)
        block_out = permutedims(reshape(block_out, model.d_model, B, T), (2, 3, 1))
        output = output .+ block_out
    end

    # Layer norm (simplified - apply to last dimension)
    output = model.layer_norm(output)

    # Apply mask
    if mask !== nothing
        mask_expanded = reshape(Float32.(mask), B, T, 1)
        output = output .* mask_expanded
    end

    # Flatten and project
    output_flat = reshape(output, B, :)
    logits = model.projection(output_flat')

    return permutedims(logits)
end
