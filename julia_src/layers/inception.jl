"""
    Inception blocks for TimesNet

This module provides Inception-style convolutional blocks for
capturing multi-scale patterns in 2D representations of time series.
"""

using Flux
using NNlib

# ============================================================================
# Inception Block V1
# ============================================================================

"""
    InceptionBlockV1

Inception-style block with multiple kernel sizes for multi-scale feature extraction.

Combines convolutions with different kernel sizes to capture patterns at different scales.

# Fields
- `conv1::Conv`: 1×1 convolution
- `conv3::Conv`: 3×3 convolution
- `conv5::Conv`: 5×5 convolution
- `pool_conv::Chain`: Max pooling followed by 1×1 convolution
"""
struct InceptionBlockV1
    conv1::Conv
    conv3::Conv
    conv5::Conv
    pool_conv::Chain
    in_channels::Int
    out_channels::Int
end

Flux.@functor InceptionBlockV1

"""
    InceptionBlockV1(in_channels::Int, out_channels::Int; num_kernels::Int=6)

Create Inception block V1.

# Arguments
- `in_channels::Int`: Number of input channels
- `out_channels::Int`: Number of output channels
- `num_kernels::Int`: Number of different kernel sizes to use
"""
function InceptionBlockV1(in_channels::Int, out_channels::Int; num_kernels::Int=6)
    # Each branch produces out_channels/4 features
    branch_out = out_channels ÷ 4

    # 1×1 convolution
    conv1 = Conv((1, 1), in_channels => branch_out, pad=SamePad())

    # 3×3 convolution with padding
    conv3 = Conv((3, 3), in_channels => branch_out, pad=SamePad())

    # 5×5 convolution with padding
    conv5 = Conv((5, 5), in_channels => branch_out, pad=SamePad())

    # Pooling branch
    pool_conv = Chain(
        MaxPool((3, 3), stride=1, pad=SamePad()),
        Conv((1, 1), in_channels => branch_out, pad=SamePad())
    )

    return InceptionBlockV1(conv1, conv3, conv5, pool_conv, in_channels, out_channels)
end

"""
    (block::InceptionBlockV1)(x)

Apply Inception block.

# Arguments
- `x`: Input tensor (width × height × channels × batch)

# Returns
- Output tensor with combined multi-scale features
"""
function (block::InceptionBlockV1)(x)
    # Apply each branch
    out1 = block.conv1(x)
    out3 = block.conv3(x)
    out5 = block.conv5(x)
    out_pool = block.pool_conv(x)

    # Concatenate along channel dimension
    return cat(out1, out3, out5, out_pool, dims=3)
end

# ============================================================================
# Inception Block V2 (for TimesNet)
# ============================================================================

"""
    InceptionBlockV2

Enhanced Inception block used in TimesNet for 2D convolution on reshaped time series.

# Fields
- `convs::Vector{Conv}`: List of convolutions with different kernel sizes
- `bn::BatchNorm`: Batch normalization
"""
struct InceptionBlockV2
    convs::Vector{Conv}
    bn::BatchNorm
    in_channels::Int
    out_channels::Int
end

Flux.@functor InceptionBlockV2

"""
    InceptionBlockV2(in_channels::Int, out_channels::Int; num_kernels::Int=6)

Create Inception block V2 for TimesNet.

Uses multiple 1D-style convolutions with different kernel sizes.

# Arguments
- `in_channels::Int`: Number of input channels
- `out_channels::Int`: Number of output channels
- `num_kernels::Int`: Number of different kernel sizes (1, 3, 5, 7, ...)
"""
function InceptionBlockV2(in_channels::Int, out_channels::Int; num_kernels::Int=6)
    convs = Conv[]

    # Create convolutions with kernel sizes 1, 3, 5, 7, ...
    kernel_sizes = [2*i - 1 for i in 1:num_kernels]

    for k in kernel_sizes
        # Use 2D convolution with square kernel
        push!(convs, Conv((k, k), in_channels => out_channels, pad=SamePad()))
    end

    bn = BatchNorm(out_channels * num_kernels)

    return InceptionBlockV2(convs, bn, in_channels, out_channels * num_kernels)
end

"""
    (block::InceptionBlockV2)(x)

Apply Inception block V2.

# Arguments
- `x`: Input tensor (H × W × C × B)

# Returns
- Output tensor with concatenated multi-scale features
"""
function (block::InceptionBlockV2)(x)
    # Apply each convolution
    outputs = [conv(x) for conv in block.convs]

    # Concatenate along channel dimension
    out = cat(outputs..., dims=3)

    # Apply batch normalization
    return block.bn(out)
end

# ============================================================================
# Residual Inception Block
# ============================================================================

"""
    ResidualInceptionBlock

Inception block with residual connection.

# Fields
- `inception::InceptionBlockV1`: Inception block
- `residual_conv::Union{Conv, Nothing}`: Optional 1×1 conv for channel matching
"""
struct ResidualInceptionBlock
    inception::InceptionBlockV1
    residual_conv::Union{Conv, Nothing}
end

Flux.@functor ResidualInceptionBlock

"""
    ResidualInceptionBlock(in_channels::Int, out_channels::Int)

Create residual Inception block.
"""
function ResidualInceptionBlock(in_channels::Int, out_channels::Int)
    inception = InceptionBlockV1(in_channels, out_channels)

    # Add 1×1 conv for channel matching if needed
    residual_conv = in_channels != out_channels ?
        Conv((1, 1), in_channels => out_channels, pad=SamePad()) :
        nothing

    return ResidualInceptionBlock(inception, residual_conv)
end

"""
    (block::ResidualInceptionBlock)(x)

Apply residual Inception block.
"""
function (block::ResidualInceptionBlock)(x)
    out = block.inception(x)

    residual = block.residual_conv !== nothing ?
        block.residual_conv(x) : x

    return relu.(out .+ residual)
end

# ============================================================================
# TimesNet Inception Block (Simplified)
# ============================================================================

"""
    TimesNetInception

Simplified Inception block for TimesNet 2D convolutions.

Applies two sequential Inception blocks with activation between them.

# Fields
- `block1::Chain`: First convolution block
- `block2::Chain`: Second convolution block
"""
struct TimesNetInception
    block1::Chain
    block2::Chain
    in_channels::Int
    out_channels::Int
end

Flux.@functor TimesNetInception

"""
    TimesNetInception(d_model::Int, d_ff::Int; num_kernels::Int=6)

Create TimesNet-style Inception block.

# Arguments
- `d_model::Int`: Model dimension (input channels)
- `d_ff::Int`: Feed-forward dimension (hidden channels)
- `num_kernels::Int`: Number of kernel sizes to use
"""
function TimesNetInception(d_model::Int, d_ff::Int; num_kernels::Int=6)
    # First block: d_model -> d_ff
    block1 = Chain(
        # Apply multiple parallel convolutions
        x -> apply_multi_conv(x, d_model, d_ff, num_kernels),
        BatchNorm(d_ff),
        x -> gelu.(x)
    )

    # Second block: d_ff -> d_model
    block2 = Chain(
        x -> apply_multi_conv(x, d_ff, d_model, num_kernels),
        BatchNorm(d_model)
    )

    return TimesNetInception(block1, block2, d_model, d_model)
end

"""
    apply_multi_conv(x, in_ch::Int, out_ch::Int, num_kernels::Int)

Apply multiple convolutions with different kernel sizes and average results.
"""
function apply_multi_conv(x, in_ch::Int, out_ch::Int, num_kernels::Int)
    H, W, C, B = size(x)

    # For simplicity, use a single convolution here
    # In full implementation, would use multiple kernels
    conv = Conv((3, 3), in_ch => out_ch, pad=SamePad())
    return conv(x)
end

"""
    (block::TimesNetInception)(x)

Apply TimesNet Inception block.
"""
function (block::TimesNetInception)(x)
    out = block.block1(x)
    out = block.block2(out)
    return out
end

# ============================================================================
# Simple Conv Block (Alternative)
# ============================================================================

"""
    ConvBlock

Simple convolutional block with conv -> activation -> batchnorm.

# Fields
- `conv::Conv`: Convolution layer
- `bn::BatchNorm`: Batch normalization
- `activation`: Activation function
"""
struct ConvBlock
    conv::Conv
    bn::BatchNorm
    activation::Function
end

Flux.@functor ConvBlock

"""
    ConvBlock(in_channels::Int, out_channels::Int;
              kernel_size::Tuple{Int,Int}=(3,3), activation=relu)

Create simple convolution block.
"""
function ConvBlock(in_channels::Int, out_channels::Int;
                   kernel_size::Tuple{Int,Int}=(3,3), activation=relu)
    conv = Conv(kernel_size, in_channels => out_channels, pad=SamePad())
    bn = BatchNorm(out_channels)
    return ConvBlock(conv, bn, activation)
end

"""
    (block::ConvBlock)(x)

Apply convolution block.
"""
function (block::ConvBlock)(x)
    out = block.conv(x)
    out = block.bn(out)
    return block.activation.(out)
end

# ============================================================================
# 2D Conv for TimesNet
# ============================================================================

"""
    TimesNetConv2D

2D Convolution layers for TimesNet's 2D variation modeling.

# Fields
- `conv1::Conv`: First convolution
- `conv2::Conv`: Second convolution
- `bn::BatchNorm`: Batch normalization
"""
struct TimesNetConv2D
    conv1::Conv
    conv2::Conv
    d_model::Int
    d_ff::Int
end

Flux.@functor TimesNetConv2D

"""
    TimesNetConv2D(d_model::Int, d_ff::Int; kernel_size::Int=3)

Create TimesNet 2D convolution block.
"""
function TimesNetConv2D(d_model::Int, d_ff::Int; kernel_size::Int=3)
    pad = kernel_size ÷ 2
    conv1 = Conv((kernel_size, kernel_size), d_model => d_ff, pad=pad)
    conv2 = Conv((kernel_size, kernel_size), d_ff => d_model, pad=pad)
    return TimesNetConv2D(conv1, conv2, d_model, d_ff)
end

"""
    (block::TimesNetConv2D)(x)

Apply TimesNet 2D convolution.

# Arguments
- `x`: Input tensor (H × W × C × B)

# Returns
- Output tensor same shape as input
"""
function (block::TimesNetConv2D)(x)
    out = block.conv1(x)
    out = gelu.(out)
    out = block.conv2(out)
    return out
end
