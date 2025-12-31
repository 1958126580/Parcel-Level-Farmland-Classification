"""
    FFT Period Detection for TimesNet

This module provides FFT-based period detection for identifying
dominant frequencies in time series data. This is a key component
of TimesNet that enables adaptive 2D convolution based on detected periods.
"""

using FFTW

# ============================================================================
# FFT Period Detection
# ============================================================================

"""
    fft_for_period(x::AbstractArray{T, 3}; k::Int=2) where T

Detect dominant periods in time series using FFT.

This function performs FFT analysis to identify the k most significant
periodic components in the input time series.

# Arguments
- `x::AbstractArray{T, 3}`: Input tensor (batch_size × seq_len × channels)
- `k::Int`: Number of top periods to detect

# Returns
- Tuple of (periods::Vector{Int}, period_weights::Matrix)
  - periods: Top-k periods detected
  - period_weights: Amplitude weights for each period

# Algorithm
1. Apply real FFT along time dimension
2. Compute amplitude spectrum
3. Average across batch and channels
4. Find top-k frequency indices
5. Convert frequencies to periods

# Example
```julia
x = randn(Float32, 16, 24, 1)  # batch=16, seq=24, channels=1
periods, weights = fft_for_period(x; k=3)
# periods might be [24, 12, 8] for dominant periods
```
"""
function fft_for_period(x::AbstractArray{T, 3}; k::Int=2) where T
    B, Time, C = size(x)

    # Apply real FFT along time dimension
    # Reshape for FFT: (B, T, C) -> (T, B*C)
    x_reshaped = reshape(permutedims(x, (2, 1, 3)), Time, B * C)

    # Compute FFT
    xf = rfft(x_reshaped, 1)  # FFT along first dimension

    # Compute amplitude spectrum
    amplitude = abs.(xf)  # (T/2+1, B*C)

    # Average across batch and channels
    freq_amplitude = vec(mean(amplitude, dims=2))  # (T/2+1,)

    # Zero out DC component (index 1)
    freq_amplitude[1] = 0

    # Find top-k frequencies
    # Get indices sorted by amplitude (descending)
    sorted_indices = sortperm(freq_amplitude, rev=true)

    # Take top-k indices (skip DC component if it appears)
    top_k_indices = Int[]
    for idx in sorted_indices
        if idx > 1 && length(top_k_indices) < k
            push!(top_k_indices, idx)
        end
        if length(top_k_indices) >= k
            break
        end
    end

    # Ensure we have k periods
    while length(top_k_indices) < k
        push!(top_k_indices, 2)  # Default to period = Time
    end

    # Convert frequency indices to periods
    # frequency = index - 1 (since index 1 is DC)
    # period = Time / frequency
    periods = Int[]
    for idx in top_k_indices
        freq = idx - 1
        if freq > 0
            period = max(2, Time ÷ freq)
        else
            period = Time
        end
        push!(periods, period)
    end

    # Get weights for these periods (normalized amplitudes)
    weights_for_batch = zeros(T, B, k)
    for b in 1:B
        for (i, idx) in enumerate(top_k_indices)
            # Get amplitude for this batch across all channels
            amp = 0.0
            for c in 1:C
                linear_idx = (b - 1) * C + c
                amp += amplitude[idx, linear_idx]
            end
            weights_for_batch[b, i] = T(amp / C)
        end
    end

    return (periods, weights_for_batch)
end

"""
    fft_for_period_batch(x::AbstractArray{T, 3}; k::Int=2) where T

Optimized FFT period detection for batched processing.

Same as fft_for_period but with batch-level optimization.
"""
function fft_for_period_batch(x::AbstractArray{T, 3}; k::Int=2) where T
    B, Time, C = size(x)

    # Pre-allocate result arrays
    periods = zeros(Int, k)
    period_weights = zeros(T, B, k)

    # Compute FFT for each sample
    for b in 1:B
        sample = x[b, :, :]  # (T, C)

        # FFT along time
        sample_fft = rfft(sample, 1)
        amplitude = abs.(sample_fft)

        # Average across channels
        freq_amp = vec(mean(amplitude, dims=2))
        freq_amp[1] = 0  # Zero DC

        # Find top-k
        sorted_idx = sortperm(freq_amp, rev=true)

        for i in 1:k
            idx = sorted_idx[i]
            if idx > 1
                freq = idx - 1
                period_weights[b, i] = freq_amp[idx]
                if b == 1
                    periods[i] = max(2, Time ÷ freq)
                end
            else
                period_weights[b, i] = 0
                if b == 1
                    periods[i] = Time
                end
            end
        end
    end

    return (periods, period_weights)
end

# ============================================================================
# Period-based Reshaping
# ============================================================================

"""
    reshape_for_2d(x::AbstractArray{T, 3}, period::Int) where T

Reshape 1D time series into 2D representation based on detected period.

This is the key operation in TimesNet that transforms temporal variation
into 2D variation that can be processed by 2D convolutions.

# Arguments
- `x::AbstractArray{T, 3}`: Input tensor (batch_size × seq_len × channels)
- `period::Int`: Detected period for reshaping

# Returns
- Reshaped tensor suitable for 2D convolution

# Example
For a time series of length 24 with period 6:
- Input: (B, 24, C)
- Reshaped: (B, 4, 6, C) = (B, num_periods, period_length, C)
- After permute for conv: (6, 4, C, B) = (W, H, C, B)
"""
function reshape_for_2d(x::AbstractArray{T, 3}, period::Int) where T
    B, Time, C = size(x)

    # Pad if necessary
    if Time % period != 0
        pad_length = period - (Time % period)
        padding = zeros(T, B, pad_length, C)
        x = cat(x, padding, dims=2)
        Time = size(x, 2)
    end

    # Calculate new dimensions
    num_periods = Time ÷ period

    # Reshape: (B, T, C) -> (B, num_periods, period, C)
    reshaped = reshape(x, B, num_periods, period, C)

    # Permute for 2D conv: (B, H, W, C) -> (W, H, C, B) for Flux Conv2D
    # Flux expects (width, height, channels, batch)
    return permutedims(reshaped, (3, 2, 4, 1))  # (period, num_periods, C, B)
end

"""
    reshape_from_2d(x::AbstractArray{T, 4}, original_len::Int) where T

Reshape 2D representation back to 1D time series.

# Arguments
- `x::AbstractArray{T, 4}`: 2D tensor (W × H × C × B) from conv output
- `original_len::Int`: Original sequence length

# Returns
- Reshaped tensor (batch_size × seq_len × channels)
"""
function reshape_from_2d(x::AbstractArray{T, 4}, original_len::Int) where T
    W, H, C, B = size(x)

    # Permute back: (W, H, C, B) -> (B, H, W, C)
    permuted = permutedims(x, (4, 2, 1, 3))

    # Reshape: (B, H, W, C) -> (B, H*W, C)
    reshaped = reshape(permuted, B, H * W, C)

    # Trim to original length
    return reshaped[:, 1:original_len, :]
end

# ============================================================================
# Adaptive Period Aggregation
# ============================================================================

"""
    aggregate_periods(outputs::Vector{<:AbstractArray}, weights::AbstractMatrix)

Aggregate outputs from different period-based representations using learned weights.

# Arguments
- `outputs::Vector`: List of k outputs from different periods
- `weights::AbstractMatrix`: Softmax weights for aggregation (B × k)

# Returns
- Aggregated output tensor
"""
function aggregate_periods(outputs::Vector{<:AbstractArray{T}}, weights::AbstractMatrix{T}) where T
    k = length(outputs)
    B, Time, C = size(outputs[1])

    # Apply softmax to weights
    weights_softmax = softmax(weights, dims=2)  # (B, k)

    # Weighted sum
    result = zeros(T, B, Time, C)
    for i in 1:k
        weight = reshape(weights_softmax[:, i], B, 1, 1)  # (B, 1, 1)
        result .+= outputs[i] .* weight
    end

    return result
end

# ============================================================================
# Spectral Analysis Utilities
# ============================================================================

"""
    compute_power_spectrum(x::AbstractVector)

Compute power spectral density of a time series.

# Arguments
- `x::AbstractVector`: Input time series

# Returns
- Tuple of (frequencies, power)
"""
function compute_power_spectrum(x::AbstractVector)
    n = length(x)
    xf = rfft(x)
    power = abs.(xf).^2 ./ n

    # Frequency bins
    freq = collect(0:length(xf)-1) ./ n

    return (freq, power)
end

"""
    find_dominant_frequencies(x::AbstractVector; k::Int=3)

Find the k dominant frequencies in a time series.

# Arguments
- `x::AbstractVector`: Input time series
- `k::Int`: Number of frequencies to find

# Returns
- Vector of (frequency, power) tuples for top-k frequencies
"""
function find_dominant_frequencies(x::AbstractVector; k::Int=3)
    freq, power = compute_power_spectrum(x)

    # Exclude DC component
    power[1] = 0

    # Find top-k
    sorted_idx = sortperm(power, rev=true)[1:min(k, length(power))]

    return [(freq[i], power[i]) for i in sorted_idx]
end

"""
    estimate_seasonality(x::AbstractVector)

Estimate the dominant seasonal period in a time series.

# Arguments
- `x::AbstractVector`: Input time series

# Returns
- `Int`: Estimated seasonal period (in number of time steps)
"""
function estimate_seasonality(x::AbstractVector)
    n = length(x)
    freq, power = compute_power_spectrum(x)

    # Exclude DC and very high frequencies
    power[1] = 0
    if length(power) > n ÷ 2
        power[n÷2:end] .= 0
    end

    # Find dominant frequency
    max_idx = argmax(power)
    dominant_freq = freq[max_idx]

    # Convert to period
    if dominant_freq > 0
        period = round(Int, 1 / dominant_freq)
    else
        period = n
    end

    return clamp(period, 2, n)
end
