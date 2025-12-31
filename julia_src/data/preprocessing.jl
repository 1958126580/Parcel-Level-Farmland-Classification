"""
    Preprocessing module for Parcel-Level Farmland Classification

This module provides data preprocessing functions for satellite image
time series (SITS) data, including:
- Normalization (z-score, min-max, per-sample)
- Missing value interpolation
- Smoothing (Savitzky-Golay)
- Temporal alignment
- Data augmentation
"""

# ============================================================================
# Normalization
# ============================================================================

"""
    Normalizer

Data normalizer supporting multiple normalization methods.

# Fields
- `method::Symbol`: Normalization method (:zscore, :minmax, :per_sample)
- `mean::Union{Vector{Float32}, Nothing}`: Feature means (for zscore)
- `std::Union{Vector{Float32}, Nothing}`: Feature standard deviations
- `min_val::Union{Vector{Float32}, Nothing}`: Feature minimums (for minmax)
- `max_val::Union{Vector{Float32}, Nothing}`: Feature maximums
"""
mutable struct Normalizer
    method::Symbol
    mean::Union{Vector{Float32}, Nothing}
    std::Union{Vector{Float32}, Nothing}
    min_val::Union{Vector{Float32}, Nothing}
    max_val::Union{Vector{Float32}, Nothing}
    fitted::Bool

    function Normalizer(method::Symbol=:zscore)
        @assert method in [:zscore, :minmax, :per_sample, :per_sample_minmax] "Unknown normalization method: $method"
        new(method, nothing, nothing, nothing, nothing, false)
    end
end

"""
    fit!(normalizer::Normalizer, data::AbstractArray{Float32, 3})

Fit normalizer on training data.

# Arguments
- `normalizer::Normalizer`: Normalizer to fit
- `data::AbstractArray{Float32, 3}`: Training data (N × T × C)
"""
function fit!(normalizer::Normalizer, data::AbstractArray{Float32, 3})
    N, T, C = size(data)

    if normalizer.method == :zscore
        # Calculate global mean and std per feature
        normalizer.mean = zeros(Float32, C)
        normalizer.std = zeros(Float32, C)

        for c in 1:C
            feature_data = vec(data[:, :, c])
            normalizer.mean[c] = mean(feature_data)
            normalizer.std[c] = std(feature_data)
            # Avoid division by zero
            if normalizer.std[c] < 1e-6
                normalizer.std[c] = 1.0f0
            end
        end

    elseif normalizer.method == :minmax
        # Calculate global min and max per feature
        normalizer.min_val = zeros(Float32, C)
        normalizer.max_val = zeros(Float32, C)

        for c in 1:C
            feature_data = vec(data[:, :, c])
            normalizer.min_val[c] = minimum(feature_data)
            normalizer.max_val[c] = maximum(feature_data)
            # Avoid division by zero
            if normalizer.max_val[c] - normalizer.min_val[c] < 1e-6
                normalizer.max_val[c] = normalizer.min_val[c] + 1.0f0
            end
        end
    end
    # For per_sample methods, no global fitting needed

    normalizer.fitted = true
end

"""
    transform(normalizer::Normalizer, data::AbstractArray{Float32, 3})

Apply normalization to data.

# Arguments
- `normalizer::Normalizer`: Fitted normalizer
- `data::AbstractArray{Float32, 3}`: Data to normalize (N × T × C)

# Returns
- `AbstractArray{Float32, 3}`: Normalized data
"""
function transform(normalizer::Normalizer, data::AbstractArray{Float32, 3})
    N, T, C = size(data)
    normalized = copy(data)

    if normalizer.method == :zscore
        @assert normalizer.fitted "Normalizer must be fitted before transform"
        for c in 1:C
            normalized[:, :, c] .= (data[:, :, c] .- normalizer.mean[c]) ./ normalizer.std[c]
        end

    elseif normalizer.method == :minmax
        @assert normalizer.fitted "Normalizer must be fitted before transform"
        for c in 1:C
            range = normalizer.max_val[c] - normalizer.min_val[c]
            normalized[:, :, c] .= (data[:, :, c] .- normalizer.min_val[c]) ./ range
        end

    elseif normalizer.method == :per_sample
        # Z-score normalization per sample
        for n in 1:N
            sample = data[n, :, :]
            sample_mean = mean(sample)
            sample_std = std(sample)
            if sample_std < 1e-6
                sample_std = 1.0f0
            end
            normalized[n, :, :] .= (sample .- sample_mean) ./ sample_std
        end

    elseif normalizer.method == :per_sample_minmax
        # Min-max normalization per sample
        for n in 1:N
            sample = data[n, :, :]
            sample_min = minimum(sample)
            sample_max = maximum(sample)
            range = sample_max - sample_min
            if range < 1e-6
                range = 1.0f0
            end
            normalized[n, :, :] .= (sample .- sample_min) ./ range
        end
    end

    return normalized
end

"""
    fit_transform!(normalizer::Normalizer, data::AbstractArray{Float32, 3})

Fit normalizer and transform data in one step.

# Arguments
- `normalizer::Normalizer`: Normalizer to fit
- `data::AbstractArray{Float32, 3}`: Data to normalize (N × T × C)

# Returns
- `AbstractArray{Float32, 3}`: Normalized data
"""
function fit_transform!(normalizer::Normalizer, data::AbstractArray{Float32, 3})
    fit!(normalizer, data)
    return transform(normalizer, data)
end

"""
    normalize_data(data::AbstractArray, method::Symbol=:zscore)

Convenience function for quick normalization.

# Arguments
- `data::AbstractArray`: Data to normalize
- `method::Symbol`: Normalization method

# Returns
- Normalized data
"""
function normalize_data(data::AbstractArray; method::Symbol=:zscore)
    data_f32 = Float32.(data)
    normalizer = Normalizer(method)
    return fit_transform!(normalizer, data_f32)
end

# ============================================================================
# Missing Value Handling
# ============================================================================

"""
    interpolate_missing(data::AbstractArray{Float32, 3})

Interpolate missing values (NaN) using linear interpolation.

# Arguments
- `data::AbstractArray{Float32, 3}`: Data with potential NaN values (N × T × C)

# Returns
- `AbstractArray{Float32, 3}`: Data with interpolated values
"""
function interpolate_missing(data::AbstractArray{Float32, 3})
    N, T, C = size(data)
    result = copy(data)

    for n in 1:N
        for c in 1:C
            series = result[n, :, c]

            # Find NaN positions
            nan_mask = isnan.(series)

            if any(nan_mask)
                # Get valid positions and values
                valid_idx = findall(.!nan_mask)

                if length(valid_idx) >= 2
                    # Linear interpolation
                    for t in 1:T
                        if nan_mask[t]
                            # Find nearest valid neighbors
                            left_idx = findlast(valid_idx .< t)
                            right_idx = findfirst(valid_idx .> t)

                            if left_idx === nothing && right_idx !== nothing
                                # Extrapolate from right
                                result[n, t, c] = series[valid_idx[right_idx]]
                            elseif right_idx === nothing && left_idx !== nothing
                                # Extrapolate from left
                                result[n, t, c] = series[valid_idx[left_idx]]
                            elseif left_idx !== nothing && right_idx !== nothing
                                # Interpolate
                                t1, t2 = valid_idx[left_idx], valid_idx[right_idx]
                                v1, v2 = series[t1], series[t2]
                                result[n, t, c] = v1 + (v2 - v1) * (t - t1) / (t2 - t1)
                            end
                        end
                    end
                elseif length(valid_idx) == 1
                    # Fill with single valid value
                    result[n, nan_mask, c] .= series[valid_idx[1]]
                else
                    # All NaN - fill with zero
                    result[n, nan_mask, c] .= 0.0f0
                end
            end
        end
    end

    return result
end

"""
    fill_gaps!(data::AbstractArray{Float32, 3}; value::Float32=0.0f0)

Fill NaN values with a constant.

# Arguments
- `data::AbstractArray{Float32, 3}`: Data with potential NaN values
- `value::Float32`: Value to fill NaN positions

# Returns
- Modified data (in-place)
"""
function fill_gaps!(data::AbstractArray{Float32, 3}; value::Float32=0.0f0)
    data[isnan.(data)] .= value
    return data
end

# ============================================================================
# Smoothing
# ============================================================================

"""
    savitzky_golay(data::Vector{Float32}, window_size::Int, poly_order::Int)

Apply Savitzky-Golay smoothing filter.

# Arguments
- `data::Vector{Float32}`: 1D time series
- `window_size::Int`: Window size (must be odd)
- `poly_order::Int`: Polynomial order

# Returns
- `Vector{Float32}`: Smoothed time series
"""
function savitzky_golay(data::Vector{Float32}, window_size::Int, poly_order::Int)
    n = length(data)

    # Ensure window size is odd
    if iseven(window_size)
        window_size += 1
    end

    half_window = (window_size - 1) ÷ 2
    @assert poly_order < window_size "Polynomial order must be less than window size"

    # Create Vandermonde matrix for polynomial fitting
    J = Float64.(-half_window:half_window)
    A = zeros(window_size, poly_order + 1)
    for i in 0:poly_order
        A[:, i+1] = J .^ i
    end

    # Calculate filter coefficients using pseudo-inverse
    coeffs = (A' * A) \ A'

    # Smoothing coefficients (for 0th derivative)
    smooth_coeffs = Float32.(coeffs[1, :])

    # Apply filter
    result = copy(data)
    for i in (half_window + 1):(n - half_window)
        result[i] = dot(smooth_coeffs, data[(i-half_window):(i+half_window)])
    end

    return result
end

"""
    smooth_data(data::AbstractArray{Float32, 3}; window_size::Int=5, poly_order::Int=2)

Apply Savitzky-Golay smoothing to all time series.

# Arguments
- `data::AbstractArray{Float32, 3}`: Data to smooth (N × T × C)
- `window_size::Int`: Smoothing window size
- `poly_order::Int`: Polynomial order

# Returns
- `AbstractArray{Float32, 3}`: Smoothed data
"""
function smooth_data(data::AbstractArray{Float32, 3}; window_size::Int=5, poly_order::Int=2)
    N, T, C = size(data)
    result = copy(data)

    for n in 1:N
        for c in 1:C
            result[n, :, c] = savitzky_golay(data[n, :, c], window_size, poly_order)
        end
    end

    return result
end

# ============================================================================
# Temporal Alignment
# ============================================================================

"""
    resample_to_length(data::AbstractArray{Float32, 3}, target_length::Int)

Resample time series to a fixed length using linear interpolation.

# Arguments
- `data::AbstractArray{Float32, 3}`: Data to resample (N × T × C)
- `target_length::Int`: Target sequence length

# Returns
- `AbstractArray{Float32, 3}`: Resampled data (N × target_length × C)
"""
function resample_to_length(data::AbstractArray{Float32, 3}, target_length::Int)
    N, T, C = size(data)
    result = zeros(Float32, N, target_length, C)

    # Source and target time indices
    src_t = range(0, 1, length=T)
    dst_t = range(0, 1, length=target_length)

    for n in 1:N
        for c in 1:C
            # Linear interpolation
            for (i, t) in enumerate(dst_t)
                # Find surrounding source indices
                idx = searchsortedlast(src_t, t)
                if idx == 0
                    result[n, i, c] = data[n, 1, c]
                elseif idx >= T
                    result[n, i, c] = data[n, T, c]
                else
                    # Linear interpolation
                    t1, t2 = src_t[idx], src_t[idx+1]
                    v1, v2 = data[n, idx, c], data[n, idx+1, c]
                    result[n, i, c] = v1 + (v2 - v1) * (t - t1) / (t2 - t1)
                end
            end
        end
    end

    return result
end

# ============================================================================
# Data Augmentation
# ============================================================================

"""
    DataAugmentation

Configuration for data augmentation.

# Fields
- `jitter::Float32`: Random noise magnitude
- `scaling::Float32`: Random scaling range
- `time_warp::Float32`: Time warping magnitude
- `magnitude_warp::Float32`: Magnitude warping range
"""
struct DataAugmentation
    jitter::Float32
    scaling::Float32
    time_warp::Float32
    magnitude_warp::Float32

    function DataAugmentation(; jitter::Float32=0.01f0, scaling::Float32=0.1f0,
                               time_warp::Float32=0.1f0, magnitude_warp::Float32=0.1f0)
        new(jitter, scaling, time_warp, magnitude_warp)
    end
end

"""
    augment_sample(data::Matrix{Float32}, aug::DataAugmentation)

Apply data augmentation to a single sample.

# Arguments
- `data::Matrix{Float32}`: Single time series (T × C)
- `aug::DataAugmentation`: Augmentation configuration

# Returns
- `Matrix{Float32}`: Augmented time series
"""
function augment_sample(data::Matrix{Float32}, aug::DataAugmentation)
    T, C = size(data)
    result = copy(data)

    # Jittering: add random noise
    if aug.jitter > 0
        noise = aug.jitter * randn(Float32, T, C)
        result .+= noise
    end

    # Scaling: random amplitude scaling
    if aug.scaling > 0
        scale = 1.0f0 + aug.scaling * (2 * rand(Float32) - 1)
        result .*= scale
    end

    # Magnitude warping: smooth random amplitude modulation
    if aug.magnitude_warp > 0
        # Create smooth warping curve using cubic spline
        n_knots = 4
        knot_values = 1.0f0 .+ aug.magnitude_warp * (2 * rand(Float32, n_knots) .- 1)

        # Interpolate to full length
        warp_curve = zeros(Float32, T)
        for t in 1:T
            pos = (t - 1) / (T - 1) * (n_knots - 1) + 1
            idx = floor(Int, pos)
            idx = clamp(idx, 1, n_knots - 1)
            frac = pos - idx
            warp_curve[t] = knot_values[idx] * (1 - frac) + knot_values[idx+1] * frac
        end

        result .*= warp_curve
    end

    return result
end

"""
    augment_dataset(data::AbstractArray{Float32, 3}, labels::Vector{Int},
                    aug::DataAugmentation; n_augments::Int=1)

Augment entire dataset.

# Arguments
- `data::AbstractArray{Float32, 3}`: Original data (N × T × C)
- `labels::Vector{Int}`: Original labels
- `aug::DataAugmentation`: Augmentation configuration
- `n_augments::Int`: Number of augmented copies per sample

# Returns
- Tuple of (augmented_data, augmented_labels)
"""
function augment_dataset(data::AbstractArray{Float32, 3}, labels::Vector{Int},
                         aug::DataAugmentation; n_augments::Int=1)
    N, T, C = size(data)

    # Original data plus augmented versions
    total_samples = N * (1 + n_augments)
    aug_data = zeros(Float32, total_samples, T, C)
    aug_labels = zeros(Int, total_samples)

    idx = 1
    for n in 1:N
        # Add original
        aug_data[idx, :, :] = data[n, :, :]
        aug_labels[idx] = labels[n]
        idx += 1

        # Add augmented versions
        for _ in 1:n_augments
            aug_data[idx, :, :] = augment_sample(data[n, :, :], aug)
            aug_labels[idx] = labels[n]
            idx += 1
        end
    end

    return (aug_data, aug_labels)
end

# ============================================================================
# Complete Preprocessing Pipeline
# ============================================================================

"""
    preprocess_data(data::AbstractArray{Float32, 3};
                    normalize::Bool=true,
                    norm_method::Symbol=:zscore,
                    interpolate::Bool=true,
                    smooth::Bool=false,
                    smooth_window::Int=5)

Apply complete preprocessing pipeline.

# Arguments
- `data::AbstractArray{Float32, 3}`: Raw data (N × T × C)
- `normalize::Bool`: Whether to normalize
- `norm_method::Symbol`: Normalization method
- `interpolate::Bool`: Whether to interpolate missing values
- `smooth::Bool`: Whether to apply smoothing
- `smooth_window::Int`: Smoothing window size

# Returns
- `AbstractArray{Float32, 3}`: Preprocessed data
"""
function preprocess_data(data::AbstractArray{Float32, 3};
                         normalize::Bool=true,
                         norm_method::Symbol=:zscore,
                         interpolate::Bool=true,
                         smooth::Bool=false,
                         smooth_window::Int=5)
    result = copy(data)

    # Interpolate missing values
    if interpolate
        result = interpolate_missing(result)
    end

    # Apply smoothing
    if smooth
        result = smooth_data(result; window_size=smooth_window)
    end

    # Normalize
    if normalize
        result = normalize_data(result; method=norm_method)
    end

    return result
end
