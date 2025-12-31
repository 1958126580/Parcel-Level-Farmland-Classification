"""
    Spectral Indices module for Parcel-Level Farmland Classification

This module computes vegetation and water indices from multispectral
satellite imagery (Sentinel-2 bands). These spectral indices are critical
for heterogeneous feature integration in farmland classification.

Sentinel-2 Bands Used:
- B2 (Blue): 490 nm
- B3 (Green): 560 nm
- B4 (Red): 665 nm
- B5 (Red Edge 1): 705 nm
- B6 (Red Edge 2): 740 nm
- B7 (Red Edge 3): 783 nm
- B8 (NIR): 842 nm
- B8A (Narrow NIR): 865 nm
- B11 (SWIR 1): 1610 nm
- B12 (SWIR 2): 2190 nm

Key Indices:
- NDVI: Normalized Difference Vegetation Index
- MNDWI: Modified Normalized Difference Water Index
- NDVIRE1: Normalized Difference Vegetation Index (Red Edge 1)
- EVI: Enhanced Vegetation Index
- SAVI: Soil Adjusted Vegetation Index
"""

# ============================================================================
# Band Indices
# ============================================================================

# Sentinel-2 band indices (1-indexed for Julia)
const BAND_BLUE = 1     # B2 - 490nm
const BAND_GREEN = 2    # B3 - 560nm
const BAND_RED = 3      # B4 - 665nm
const BAND_RE1 = 4      # B5 - Red Edge 1 (705nm)
const BAND_RE2 = 5      # B6 - Red Edge 2 (740nm)
const BAND_RE3 = 6      # B7 - Red Edge 3 (783nm)
const BAND_NIR = 7      # B8 - NIR (842nm)
const BAND_NNIR = 8     # B8A - Narrow NIR (865nm)
const BAND_SWIR1 = 9    # B11 - SWIR1 (1610nm)
const BAND_SWIR2 = 10   # B12 - SWIR2 (2190nm)

# ============================================================================
# Vegetation Indices
# ============================================================================

"""
    compute_ndvi(red::AbstractArray, nir::AbstractArray)

Compute Normalized Difference Vegetation Index (NDVI).

NDVI = (NIR - Red) / (NIR + Red)

NDVI ranges from -1 to 1, where:
- Values close to 1 indicate dense vegetation
- Values around 0 indicate bare soil
- Negative values indicate water or snow

# Arguments
- `red::AbstractArray`: Red band reflectance (B4)
- `nir::AbstractArray`: Near-infrared reflectance (B8)

# Returns
- `AbstractArray`: NDVI values

# Example
```julia
ndvi = compute_ndvi(red_band, nir_band)
```
"""
function compute_ndvi(red::AbstractArray, nir::AbstractArray)
    # Avoid division by zero
    denominator = nir .+ red
    denominator = map(x -> x == 0 ? eps(Float32) : x, denominator)
    return (nir .- red) ./ denominator
end

"""
    compute_ndvi(bands::AbstractArray{T, 3}; red_idx::Int=BAND_RED, nir_idx::Int=BAND_NIR) where T

Compute NDVI from multi-band array.

# Arguments
- `bands::AbstractArray{T, 3}`: Multi-band data (N × T × Bands) or (T × Bands)
- `red_idx::Int`: Index of red band
- `nir_idx::Int`: Index of NIR band

# Returns
- `AbstractArray`: NDVI time series
"""
function compute_ndvi(bands::AbstractArray{T, 3};
                      red_idx::Int=BAND_RED, nir_idx::Int=BAND_NIR) where T
    red = bands[:, :, red_idx]
    nir = bands[:, :, nir_idx]
    return compute_ndvi(red, nir)
end

"""
    compute_evi(blue::AbstractArray, red::AbstractArray, nir::AbstractArray;
                G::Float32=2.5f0, C1::Float32=6.0f0, C2::Float32=7.5f0, L::Float32=1.0f0)

Compute Enhanced Vegetation Index (EVI).

EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

EVI is less sensitive to atmospheric conditions and soil background
than NDVI, especially useful for dense vegetation.

# Arguments
- `blue::AbstractArray`: Blue band reflectance (B2)
- `red::AbstractArray`: Red band reflectance (B4)
- `nir::AbstractArray`: Near-infrared reflectance (B8)
- `G::Float32`: Gain factor (default: 2.5)
- `C1::Float32`: Aerosol coefficient 1 (default: 6.0)
- `C2::Float32`: Aerosol coefficient 2 (default: 7.5)
- `L::Float32`: Canopy background adjustment (default: 1.0)

# Returns
- `AbstractArray`: EVI values
"""
function compute_evi(blue::AbstractArray, red::AbstractArray, nir::AbstractArray;
                     G::Float32=2.5f0, C1::Float32=6.0f0, C2::Float32=7.5f0, L::Float32=1.0f0)
    denominator = nir .+ C1 .* red .- C2 .* blue .+ L
    denominator = map(x -> abs(x) < eps(Float32) ? eps(Float32) : x, denominator)
    return G .* (nir .- red) ./ denominator
end

"""
    compute_savi(red::AbstractArray, nir::AbstractArray; L::Float32=0.5f0)

Compute Soil Adjusted Vegetation Index (SAVI).

SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)

SAVI minimizes soil brightness influences in areas with low vegetation cover.

# Arguments
- `red::AbstractArray`: Red band reflectance
- `nir::AbstractArray`: Near-infrared reflectance
- `L::Float32`: Soil brightness correction factor (default: 0.5)

# Returns
- `AbstractArray`: SAVI values
"""
function compute_savi(red::AbstractArray, nir::AbstractArray; L::Float32=0.5f0)
    denominator = nir .+ red .+ L
    denominator = map(x -> x == 0 ? eps(Float32) : x, denominator)
    return (nir .- red) ./ denominator .* (1 + L)
end

"""
    compute_ndvire1(re1::AbstractArray, nnir::AbstractArray)

Compute Normalized Difference Vegetation Index using Red Edge 1 band (NDVIRE1).

NDVIRE1 = (B8A - B5) / (B8A + B5)

This index is sensitive to chlorophyll content and is useful for
distinguishing between crop types with similar NDVI but different
chlorophyll levels.

# Arguments
- `re1::AbstractArray`: Red Edge 1 band reflectance (B5, 705nm)
- `nnir::AbstractArray`: Narrow NIR band reflectance (B8A, 865nm)

# Returns
- `AbstractArray`: NDVIRE1 values
"""
function compute_ndvire1(re1::AbstractArray, nnir::AbstractArray)
    denominator = nnir .+ re1
    denominator = map(x -> x == 0 ? eps(Float32) : x, denominator)
    return (nnir .- re1) ./ denominator
end

"""
    compute_ndvire2(re2::AbstractArray, nnir::AbstractArray)

Compute NDVI using Red Edge 2 band.

NDVIRE2 = (B8A - B6) / (B8A + B6)
"""
function compute_ndvire2(re2::AbstractArray, nnir::AbstractArray)
    denominator = nnir .+ re2
    denominator = map(x -> x == 0 ? eps(Float32) : x, denominator)
    return (nnir .- re2) ./ denominator
end

"""
    compute_ndre(re1::AbstractArray, nir::AbstractArray)

Compute Normalized Difference Red Edge (NDRE).

NDRE = (NIR - RE1) / (NIR + RE1)

NDRE is particularly useful for assessing crop health and nitrogen
content in late-stage crops.
"""
function compute_ndre(re1::AbstractArray, nir::AbstractArray)
    denominator = nir .+ re1
    denominator = map(x -> x == 0 ? eps(Float32) : x, denominator)
    return (nir .- re1) ./ denominator
end

# ============================================================================
# Water Indices
# ============================================================================

"""
    compute_mndwi(green::AbstractArray, swir::AbstractArray)

Compute Modified Normalized Difference Water Index (MNDWI).

MNDWI = (Green - SWIR) / (Green + SWIR)

MNDWI is used to detect water bodies and distinguish irrigated fields.

# Arguments
- `green::AbstractArray`: Green band reflectance (B3)
- `swir::AbstractArray`: SWIR band reflectance (B11)

# Returns
- `AbstractArray`: MNDWI values
"""
function compute_mndwi(green::AbstractArray, swir::AbstractArray)
    denominator = green .+ swir
    denominator = map(x -> x == 0 ? eps(Float32) : x, denominator)
    return (green .- swir) ./ denominator
end

"""
    compute_ndwi(green::AbstractArray, nir::AbstractArray)

Compute Normalized Difference Water Index (NDWI).

NDWI = (Green - NIR) / (Green + NIR)

NDWI is sensitive to vegetation water content.
"""
function compute_ndwi(green::AbstractArray, nir::AbstractArray)
    denominator = green .+ nir
    denominator = map(x -> x == 0 ? eps(Float32) : x, denominator)
    return (green .- nir) ./ denominator
end

"""
    compute_ndmi(nir::AbstractArray, swir::AbstractArray)

Compute Normalized Difference Moisture Index (NDMI).

NDMI = (NIR - SWIR) / (NIR + SWIR)

NDMI is sensitive to moisture levels in vegetation.
"""
function compute_ndmi(nir::AbstractArray, swir::AbstractArray)
    denominator = nir .+ swir
    denominator = map(x -> x == 0 ? eps(Float32) : x, denominator)
    return (nir .- swir) ./ denominator
end

# ============================================================================
# Composite Spectral Indices
# ============================================================================

"""
    compute_spectral_indices(bands::AbstractArray{T, 3}) where T

Compute multiple spectral indices from multi-band data.

This function extracts heterogeneous features by computing:
- NDVI: vegetation greenness
- MNDWI: water content / irrigation
- NDVIRE1: chlorophyll sensitivity

# Arguments
- `bands::AbstractArray{T, 3}`: Multi-band time series (N × T × Bands)
  Bands should be in order: Blue, Green, Red, RE1, RE2, RE3, NIR, NNIR, SWIR1

# Returns
- `AbstractArray`: Combined spectral indices (N × T × 3)
  Channels: [NDVI, MNDWI, NDVIRE1]

# Example
```julia
indices = compute_spectral_indices(multispectral_data)
# indices[:,:,1] = NDVI
# indices[:,:,2] = MNDWI
# indices[:,:,3] = NDVIRE1
```
"""
function compute_spectral_indices(bands::AbstractArray{T, 3}) where T
    N, Time, C = size(bands)

    # Ensure we have enough bands
    @assert C >= 9 "Need at least 9 bands for spectral index computation"

    # Compute indices
    ndvi = compute_ndvi(bands[:, :, BAND_RED], bands[:, :, BAND_NIR])
    mndwi = compute_mndwi(bands[:, :, BAND_GREEN], bands[:, :, BAND_SWIR1])
    ndvire1 = compute_ndvire1(bands[:, :, BAND_RE1], bands[:, :, BAND_NNIR])

    # Stack into single array
    result = zeros(Float32, N, Time, 3)
    result[:, :, 1] = Float32.(ndvi)
    result[:, :, 2] = Float32.(mndwi)
    result[:, :, 3] = Float32.(ndvire1)

    return result
end

"""
    compute_all_indices(bands::AbstractArray{T, 3}) where T

Compute comprehensive set of spectral indices.

Indices computed:
1. NDVI
2. EVI
3. SAVI
4. NDVIRE1
5. NDRE
6. MNDWI
7. NDMI

# Arguments
- `bands::AbstractArray{T, 3}`: Multi-band time series (N × T × Bands)

# Returns
- `AbstractArray`: All spectral indices (N × T × 7)
"""
function compute_all_indices(bands::AbstractArray{T, 3}) where T
    N, Time, C = size(bands)

    @assert C >= 9 "Need at least 9 bands"

    # Extract individual bands
    blue = bands[:, :, BAND_BLUE]
    green = bands[:, :, BAND_GREEN]
    red = bands[:, :, BAND_RED]
    re1 = bands[:, :, BAND_RE1]
    nir = bands[:, :, BAND_NIR]
    nnir = bands[:, :, BAND_NNIR]
    swir1 = bands[:, :, BAND_SWIR1]

    # Compute all indices
    ndvi = compute_ndvi(red, nir)
    evi = compute_evi(blue, red, nir)
    savi = compute_savi(red, nir)
    ndvire1 = compute_ndvire1(re1, nnir)
    ndre = compute_ndre(re1, nir)
    mndwi = compute_mndwi(green, swir1)
    ndmi = compute_ndmi(nir, swir1)

    # Stack results
    result = zeros(Float32, N, Time, 7)
    result[:, :, 1] = Float32.(ndvi)
    result[:, :, 2] = Float32.(evi)
    result[:, :, 3] = Float32.(savi)
    result[:, :, 4] = Float32.(ndvire1)
    result[:, :, 5] = Float32.(ndre)
    result[:, :, 6] = Float32.(mndwi)
    result[:, :, 7] = Float32.(ndmi)

    return result
end

# ============================================================================
# Heterogeneous Feature Integration
# ============================================================================

"""
    HeterogeneousFeatureExtractor

Configuration for heterogeneous feature extraction.

# Fields
- `indices::Vector{Symbol}`: Which indices to compute
- `include_raw_bands::Bool`: Whether to include raw spectral bands
- `selected_bands::Vector{Int}`: Which raw bands to include
"""
struct HeterogeneousFeatureExtractor
    indices::Vector{Symbol}
    include_raw_bands::Bool
    selected_bands::Vector{Int}

    function HeterogeneousFeatureExtractor(;
            indices::Vector{Symbol}=[:ndvi, :mndwi, :ndvire1],
            include_raw_bands::Bool=false,
            selected_bands::Vector{Int}=[BAND_RED, BAND_NIR, BAND_SWIR1])
        new(indices, include_raw_bands, selected_bands)
    end
end

"""
    extract_features(extractor::HeterogeneousFeatureExtractor,
                     bands::AbstractArray{T, 3}) where T

Extract heterogeneous features using the configured extractor.

# Arguments
- `extractor::HeterogeneousFeatureExtractor`: Feature extractor configuration
- `bands::AbstractArray{T, 3}`: Multi-band time series (N × T × Bands)

# Returns
- `AbstractArray`: Extracted features (N × T × num_features)
"""
function extract_features(extractor::HeterogeneousFeatureExtractor,
                          bands::AbstractArray{T, 3}) where T
    N, Time, C = size(bands)

    features = []

    # Compute requested spectral indices
    for idx_name in extractor.indices
        if idx_name == :ndvi
            push!(features, compute_ndvi(bands[:, :, BAND_RED], bands[:, :, BAND_NIR]))
        elseif idx_name == :evi
            push!(features, compute_evi(bands[:, :, BAND_BLUE],
                                        bands[:, :, BAND_RED],
                                        bands[:, :, BAND_NIR]))
        elseif idx_name == :savi
            push!(features, compute_savi(bands[:, :, BAND_RED], bands[:, :, BAND_NIR]))
        elseif idx_name == :ndvire1
            push!(features, compute_ndvire1(bands[:, :, BAND_RE1], bands[:, :, BAND_NNIR]))
        elseif idx_name == :ndre
            push!(features, compute_ndre(bands[:, :, BAND_RE1], bands[:, :, BAND_NIR]))
        elseif idx_name == :mndwi
            push!(features, compute_mndwi(bands[:, :, BAND_GREEN], bands[:, :, BAND_SWIR1]))
        elseif idx_name == :ndmi
            push!(features, compute_ndmi(bands[:, :, BAND_NIR], bands[:, :, BAND_SWIR1]))
        else
            @warn "Unknown index: $idx_name"
        end
    end

    # Include raw bands if requested
    if extractor.include_raw_bands
        for band_idx in extractor.selected_bands
            push!(features, bands[:, :, band_idx])
        end
    end

    # Stack all features
    num_features = length(features)
    result = zeros(Float32, N, Time, num_features)
    for (i, feat) in enumerate(features)
        result[:, :, i] = Float32.(feat)
    end

    return result
end

"""
    default_feature_extractor()

Create default heterogeneous feature extractor.

Uses NDVI, MNDWI, and NDVIRE1 as per the paper methodology.
"""
function default_feature_extractor()
    return HeterogeneousFeatureExtractor(
        indices=[:ndvi, :mndwi, :ndvire1],
        include_raw_bands=false
    )
end

"""
    comprehensive_feature_extractor()

Create comprehensive feature extractor with all available indices.
"""
function comprehensive_feature_extractor()
    return HeterogeneousFeatureExtractor(
        indices=[:ndvi, :evi, :savi, :ndvire1, :ndre, :mndwi, :ndmi],
        include_raw_bands=true,
        selected_bands=[BAND_NIR, BAND_RED, BAND_SWIR1]
    )
end
