"""
    ParcelFarmlandClassification

A comprehensive Julia implementation of the research paper:
"Parcel-Level Farmland Classification via Deep Heterogeneous Feature Integration
and Iterative Self-Optimization"

This module provides:
- Time series data loading and preprocessing for satellite imagery
- Deep heterogeneous feature extraction (spectral indices: NDVI, MNDWI, NDVIRE1)
- Multiple deep learning models: TimesNet, Transformer, LSTM+Attention
- Iterative self-optimization for semi-supervised learning
- Comprehensive evaluation metrics and visualization

Author: Implementation based on the research paper methodology
Julia Version: 1.12.3
Date: December 2025

# Key Features:
1. Multi-temporal satellite image time series processing
2. Deep learning classification with TimesNet architecture
3. Heterogeneous feature integration from spectral bands
4. Self-training with pseudo-label optimization
5. Parcel-level classification with field boundary integration

# Usage:
```julia
using ParcelFarmlandClassification

# Load and preprocess data
data = load_farmland_data("path/to/data")

# Train model with iterative self-optimization
model = train_timesnet(data; epochs=100, lr=0.0001)

# Evaluate and visualize results
results = evaluate_model(model, data.test)
plot_confusion_matrix(results)
```
"""
module ParcelFarmlandClassification

# ============================================================================
# Standard Library Imports
# ============================================================================
using Random
using Statistics
using LinearAlgebra
using Printf
using Dates

# ============================================================================
# External Package Imports
# ============================================================================
using Flux                      # Deep learning framework
using Flux: Chain, Dense, relu, gelu, softmax, crossentropy
using Flux: params, gradient, Optimise, Adam, ClipNorm
using Flux: DataLoader, onehotbatch, onecold
using CUDA                      # GPU acceleration (optional)
using NNlib                     # Neural network primitives
using Zygote                    # Automatic differentiation
using FFTW                      # Fast Fourier Transform for period detection
using CSV                       # Data loading
using DataFrames               # Data manipulation
using ProgressMeter            # Progress visualization
using Plots                     # Visualization
using StatsPlots               # Statistical plots
using Colors                    # Color handling for visualizations
using MLUtils                   # Machine learning utilities
using StatsBase                 # Statistical functions
using JSON3                     # Configuration handling

# ============================================================================
# Module Exports
# ============================================================================
export
    # Data loading and preprocessing
    FarmlandDataset,
    TimeSeriesData,
    load_ts_file,
    load_farmland_data,
    preprocess_data,
    create_dataloaders,
    normalize_data,

    # Spectral indices computation
    compute_ndvi,
    compute_mndwi,
    compute_ndvire1,
    compute_spectral_indices,

    # Model architectures
    TimesBlock,
    TimesNet,
    TransformerEncoder,
    LSTMAttention,
    Classifier,
    create_model,

    # Layers and components
    DataEmbedding,
    PositionalEncoding,
    MultiHeadAttention,
    InceptionBlock,
    FFTForPeriod,

    # Training and optimization
    train_model!,
    iterative_self_optimization!,
    evaluate_model,
    predict,

    # Evaluation metrics
    accuracy,
    f1_score,
    precision,
    recall,
    confusion_matrix,
    classification_report,

    # Visualization
    plot_confusion_matrix,
    plot_time_series,
    plot_training_curves,
    plot_spectral_indices,
    plot_classification_map,

    # Configuration
    Config,
    default_config

# ============================================================================
# Include Submodules
# ============================================================================
include("utils/config.jl")
include("utils/metrics.jl")
include("utils/helpers.jl")
include("data/dataset.jl")
include("data/preprocessing.jl")
include("data/spectral_indices.jl")
include("layers/embedding.jl")
include("layers/attention.jl")
include("layers/inception.jl")
include("layers/fft_period.jl")
include("models/timesnet.jl")
include("models/transformer.jl")
include("models/lstm_attention.jl")
include("models/classifier.jl")
include("experiments/training.jl")
include("experiments/self_optimization.jl")
include("experiments/evaluation.jl")
include("visualization/plots.jl")
include("visualization/confusion_matrix.jl")

# ============================================================================
# Module Initialization
# ============================================================================
function __init__()
    # Check CUDA availability
    if CUDA.functional()
        @info "CUDA is available. GPU acceleration enabled."
    else
        @info "CUDA is not available. Running on CPU."
    end

    # Set random seed for reproducibility
    Random.seed!(2021)
end

end # module ParcelFarmlandClassification
