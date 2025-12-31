"""
    Dataset module for Parcel-Level Farmland Classification

This module provides data structures and loading functions for
handling satellite image time series (SITS) data for farmland classification.

Supports:
- UEA Archive .ts format
- Custom CSV format
- Pixel-level and parcel-level data
"""

# ============================================================================
# Data Structures
# ============================================================================

"""
    TimeSeriesData

Structure representing a single time series sample.

# Fields
- `data::Matrix{Float32}`: Time series data (T × C) - time steps × channels
- `label::Int`: Class label
- `id::String`: Sample identifier
- `padding_mask::Vector{Bool}`: Mask for padded positions (true = valid data)
"""
struct TimeSeriesData
    data::Matrix{Float32}
    label::Int
    id::String
    padding_mask::Vector{Bool}

    function TimeSeriesData(data::AbstractMatrix, label::Int;
                            id::String="", padding_mask::Vector{Bool}=Bool[])
        T, C = size(data)
        if isempty(padding_mask)
            padding_mask = ones(Bool, T)
        end
        new(Float32.(data), label, id, padding_mask)
    end
end

"""
    FarmlandDataset

Dataset structure for farmland classification.

# Fields
- `samples::Vector{TimeSeriesData}`: Vector of time series samples
- `feature_dim::Int`: Number of features per time step
- `seq_len::Int`: Maximum sequence length
- `num_classes::Int`: Number of output classes
- `class_names::Vector{String}`: Names of each class
- `class_weights::Vector{Float32}`: Class weights for imbalanced data
"""
struct FarmlandDataset
    samples::Vector{TimeSeriesData}
    feature_dim::Int
    seq_len::Int
    num_classes::Int
    class_names::Vector{String}
    class_weights::Vector{Float32}

    function FarmlandDataset(samples::Vector{TimeSeriesData};
                             class_names::Vector{String}=String[])
        # Determine dimensions from data
        feature_dim = size(samples[1].data, 2)
        seq_len = maximum(size(s.data, 1) for s in samples)

        # Get unique labels
        labels = [s.label for s in samples]
        num_classes = maximum(labels)

        # Create default class names if not provided
        if isempty(class_names)
            class_names = ["Class $i" for i in 1:num_classes]
        end

        # Calculate class weights (inverse frequency)
        class_counts = [count(==(c), labels) for c in 1:num_classes]
        total = sum(class_counts)
        class_weights = Float32.([total / (num_classes * max(c, 1)) for c in class_counts])

        new(samples, feature_dim, seq_len, num_classes, class_names, class_weights)
    end
end

# ============================================================================
# UEA .ts File Loading
# ============================================================================

"""
    parse_ts_line(line::String)

Parse a single line from a .ts file.

# Arguments
- `line::String`: Data line from .ts file

# Returns
- Tuple of (data::Matrix{Float32}, label::Int)
"""
function parse_ts_line(line::String)
    # Split by colon to separate data from label
    parts = split(strip(line), ':')

    if length(parts) != 2
        error("Invalid .ts line format: expected data:label")
    end

    # Parse the label (handle floating point labels like 1.0)
    label = round(Int, parse(Float64, parts[2]))

    # Parse the time series data
    # Format: dimension1,dimension2,...
    # For univariate: val1,val2,val3,...
    # For multivariate: dim1_t1,dim1_t2,...:dim2_t1,dim2_t2,...

    data_str = parts[1]

    # Check if multivariate (contains colon in data part before label colon)
    # In our case, data is univariate with comma-separated values
    values = [parse(Float32, v) for v in split(data_str, ',')]

    # Reshape to (T, 1) for univariate data
    data = reshape(values, :, 1)

    return (data, label)
end

"""
    load_ts_file(filepath::String)

Load a UEA Archive .ts format file.

# Arguments
- `filepath::String`: Path to .ts file

# Returns
- `FarmlandDataset`: Loaded dataset

# Example
```julia
dataset = load_ts_file("WeiganFarmland_TRAIN.ts")
```
"""
function load_ts_file(filepath::String)
    @info "Loading .ts file: $filepath"

    samples = TimeSeriesData[]
    class_names = String[]
    seq_len = 0
    num_dimensions = 1

    open(filepath, "r") do f
        in_data = false
        sample_idx = 0

        for line in eachline(f)
            line = strip(line)

            # Skip empty lines and comments
            if isempty(line) || startswith(line, "#")
                continue
            end

            # Parse header information
            if startswith(line, "@")
                if startswith(line, "@problemName")
                    # Extract problem name
                elseif startswith(line, "@seriesLength")
                    seq_len = parse(Int, split(line)[2])
                elseif startswith(line, "@dimensions")
                    num_dimensions = parse(Int, split(line)[2])
                elseif startswith(line, "@classLabel")
                    # Parse class labels: @classLabel true 1.0 2.0 3.0 ...
                    parts = split(line)
                    if length(parts) > 2
                        class_names = ["Class $(round(Int, parse(Float64, p)))"
                                       for p in parts[3:end]]
                    end
                elseif line == "@data"
                    in_data = true
                end
            elseif in_data
                # Parse data line
                try
                    data, label = parse_ts_line(line)
                    sample_idx += 1
                    id = "sample_$sample_idx"

                    push!(samples, TimeSeriesData(data, label; id=id))
                catch e
                    @warn "Failed to parse line: $line" exception=e
                end
            end
        end
    end

    @info "Loaded $(length(samples)) samples with sequence length $seq_len"

    # Update class names based on actual data
    if isempty(class_names)
        max_label = maximum(s.label for s in samples)
        class_names = ["Cotton", "Corn", "Pepper", "Jujube",
                       "Pear", "Apricot", "Tomato", "Others"][1:max_label]
    end

    return FarmlandDataset(samples; class_names=class_names)
end

# ============================================================================
# CSV Data Loading
# ============================================================================

"""
    load_csv_data(filepath::String; label_col::String="label",
                  id_col::String="id", time_cols::Vector{String}=String[])

Load time series data from CSV file.

# Arguments
- `filepath::String`: Path to CSV file
- `label_col::String`: Name of label column
- `id_col::String`: Name of ID column
- `time_cols::Vector{String}`: Names of time series columns (in order)

# Returns
- `FarmlandDataset`: Loaded dataset
"""
function load_csv_data(filepath::String; label_col::String="label",
                       id_col::String="id", time_cols::Vector{String}=String[])
    @info "Loading CSV file: $filepath"

    df = CSV.read(filepath, DataFrame)

    samples = TimeSeriesData[]

    # If time_cols not specified, use all numeric columns except label and id
    if isempty(time_cols)
        exclude_cols = [label_col, id_col]
        time_cols = [String(c) for c in names(df)
                     if !(String(c) in exclude_cols) && eltype(df[!, c]) <: Number]
    end

    for row in eachrow(df)
        # Extract time series data
        data = Float32[row[Symbol(c)] for c in time_cols]
        data = reshape(data, :, 1)  # Reshape to (T, 1)

        # Get label and ID
        label = Int(row[Symbol(label_col)])
        id = haskey(row, Symbol(id_col)) ? string(row[Symbol(id_col)]) : ""

        push!(samples, TimeSeriesData(data, label; id=id))
    end

    @info "Loaded $(length(samples)) samples"
    return FarmlandDataset(samples)
end

# ============================================================================
# Data Access
# ============================================================================

"""
    Base.length(dataset::FarmlandDataset)

Get the number of samples in the dataset.
"""
Base.length(dataset::FarmlandDataset) = length(dataset.samples)

"""
    Base.getindex(dataset::FarmlandDataset, idx::Int)

Get a sample by index.
"""
Base.getindex(dataset::FarmlandDataset, idx::Int) = dataset.samples[idx]

"""
    Base.getindex(dataset::FarmlandDataset, indices::AbstractVector{Int})

Get multiple samples by indices.
"""
function Base.getindex(dataset::FarmlandDataset, indices::AbstractVector{Int})
    return FarmlandDataset(dataset.samples[indices];
                           class_names=dataset.class_names)
end

"""
    get_batch(dataset::FarmlandDataset, indices::AbstractVector{Int})

Get a batch of data for training.

# Arguments
- `dataset::FarmlandDataset`: Dataset to sample from
- `indices::AbstractVector{Int}`: Indices of samples to include

# Returns
- Tuple of (data::Array{Float32,3}, labels::Vector{Int}, masks::Matrix{Bool})
  - data: (batch_size, seq_len, feature_dim)
  - labels: (batch_size,)
  - masks: (batch_size, seq_len)
"""
function get_batch(dataset::FarmlandDataset, indices::AbstractVector{Int})
    batch_size = length(indices)
    seq_len = dataset.seq_len
    feature_dim = dataset.feature_dim

    # Initialize arrays
    data = zeros(Float32, batch_size, seq_len, feature_dim)
    labels = zeros(Int, batch_size)
    masks = zeros(Bool, batch_size, seq_len)

    for (i, idx) in enumerate(indices)
        sample = dataset.samples[idx]
        sample_len = size(sample.data, 1)

        # Copy data (pad if necessary)
        copy_len = min(sample_len, seq_len)
        data[i, 1:copy_len, :] = sample.data[1:copy_len, :]

        # Set label
        labels[i] = sample.label

        # Set mask
        masks[i, 1:copy_len] .= true
    end

    return (data, labels, masks)
end

"""
    iterate(dataset::FarmlandDataset, state=1)

Enable iteration over dataset.
"""
function Base.iterate(dataset::FarmlandDataset, state=1)
    if state > length(dataset)
        return nothing
    else
        return (dataset[state], state + 1)
    end
end

# ============================================================================
# DataLoader Creation
# ============================================================================

"""
    create_dataloaders(train_data::FarmlandDataset, test_data::FarmlandDataset;
                       batch_size::Int=16, shuffle_train::Bool=true)

Create training and test data loaders.

# Arguments
- `train_data::FarmlandDataset`: Training dataset
- `test_data::FarmlandDataset`: Test dataset
- `batch_size::Int`: Batch size
- `shuffle_train::Bool`: Whether to shuffle training data

# Returns
- Tuple of (train_loader, test_loader)
"""
function create_dataloaders(train_data::FarmlandDataset, test_data::FarmlandDataset;
                            batch_size::Int=16, shuffle_train::Bool=true)

    # Get all training data as arrays
    train_indices = collect(1:length(train_data))
    test_indices = collect(1:length(test_data))

    train_x, train_y, train_masks = get_batch(train_data, train_indices)
    test_x, test_y, test_masks = get_batch(test_data, test_indices)

    # Create data loaders
    train_loader = DataLoader((train_x, train_y, train_masks);
                              batchsize=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader((test_x, test_y, test_masks);
                             batchsize=batch_size, shuffle=false)

    return (train_loader, test_loader)
end

# ============================================================================
# Data Statistics
# ============================================================================

"""
    dataset_summary(dataset::FarmlandDataset)

Print summary statistics of the dataset.

# Arguments
- `dataset::FarmlandDataset`: Dataset to summarize
"""
function dataset_summary(dataset::FarmlandDataset)
    println("\n" * "="^50)
    println("Dataset Summary")
    println("="^50)
    println("Number of samples:     $(length(dataset))")
    println("Sequence length:       $(dataset.seq_len)")
    println("Feature dimension:     $(dataset.feature_dim)")
    println("Number of classes:     $(dataset.num_classes)")
    println()

    # Class distribution
    labels = [s.label for s in dataset.samples]
    println("Class Distribution:")
    println("-"^30)
    for (i, name) in enumerate(dataset.class_names)
        count = sum(labels .== i)
        pct = 100.0 * count / length(labels)
        @printf("  %8s: %5d (%.1f%%)\n", name, count, pct)
    end

    # Data statistics
    all_data = vcat([s.data for s in dataset.samples]...)
    println()
    println("Data Statistics:")
    println("-"^30)
    @printf("  Min:    %.4f\n", minimum(all_data))
    @printf("  Max:    %.4f\n", maximum(all_data))
    @printf("  Mean:   %.4f\n", mean(all_data))
    @printf("  Std:    %.4f\n", std(all_data))
    println("="^50 * "\n")
end

# ============================================================================
# Synthetic Data Generation (for testing)
# ============================================================================

"""
    generate_synthetic_data(n_samples::Int=1000, seq_len::Int=24,
                            n_classes::Int=8, n_features::Int=1;
                            noise::Float64=0.1)

Generate synthetic time series data for testing.

# Arguments
- `n_samples::Int`: Number of samples to generate
- `seq_len::Int`: Sequence length
- `n_classes::Int`: Number of classes
- `n_features::Int`: Number of features per time step
- `noise::Float64`: Noise level

# Returns
- `FarmlandDataset`: Generated synthetic dataset
"""
function generate_synthetic_data(n_samples::Int=1000, seq_len::Int=24,
                                 n_classes::Int=8, n_features::Int=1;
                                 noise::Float64=0.1)
    @info "Generating synthetic dataset with $n_samples samples"

    samples = TimeSeriesData[]

    # Define different patterns for each class
    patterns = [
        # Class 1: Early peak (spring crop)
        t -> 0.5 * sin(2π * t / seq_len + π/4) + 0.5,
        # Class 2: Late peak (summer crop)
        t -> 0.5 * sin(2π * t / seq_len - π/4) + 0.5,
        # Class 3: Double peak
        t -> 0.3 * sin(4π * t / seq_len) + 0.3 * sin(2π * t / seq_len) + 0.4,
        # Class 4: Constant high
        t -> 0.8 + 0.1 * sin(2π * t / seq_len),
        # Class 5: Linear increase
        t -> 0.2 + 0.6 * t / seq_len,
        # Class 6: Linear decrease
        t -> 0.8 - 0.6 * t / seq_len,
        # Class 7: Step function
        t -> t < seq_len / 2 ? 0.3 : 0.7,
        # Class 8: Random (noise)
        t -> 0.5
    ]

    for i in 1:n_samples
        label = rand(1:n_classes)
        pattern = patterns[min(label, length(patterns))]

        # Generate time series with pattern and noise
        data = zeros(Float32, seq_len, n_features)
        for t in 1:seq_len
            for f in 1:n_features
                data[t, f] = pattern(t) + noise * randn(Float32)
            end
        end

        # Clip to valid range
        data = clamp.(data, 0.0f0, 1.0f0)

        push!(samples, TimeSeriesData(data, label; id="synth_$i"))
    end

    class_names = ["Cotton", "Corn", "Pepper", "Jujube",
                   "Pear", "Apricot", "Tomato", "Others"]

    return FarmlandDataset(samples; class_names=class_names[1:n_classes])
end
