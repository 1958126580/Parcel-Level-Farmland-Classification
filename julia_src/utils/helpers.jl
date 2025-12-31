"""
    Helper utilities for Parcel-Level Farmland Classification

This module provides utility functions for:
- Device management (CPU/GPU)
- Random seed setting
- Early stopping
- Learning rate scheduling
- Checkpoint management
- Progress monitoring
"""

# ============================================================================
# Device Management
# ============================================================================

"""
    get_device(use_gpu::Bool=true)

Get the appropriate device for computation.

# Arguments
- `use_gpu::Bool`: Whether to attempt GPU usage

# Returns
- Device function for moving data to appropriate device
"""
function get_device(use_gpu::Bool=true)
    if use_gpu && CUDA.functional()
        @info "Using GPU for computation"
        return gpu
    else
        @info "Using CPU for computation"
        return cpu
    end
end

"""
    to_device(x, device)

Move data to specified device.

# Arguments
- `x`: Data to move
- `device`: Target device (gpu or cpu function)

# Returns
- Data on target device
"""
to_device(x, device) = device(x)

"""
    to_device(x::AbstractArray{<:Number}, device)

Move numeric arrays to device.
"""
to_device(x::AbstractArray{<:Number}, device) = device(Float32.(x))

# ============================================================================
# Random Seed Management
# ============================================================================

"""
    set_seed(seed::Int)

Set random seed for reproducibility across all random number generators.

# Arguments
- `seed::Int`: Random seed value
"""
function set_seed(seed::Int)
    Random.seed!(seed)
    if CUDA.functional()
        CUDA.seed!(seed)
    end
    @info "Random seed set to $seed"
end

# ============================================================================
# Early Stopping
# ============================================================================

"""
    EarlyStopping

Early stopping callback to prevent overfitting.

# Fields
- `patience::Int`: Number of epochs to wait for improvement
- `min_delta::Float64`: Minimum change to qualify as improvement
- `mode::Symbol`: :min for loss, :max for accuracy
- `best_score::Float64`: Best metric value seen
- `counter::Int`: Epochs since last improvement
- `early_stop::Bool`: Whether to stop training
"""
mutable struct EarlyStopping
    patience::Int
    min_delta::Float64
    mode::Symbol
    best_score::Float64
    counter::Int
    early_stop::Bool

    function EarlyStopping(; patience::Int=10, min_delta::Float64=0.0, mode::Symbol=:min)
        best_score = mode == :min ? Inf : -Inf
        new(patience, min_delta, mode, best_score, 0, false)
    end
end

"""
    (es::EarlyStopping)(current_score::Float64)

Check if training should stop based on current score.

# Arguments
- `current_score::Float64`: Current metric value

# Returns
- `Bool`: True if improvement occurred, false otherwise
"""
function (es::EarlyStopping)(current_score::Float64)
    improved = false

    if es.mode == :min
        if current_score < es.best_score - es.min_delta
            es.best_score = current_score
            es.counter = 0
            improved = true
        else
            es.counter += 1
        end
    else  # :max
        if current_score > es.best_score + es.min_delta
            es.best_score = current_score
            es.counter = 0
            improved = true
        else
            es.counter += 1
        end
    end

    if es.counter >= es.patience
        es.early_stop = true
        @info "Early stopping triggered after $(es.counter) epochs without improvement"
    end

    return improved
end

"""
    reset!(es::EarlyStopping)

Reset early stopping state.
"""
function reset!(es::EarlyStopping)
    es.best_score = es.mode == :min ? Inf : -Inf
    es.counter = 0
    es.early_stop = false
end

# ============================================================================
# Learning Rate Scheduling
# ============================================================================

"""
    StepLRScheduler

Step learning rate scheduler.

# Fields
- `base_lr::Float64`: Initial learning rate
- `step_size::Int`: Epochs between LR reductions
- `gamma::Float64`: Multiplicative factor for LR reduction
"""
mutable struct StepLRScheduler
    base_lr::Float64
    step_size::Int
    gamma::Float64
    current_lr::Float64

    function StepLRScheduler(base_lr::Float64; step_size::Int=10, gamma::Float64=0.5)
        new(base_lr, step_size, gamma, base_lr)
    end
end

"""
    (scheduler::StepLRScheduler)(epoch::Int)

Get learning rate for current epoch.

# Arguments
- `epoch::Int`: Current epoch number

# Returns
- `Float64`: Learning rate for this epoch
"""
function (scheduler::StepLRScheduler)(epoch::Int)
    scheduler.current_lr = scheduler.base_lr * (scheduler.gamma ^ (epoch ÷ scheduler.step_size))
    return scheduler.current_lr
end

"""
    CosineAnnealingLR

Cosine annealing learning rate scheduler.

# Fields
- `base_lr::Float64`: Initial learning rate
- `T_max::Int`: Maximum number of epochs
- `eta_min::Float64`: Minimum learning rate
"""
mutable struct CosineAnnealingLR
    base_lr::Float64
    T_max::Int
    eta_min::Float64
    current_lr::Float64

    function CosineAnnealingLR(base_lr::Float64; T_max::Int=100, eta_min::Float64=1e-6)
        new(base_lr, T_max, eta_min, base_lr)
    end
end

"""
    (scheduler::CosineAnnealingLR)(epoch::Int)

Get learning rate for current epoch using cosine annealing.

# Arguments
- `epoch::Int`: Current epoch number

# Returns
- `Float64`: Learning rate for this epoch
"""
function (scheduler::CosineAnnealingLR)(epoch::Int)
    scheduler.current_lr = scheduler.eta_min +
        (scheduler.base_lr - scheduler.eta_min) *
        (1 + cos(π * epoch / scheduler.T_max)) / 2
    return scheduler.current_lr
end

"""
    adjust_learning_rate!(optimizer, new_lr::Float64)

Adjust optimizer learning rate.

# Arguments
- `optimizer`: Flux optimizer
- `new_lr::Float64`: New learning rate
"""
function adjust_learning_rate!(optimizer, new_lr::Float64)
    optimizer.eta = new_lr
end

# ============================================================================
# Checkpoint Management
# ============================================================================

"""
    Checkpoint

Model checkpoint for saving and loading.

# Fields
- `model_state`: Model parameters
- `optimizer_state`: Optimizer state
- `epoch::Int`: Current epoch
- `best_score::Float64`: Best validation score
- `config`: Training configuration
"""
struct Checkpoint
    model_state::Any
    optimizer_state::Any
    epoch::Int
    best_score::Float64
    config::Any
end

"""
    save_checkpoint(model, optimizer, epoch::Int, best_score::Float64,
                    config, filepath::String)

Save a training checkpoint.

# Arguments
- `model`: Model to save
- `optimizer`: Optimizer state
- `epoch::Int`: Current epoch
- `best_score::Float64`: Best score achieved
- `config`: Training configuration
- `filepath::String`: Output file path
"""
function save_checkpoint(model, optimizer, epoch::Int, best_score::Float64,
                         config, filepath::String)
    checkpoint = Checkpoint(
        Flux.state(model),
        optimizer,
        epoch,
        best_score,
        config
    )

    BSON.@save filepath checkpoint
    @info "Checkpoint saved to $filepath"
end

"""
    load_checkpoint(filepath::String)

Load a training checkpoint.

# Arguments
- `filepath::String`: Checkpoint file path

# Returns
- `Checkpoint`: Loaded checkpoint
"""
function load_checkpoint(filepath::String)
    BSON.@load filepath checkpoint
    @info "Checkpoint loaded from $filepath"
    return checkpoint
end

# ============================================================================
# Progress Monitoring
# ============================================================================

"""
    TrainingLogger

Logger for tracking training progress.

# Fields
- `train_losses::Vector{Float64}`: Training losses per epoch
- `val_losses::Vector{Float64}`: Validation losses per epoch
- `train_accs::Vector{Float64}`: Training accuracies per epoch
- `val_accs::Vector{Float64}`: Validation accuracies per epoch
- `learning_rates::Vector{Float64}`: Learning rates per epoch
"""
mutable struct TrainingLogger
    train_losses::Vector{Float64}
    val_losses::Vector{Float64}
    train_accs::Vector{Float64}
    val_accs::Vector{Float64}
    learning_rates::Vector{Float64}
    epochs::Vector{Int}

    TrainingLogger() = new([], [], [], [], [], [])
end

"""
    log_epoch!(logger::TrainingLogger, epoch::Int, train_loss::Float64,
               val_loss::Float64, train_acc::Float64, val_acc::Float64, lr::Float64)

Log metrics for an epoch.
"""
function log_epoch!(logger::TrainingLogger, epoch::Int, train_loss::Float64,
                    val_loss::Float64, train_acc::Float64, val_acc::Float64, lr::Float64)
    push!(logger.epochs, epoch)
    push!(logger.train_losses, train_loss)
    push!(logger.val_losses, val_loss)
    push!(logger.train_accs, train_acc)
    push!(logger.val_accs, val_acc)
    push!(logger.learning_rates, lr)
end

"""
    print_epoch_summary(epoch::Int, total_epochs::Int, train_loss::Float64,
                        val_loss::Float64, train_acc::Float64, val_acc::Float64,
                        lr::Float64, elapsed_time::Float64)

Print formatted epoch summary.
"""
function print_epoch_summary(epoch::Int, total_epochs::Int, train_loss::Float64,
                             val_loss::Float64, train_acc::Float64, val_acc::Float64,
                             lr::Float64, elapsed_time::Float64)
    @printf("Epoch %3d/%3d | Train Loss: %.4f | Val Loss: %.4f | " *
            "Train Acc: %.4f | Val Acc: %.4f | LR: %.6f | Time: %.2fs\n",
            epoch, total_epochs, train_loss, val_loss, train_acc, val_acc,
            lr, elapsed_time)
end

# ============================================================================
# Data Utilities
# ============================================================================

"""
    train_val_split(data::AbstractArray, labels::AbstractArray;
                    val_ratio::Float64=0.2, shuffle::Bool=true)

Split data into training and validation sets.

# Arguments
- `data::AbstractArray`: Input data
- `labels::AbstractArray`: Labels
- `val_ratio::Float64`: Validation set ratio
- `shuffle::Bool`: Whether to shuffle before splitting

# Returns
- Tuple of (train_data, train_labels, val_data, val_labels)
"""
function train_val_split(data::AbstractArray, labels::AbstractArray;
                         val_ratio::Float64=0.2, shuffle::Bool=true)
    n = size(data, 1)
    indices = collect(1:n)

    if shuffle
        shuffle!(indices)
    end

    n_val = round(Int, n * val_ratio)
    val_indices = indices[1:n_val]
    train_indices = indices[n_val+1:end]

    return (data[train_indices, :, :], labels[train_indices],
            data[val_indices, :, :], labels[val_indices])
end

"""
    stratified_split(labels::AbstractVector; train_ratio::Float64=0.8)

Create stratified train/test split indices.

# Arguments
- `labels::AbstractVector`: Class labels
- `train_ratio::Float64`: Training set ratio

# Returns
- Tuple of (train_indices, test_indices)
"""
function stratified_split(labels::AbstractVector; train_ratio::Float64=0.8)
    unique_labels = unique(labels)
    train_indices = Int[]
    test_indices = Int[]

    for label in unique_labels
        label_indices = findall(x -> x == label, labels)
        n_train = round(Int, length(label_indices) * train_ratio)

        shuffle!(label_indices)
        append!(train_indices, label_indices[1:n_train])
        append!(test_indices, label_indices[n_train+1:end])
    end

    return (train_indices, test_indices)
end

# ============================================================================
# Batch Utilities
# ============================================================================

"""
    create_batches(data::AbstractArray, labels::AbstractVector,
                   batch_size::Int; shuffle::Bool=true)

Create batches for training.

# Arguments
- `data::AbstractArray`: Input data (N × T × C)
- `labels::AbstractVector`: Labels
- `batch_size::Int`: Batch size
- `shuffle::Bool`: Whether to shuffle data

# Returns
- Vector of (batch_data, batch_labels) tuples
"""
function create_batches(data::AbstractArray, labels::AbstractVector,
                        batch_size::Int; shuffle::Bool=true)
    n = size(data, 1)
    indices = collect(1:n)

    if shuffle
        shuffle!(indices)
    end

    batches = []
    for i in 1:batch_size:n
        end_idx = min(i + batch_size - 1, n)
        batch_indices = indices[i:end_idx]

        batch_data = data[batch_indices, :, :]
        batch_labels = labels[batch_indices]

        push!(batches, (batch_data, batch_labels))
    end

    return batches
end

# ============================================================================
# Gradient Clipping
# ============================================================================

"""
    clip_gradients!(grads, max_norm::Float64)

Clip gradients by global norm.

# Arguments
- `grads`: Gradients to clip
- `max_norm::Float64`: Maximum gradient norm
"""
function clip_gradients!(grads, max_norm::Float64)
    total_norm = 0.0
    for g in grads
        if g !== nothing
            total_norm += sum(abs2, g)
        end
    end
    total_norm = sqrt(total_norm)

    if total_norm > max_norm
        scale = max_norm / total_norm
        for g in grads
            if g !== nothing
                g .*= scale
            end
        end
    end

    return total_norm
end

# ============================================================================
# Time Formatting
# ============================================================================

"""
    format_time(seconds::Float64)

Format seconds into human-readable string.

# Arguments
- `seconds::Float64`: Time in seconds

# Returns
- `String`: Formatted time string
"""
function format_time(seconds::Float64)
    if seconds < 60
        return @sprintf("%.1fs", seconds)
    elseif seconds < 3600
        mins = floor(Int, seconds / 60)
        secs = seconds - mins * 60
        return @sprintf("%dm %.1fs", mins, secs)
    else
        hours = floor(Int, seconds / 3600)
        mins = floor(Int, (seconds - hours * 3600) / 60)
        return @sprintf("%dh %dm", hours, mins)
    end
end
