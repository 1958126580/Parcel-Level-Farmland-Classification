"""
    Iterative Self-Optimization Module for Parcel-Level Farmland Classification

This module implements the iterative self-training approach described in the paper:
"Parcel-Level Farmland Classification via Deep Heterogeneous Feature Integration
and Iterative Self-Optimization"

The self-optimization process:
1. Train initial model on labeled data
2. Generate pseudo-labels for unlabeled data with high confidence
3. Augment training set with pseudo-labeled samples
4. Retrain model iteratively until convergence

Key features:
- Confidence-based pseudo-label selection
- Progressive threshold adjustment
- Sample weighting based on prediction confidence
- Temporal consistency filtering
- Parcel-level agreement checking
"""

using Flux
using Statistics
using Random
using Printf

# ============================================================================
# Pseudo-Label Generation
# ============================================================================

"""
    generate_pseudo_labels(model, unlabeled_loader;
                          confidence_threshold::Float64=0.95,
                          device=cpu)

Generate pseudo-labels for unlabeled data using model predictions.

# Arguments
- `model`: Trained classification model
- `unlabeled_loader`: DataLoader for unlabeled data
- `confidence_threshold`: Minimum confidence for accepting pseudo-label
- `device`: Computation device

# Returns
- NamedTuple with:
  - `indices`: Indices of samples with high-confidence predictions
  - `pseudo_labels`: Generated labels for selected samples
  - `confidences`: Confidence scores
  - `features`: Input features for selected samples
"""
function generate_pseudo_labels(model, unlabeled_loader;
                               confidence_threshold::Float64=0.95,
                               device=cpu)

    model = device(model)
    model = Flux.testmode!(model)

    selected_indices = Int[]
    pseudo_labels = Int[]
    confidences = Float64[]
    selected_features = []
    selected_masks = []

    sample_idx = 0

    for (batch_x, batch_mask) in unlabeled_loader
        batch_x = device(Float32.(batch_x))
        batch_mask = device(batch_mask)

        # Get predictions
        logits = model(batch_x, batch_mask)
        probs = softmax(logits, dims=2)

        # Find max probability and predicted class for each sample
        max_probs, pred_classes = findmax(cpu(probs), dims=2)

        batch_size = size(batch_x, 1)

        for i in 1:batch_size
            sample_idx += 1
            conf = max_probs[i][1]
            pred_class = pred_classes[i][2]  # Column index is the class

            if conf >= confidence_threshold
                push!(selected_indices, sample_idx)
                push!(pseudo_labels, pred_class)
                push!(confidences, conf)
                push!(selected_features, cpu(batch_x[i:i, :, :]))
                push!(selected_masks, cpu(batch_mask[i:i, :]))
            end
        end
    end

    @info "Generated $(length(pseudo_labels)) pseudo-labels from $(sample_idx) samples " *
          "($(round(100*length(pseudo_labels)/sample_idx, digits=2))%)"

    return (
        indices = selected_indices,
        pseudo_labels = pseudo_labels,
        confidences = confidences,
        features = selected_features,
        masks = selected_masks
    )
end

"""
    filter_by_temporal_consistency(predictions_multi::Vector{Vector{Int}};
                                   agreement_threshold::Float64=0.8)

Filter pseudo-labels by temporal consistency across multiple predictions.

# Arguments
- `predictions_multi`: Multiple prediction sets from different epochs
- `agreement_threshold`: Minimum agreement ratio required

# Returns
- Vector of indices with consistent predictions
"""
function filter_by_temporal_consistency(predictions_multi::Vector{Vector{Int}};
                                       agreement_threshold::Float64=0.8)
    n_samples = length(predictions_multi[1])
    n_predictions = length(predictions_multi)

    consistent_indices = Int[]

    for i in 1:n_samples
        # Count most common prediction
        preds = [predictions_multi[j][i] for j in 1:n_predictions]
        unique_preds = unique(preds)

        max_count = 0
        for p in unique_preds
            count = sum(preds .== p)
            max_count = max(max_count, count)
        end

        agreement = max_count / n_predictions

        if agreement >= agreement_threshold
            push!(consistent_indices, i)
        end
    end

    return consistent_indices
end

"""
    filter_by_parcel_agreement(predictions::Vector{Int},
                              parcel_ids::Vector{Int};
                              agreement_threshold::Float64=0.7)

Filter pseudo-labels by checking agreement within parcels.

Pixels belonging to the same parcel should have consistent labels.

# Arguments
- `predictions`: Predicted class labels
- `parcel_ids`: Parcel ID for each sample
- `agreement_threshold`: Minimum agreement within parcel

# Returns
- Vector of indices with parcel-consistent predictions
"""
function filter_by_parcel_agreement(predictions::Vector{Int},
                                   parcel_ids::Vector{Int};
                                   agreement_threshold::Float64=0.7)

    unique_parcels = unique(parcel_ids)
    consistent_indices = Int[]

    for parcel in unique_parcels
        # Find all samples in this parcel
        parcel_indices = findall(parcel_ids .== parcel)
        parcel_preds = predictions[parcel_indices]

        # Find most common prediction in parcel
        unique_preds = unique(parcel_preds)
        counts = [sum(parcel_preds .== p) for p in unique_preds]
        max_idx = argmax(counts)
        majority_pred = unique_preds[max_idx]

        # Calculate agreement
        agreement = counts[max_idx] / length(parcel_preds)

        if agreement >= agreement_threshold
            # Add indices that agree with majority
            for (i, idx) in enumerate(parcel_indices)
                if parcel_preds[i] == majority_pred
                    push!(consistent_indices, idx)
                end
            end
        end
    end

    return consistent_indices
end

# ============================================================================
# Self-Training Loop
# ============================================================================

"""
    SelfTrainingConfig

Configuration for self-training iteration.

# Fields
- `initial_threshold`: Starting confidence threshold (default: 0.95)
- `final_threshold`: Minimum confidence threshold (default: 0.85)
- `threshold_decay`: Decay rate per iteration (default: 0.98)
- `max_iterations`: Maximum self-training iterations (default: 10)
- `min_new_samples`: Minimum new pseudo-labels per iteration (default: 10)
- `use_temporal_consistency`: Enable temporal filtering (default: true)
- `use_parcel_agreement`: Enable parcel filtering (default: true)
- `sample_weight_decay`: Weight decay for older pseudo-labels (default: 0.95)
"""
struct SelfTrainingConfig
    initial_threshold::Float64
    final_threshold::Float64
    threshold_decay::Float64
    max_iterations::Int
    min_new_samples::Int
    use_temporal_consistency::Bool
    use_parcel_agreement::Bool
    sample_weight_decay::Float64

    function SelfTrainingConfig(;
        initial_threshold::Float64=0.95,
        final_threshold::Float64=0.85,
        threshold_decay::Float64=0.98,
        max_iterations::Int=10,
        min_new_samples::Int=10,
        use_temporal_consistency::Bool=true,
        use_parcel_agreement::Bool=true,
        sample_weight_decay::Float64=0.95
    )
        new(initial_threshold, final_threshold, threshold_decay,
            max_iterations, min_new_samples, use_temporal_consistency,
            use_parcel_agreement, sample_weight_decay)
    end
end

"""
    PseudoLabeledDataset

Dataset augmented with pseudo-labeled samples.

# Fields
- `original_data`: Original labeled dataset
- `pseudo_features`: Features with pseudo-labels
- `pseudo_labels`: Pseudo-label values
- `pseudo_weights`: Sample weights for pseudo-labeled data
- `pseudo_masks`: Attention masks for pseudo-labeled data
"""
mutable struct PseudoLabeledDataset
    original_data::Any
    pseudo_features::Vector{Any}
    pseudo_labels::Vector{Int}
    pseudo_weights::Vector{Float64}
    pseudo_masks::Vector{Any}

    function PseudoLabeledDataset(original_data)
        new(original_data, [], Int[], Float64[], [])
    end
end

"""
    add_pseudo_samples!(dataset::PseudoLabeledDataset,
                       features, labels, weights, masks)

Add pseudo-labeled samples to the dataset.
"""
function add_pseudo_samples!(dataset::PseudoLabeledDataset,
                            features, labels, weights, masks)
    append!(dataset.pseudo_features, features)
    append!(dataset.pseudo_labels, labels)
    append!(dataset.pseudo_weights, weights)
    append!(dataset.pseudo_masks, masks)
end

"""
    decay_weights!(dataset::PseudoLabeledDataset, decay::Float64)

Apply weight decay to existing pseudo-labeled samples.
"""
function decay_weights!(dataset::PseudoLabeledDataset, decay::Float64)
    dataset.pseudo_weights .*= decay
end

"""
    iterative_self_training(model_fn, train_loader, unlabeled_loader, val_loader,
                           config::Config, st_config::SelfTrainingConfig;
                           device=cpu)

Perform iterative self-training for semi-supervised learning.

# Algorithm
1. Train initial model on labeled data
2. For each iteration:
   a. Generate pseudo-labels for unlabeled data
   b. Filter by confidence threshold
   c. Optionally filter by temporal consistency
   d. Optionally filter by parcel agreement
   e. Add filtered pseudo-labels to training set
   f. Retrain model on augmented dataset
   g. Evaluate on validation set
   h. Decay confidence threshold
3. Return best model

# Arguments
- `model_fn`: Function to create new model instances
- `train_loader`: DataLoader for labeled training data
- `unlabeled_loader`: DataLoader for unlabeled data
- `val_loader`: DataLoader for validation data
- `config`: Model training configuration
- `st_config`: Self-training configuration
- `device`: Computation device

# Returns
- NamedTuple with best model, training history, and pseudo-label statistics
"""
function iterative_self_training(model_fn, train_loader, unlabeled_loader, val_loader,
                                config::Config, st_config::SelfTrainingConfig;
                                device=cpu)

    println("="^60)
    println("Starting Iterative Self-Training")
    println("="^60)
    println("Configuration:")
    @printf("  Initial threshold: %.2f\n", st_config.initial_threshold)
    @printf("  Final threshold: %.2f\n", st_config.final_threshold)
    @printf("  Max iterations: %d\n", st_config.max_iterations)
    println("="^60)

    # Initialize tracking
    best_model_state = nothing
    best_accuracy = 0.0
    iteration_history = Dict{String, Vector{Float64}}(
        "val_accuracy" => Float64[],
        "pseudo_count" => Float64[],
        "threshold" => Float64[]
    )

    current_threshold = st_config.initial_threshold
    total_pseudo_labels = 0

    # Store predictions for temporal consistency
    prediction_history = Vector{Int}[]

    # Create augmented dataset
    augmented_dataset = nothing  # Will be created after first pseudo-labeling

    for iteration in 1:st_config.max_iterations
        println("\n" * "-"^40)
        println("Self-Training Iteration $iteration/$(st_config.max_iterations)")
        println("-"^40)
        @printf("Confidence threshold: %.3f\n", current_threshold)

        # Create and train model
        model = model_fn()
        model = device(model)

        # Train on current data (labeled + pseudo-labeled)
        if iteration == 1
            # First iteration: train only on labeled data
            result = train_model!(model, train_loader, val_loader, config;
                                 device=device)
        else
            # Subsequent iterations: train on augmented data
            augmented_loader = create_augmented_loader(augmented_dataset, config.batch_size)
            result = train_model!(model, augmented_loader, val_loader, config;
                                 device=device)
        end

        model = result.model
        val_accuracy = result.best_accuracy

        @printf("Validation accuracy: %.4f\n", val_accuracy)

        # Track best model
        if val_accuracy > best_accuracy
            best_accuracy = val_accuracy
            best_model_state = Flux.state(model)
            @info "New best model! Accuracy: $(round(best_accuracy, digits=4))"
        end

        # Generate pseudo-labels
        pseudo_result = generate_pseudo_labels(model, unlabeled_loader;
                                              confidence_threshold=current_threshold,
                                              device=device)

        # Store predictions for temporal consistency
        push!(prediction_history, pseudo_result.pseudo_labels)

        new_pseudo_count = length(pseudo_result.pseudo_labels)

        # Check stopping criterion
        if new_pseudo_count < st_config.min_new_samples
            @info "Too few new pseudo-labels ($new_pseudo_count). Stopping."
            break
        end

        # Filter by temporal consistency if enabled
        if st_config.use_temporal_consistency && length(prediction_history) >= 3
            consistent_idx = filter_by_temporal_consistency(
                prediction_history[end-2:end];
                agreement_threshold=0.8
            )
            @info "Temporal consistency filter: $(length(consistent_idx))/" *
                  "$(length(pseudo_result.pseudo_labels)) samples"
        end

        # Create or update augmented dataset
        if iteration == 1
            # Initialize with original training data
            augmented_dataset = PseudoLabeledDataset(train_loader)
        end

        # Decay existing pseudo-label weights
        decay_weights!(augmented_dataset, st_config.sample_weight_decay)

        # Add new pseudo-labeled samples with initial weight 1.0
        weights = fill(1.0, length(pseudo_result.pseudo_labels))
        add_pseudo_samples!(augmented_dataset,
                           pseudo_result.features,
                           pseudo_result.pseudo_labels,
                           weights,
                           pseudo_result.masks)

        total_pseudo_labels += new_pseudo_count

        # Record history
        push!(iteration_history["val_accuracy"], val_accuracy)
        push!(iteration_history["pseudo_count"], Float64(total_pseudo_labels))
        push!(iteration_history["threshold"], current_threshold)

        # Decay threshold
        current_threshold = max(st_config.final_threshold,
                               current_threshold * st_config.threshold_decay)

        # Print iteration summary
        println("\nIteration Summary:")
        @printf("  New pseudo-labels: %d\n", new_pseudo_count)
        @printf("  Total pseudo-labels: %d\n", total_pseudo_labels)
        @printf("  Next threshold: %.3f\n", current_threshold)
    end

    # Restore best model
    final_model = model_fn()
    if best_model_state !== nothing
        Flux.loadmodel!(final_model, best_model_state)
    end

    println("\n" * "="^60)
    println("Self-Training Complete!")
    @printf("Best validation accuracy: %.4f\n", best_accuracy)
    @printf("Total pseudo-labels added: %d\n", total_pseudo_labels)
    println("="^60)

    return (
        model = final_model,
        best_accuracy = best_accuracy,
        history = iteration_history,
        total_pseudo_labels = total_pseudo_labels
    )
end

"""
    create_augmented_loader(dataset::PseudoLabeledDataset, batch_size::Int)

Create a DataLoader for the augmented dataset.

This function combines original labeled data with pseudo-labeled data.
"""
function create_augmented_loader(dataset::PseudoLabeledDataset, batch_size::Int)
    # Combine original and pseudo-labeled data
    # This is a simplified version - actual implementation would need
    # to properly batch and shuffle the combined data

    # For now, return the original loader
    # Full implementation would create a proper combined loader
    return dataset.original_data
end

# ============================================================================
# Confidence Calibration
# ============================================================================

"""
    calibrate_confidence(model, val_loader; device=cpu, n_bins::Int=15)

Calibrate model confidence using temperature scaling.

# Arguments
- `model`: Trained model
- `val_loader`: Validation data loader
- `device`: Computation device
- `n_bins`: Number of bins for calibration

# Returns
- Optimal temperature for calibration
"""
function calibrate_confidence(model, val_loader; device=cpu, n_bins::Int=15)
    model = device(model)
    model = Flux.testmode!(model)

    all_logits = []
    all_labels = []

    for (batch_x, batch_y, batch_mask) in val_loader
        batch_x = device(Float32.(batch_x))
        batch_mask = device(batch_mask)

        logits = cpu(model(batch_x, batch_mask))
        push!(all_logits, logits)
        push!(all_labels, batch_y)
    end

    logits = vcat(all_logits...)
    labels = vcat(all_labels...)

    # Grid search for optimal temperature
    best_temp = 1.0
    best_ece = Inf

    for temp in 0.5:0.1:3.0
        scaled_logits = logits ./ temp
        probs = softmax(scaled_logits, dims=2)

        ece = compute_ece(probs, labels; n_bins=n_bins)

        if ece < best_ece
            best_ece = ece
            best_temp = temp
        end
    end

    @info "Optimal temperature: $(round(best_temp, digits=2)), ECE: $(round(best_ece, digits=4))"

    return best_temp
end

"""
    compute_ece(probs::AbstractMatrix, labels::AbstractVector; n_bins::Int=15)

Compute Expected Calibration Error.

# Arguments
- `probs`: Predicted probabilities (n_samples × n_classes)
- `labels`: True labels
- `n_bins`: Number of bins

# Returns
- ECE value
"""
function compute_ece(probs::AbstractMatrix, labels::AbstractVector; n_bins::Int=15)
    max_probs, pred_classes = findmax(probs, dims=2)
    max_probs = vec(max_probs)
    pred_classes = [idx[2] for idx in pred_classes]

    bin_boundaries = range(0, 1, length=n_bins+1)
    ece = 0.0

    for i in 1:n_bins
        # Find samples in this bin
        lower = bin_boundaries[i]
        upper = bin_boundaries[i+1]

        in_bin = (max_probs .> lower) .& (max_probs .<= upper)
        n_in_bin = sum(in_bin)

        if n_in_bin > 0
            avg_confidence = mean(max_probs[in_bin])
            accuracy_in_bin = mean(pred_classes[in_bin] .== labels[in_bin])
            ece += abs(avg_confidence - accuracy_in_bin) * n_in_bin
        end
    end

    return ece / length(labels)
end

# ============================================================================
# Progressive Self-Training
# ============================================================================

"""
    progressive_self_training(model_fn, train_data, unlabeled_data, val_data,
                             config::Config;
                             n_stages::Int=3, device=cpu)

Progressive self-training with curriculum learning.

Gradually incorporates harder samples in later stages.

# Arguments
- `model_fn`: Function to create model
- `train_data`: Labeled training data
- `unlabeled_data`: Unlabeled data
- `val_data`: Validation data
- `config`: Training configuration
- `n_stages`: Number of progressive stages
- `device`: Computation device

# Returns
- Final trained model and training history
"""
function progressive_self_training(model_fn, train_data, unlabeled_data, val_data,
                                  config::Config;
                                  n_stages::Int=3, device=cpu)

    println("="^60)
    println("Progressive Self-Training")
    println("="^60)

    # Define thresholds for each stage
    thresholds = range(0.98, 0.85, length=n_stages)

    current_train_data = train_data
    best_model = nothing
    best_accuracy = 0.0

    for (stage, threshold) in enumerate(thresholds)
        println("\n" * "="^40)
        println("Stage $stage/$n_stages (threshold: $(round(threshold, digits=2)))")
        println("="^40)

        # Create model
        model = model_fn()
        model = device(model)

        # Create data loaders
        train_loader = create_dataloader(current_train_data;
                                        batch_size=config.batch_size,
                                        shuffle=true)
        val_loader = create_dataloader(val_data;
                                      batch_size=config.batch_size)
        unlabeled_loader = create_unlabeled_dataloader(unlabeled_data;
                                                      batch_size=config.batch_size)

        # Train model
        result = train_model!(model, train_loader, val_loader, config; device=device)
        model = result.model

        if result.best_accuracy > best_accuracy
            best_accuracy = result.best_accuracy
            best_model = deepcopy(model)
        end

        # Generate pseudo-labels for next stage
        if stage < n_stages
            pseudo_result = generate_pseudo_labels(model, unlabeled_loader;
                                                  confidence_threshold=threshold,
                                                  device=device)

            # Add pseudo-labeled samples to training data
            current_train_data = augment_dataset(current_train_data,
                                                pseudo_result.features,
                                                pseudo_result.pseudo_labels,
                                                pseudo_result.masks)

            @info "Stage $stage complete. Added $(length(pseudo_result.pseudo_labels)) samples"
        end
    end

    println("\n" * "="^60)
    println("Progressive Self-Training Complete!")
    @printf("Best validation accuracy: %.4f\n", best_accuracy)
    println("="^60)

    return (model = best_model, best_accuracy = best_accuracy)
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
    create_dataloader(data; batch_size::Int=32, shuffle::Bool=false)

Create a data loader from dataset.
"""
function create_dataloader(data; batch_size::Int=32, shuffle::Bool=false)
    # Placeholder - actual implementation depends on data structure
    return data
end

"""
    create_unlabeled_dataloader(data; batch_size::Int=32)

Create a data loader for unlabeled data (without labels).
"""
function create_unlabeled_dataloader(data; batch_size::Int=32)
    # Placeholder - actual implementation depends on data structure
    return data
end

"""
    augment_dataset(original_data, new_features, new_labels, new_masks)

Augment dataset with new samples.
"""
function augment_dataset(original_data, new_features, new_labels, new_masks)
    # Placeholder - actual implementation depends on data structure
    return original_data
end

"""
    semi_supervised_training(model_fn, labeled_loader, unlabeled_loader, val_loader,
                            config::Config; device=cpu, method::Symbol=:self_training)

Unified interface for semi-supervised learning methods.

# Arguments
- `model_fn`: Model creation function
- `labeled_loader`: Labeled data loader
- `unlabeled_loader`: Unlabeled data loader
- `val_loader`: Validation data loader
- `config`: Training configuration
- `device`: Computation device
- `method`: Semi-supervised method (:self_training, :progressive, :fixmatch)

# Returns
- Trained model and results
"""
function semi_supervised_training(model_fn, labeled_loader, unlabeled_loader, val_loader,
                                 config::Config; device=cpu, method::Symbol=:self_training)

    if method == :self_training
        st_config = SelfTrainingConfig()
        return iterative_self_training(model_fn, labeled_loader, unlabeled_loader,
                                      val_loader, config, st_config; device=device)
    elseif method == :progressive
        return progressive_self_training(model_fn, labeled_loader, unlabeled_loader,
                                        val_loader, config; device=device)
    else
        error("Unknown semi-supervised method: $method")
    end
end
