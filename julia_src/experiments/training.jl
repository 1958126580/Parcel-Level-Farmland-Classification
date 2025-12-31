"""
    Training module for Parcel-Level Farmland Classification

This module provides training infrastructure for deep learning models:
- Training loop with early stopping
- Loss functions
- Optimizer configuration
- Learning rate scheduling
- Training monitoring and logging
"""

using Flux
using Flux: crossentropy, logitcrossentropy
using ProgressMeter
using Statistics
using Printf

# ============================================================================
# Loss Functions
# ============================================================================

"""
    weighted_crossentropy(logits, targets, weights)

Cross-entropy loss with class weights for imbalanced data.

# Arguments
- `logits`: Model output logits (batch_size × num_classes)
- `targets`: One-hot encoded targets or class indices
- `weights`: Class weights

# Returns
- Weighted cross-entropy loss
"""
function weighted_crossentropy(logits, targets, weights)
    probs = softmax(logits, dims=2)

    # If targets are indices, convert to one-hot
    if ndims(targets) == 1
        targets_oh = Flux.onehotbatch(targets, 1:size(logits, 2))
        targets_oh = permutedims(Float32.(targets_oh))  # (B, C)
    else
        targets_oh = targets
    end

    # Compute weighted loss
    loss = -sum(targets_oh .* log.(probs .+ eps(Float32)) .* weights', dims=2)
    return mean(loss)
end

"""
    focal_loss(logits, targets; gamma::Float32=2.0f0, alpha::Float32=0.25f0)

Focal loss for handling class imbalance.

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

# Arguments
- `logits`: Model output logits
- `targets`: Class indices
- `gamma`: Focusing parameter
- `alpha`: Class balancing parameter
"""
function focal_loss(logits, targets; gamma::Float32=2.0f0, alpha::Float32=0.25f0)
    probs = softmax(logits, dims=2)
    num_classes = size(logits, 2)

    # Get probability of true class for each sample
    targets_oh = Flux.onehotbatch(targets, 1:num_classes)
    targets_oh = permutedims(Float32.(targets_oh))  # (B, C)

    pt = sum(probs .* targets_oh, dims=2)  # (B, 1)

    # Focal loss
    loss = -alpha .* (1 .- pt).^gamma .* log.(pt .+ eps(Float32))

    return mean(loss)
end

"""
    label_smoothing_loss(logits, targets; smoothing::Float32=0.1f0)

Cross-entropy with label smoothing.

# Arguments
- `logits`: Model output logits
- `targets`: Class indices
- `smoothing`: Smoothing factor
"""
function label_smoothing_loss(logits, targets; smoothing::Float32=0.1f0)
    num_classes = size(logits, 2)
    probs = softmax(logits, dims=2)

    # Create smoothed labels
    targets_oh = Flux.onehotbatch(targets, 1:num_classes)
    targets_oh = permutedims(Float32.(targets_oh))  # (B, C)

    smooth_targets = targets_oh .* (1 - smoothing) .+ (smoothing / num_classes)

    # Cross-entropy with smooth targets
    loss = -sum(smooth_targets .* log.(probs .+ eps(Float32)), dims=2)

    return mean(loss)
end

# ============================================================================
# Training Step
# ============================================================================

"""
    train_step!(model, opt_state, batch_x, batch_y, batch_mask;
                loss_fn=logitcrossentropy, max_grad_norm::Float64=4.0)

Perform single training step.

# Arguments
- `model`: Model to train
- `opt_state`: Optimizer state
- `batch_x`: Input batch
- `batch_y`: Target batch
- `batch_mask`: Padding mask
- `loss_fn`: Loss function
- `max_grad_norm`: Maximum gradient norm for clipping

# Returns
- Loss value for this batch
"""
function train_step!(model, opt_state, batch_x, batch_y, batch_mask;
                     loss_fn=logitcrossentropy, max_grad_norm::Float64=4.0)

    # Forward pass and compute gradients
    loss, grads = Flux.withgradient(model) do m
        logits = m(batch_x, batch_mask)
        loss_fn(logits', batch_y)
    end

    # Clip gradients
    grads = Flux.clipnorm(grads, max_grad_norm)

    # Update parameters
    Flux.update!(opt_state, model, grads[1])

    return loss
end

# ============================================================================
# Evaluation Step
# ============================================================================

"""
    eval_step(model, batch_x, batch_y, batch_mask; loss_fn=logitcrossentropy)

Perform single evaluation step.

# Arguments
- `model`: Model to evaluate
- `batch_x`: Input batch
- `batch_y`: Target batch
- `batch_mask`: Padding mask
- `loss_fn`: Loss function

# Returns
- Tuple of (loss, predictions, labels)
"""
function eval_step(model, batch_x, batch_y, batch_mask; loss_fn=logitcrossentropy)
    logits = model(batch_x, batch_mask)
    loss = loss_fn(logits', batch_y)

    predictions = vec(mapslices(argmax, softmax(logits, dims=2), dims=2))

    return (loss, predictions, vec(batch_y))
end

# ============================================================================
# Training Loop
# ============================================================================

"""
    train_model!(model, train_loader, val_loader, config::Config;
                 device=cpu, loss_fn=logitcrossentropy)

Train model with early stopping.

# Arguments
- `model`: Model to train
- `train_loader`: Training data loader
- `val_loader`: Validation data loader
- `config::Config`: Training configuration
- `device`: Device (cpu or gpu)
- `loss_fn`: Loss function

# Returns
- NamedTuple with training history and best model
"""
function train_model!(model, train_loader, val_loader, config::Config;
                      device=cpu, loss_fn=logitcrossentropy)

    @info "Starting training..."
    print_config(config)

    # Move model to device
    model = device(model)

    # Setup optimizer
    opt = Adam(config.learning_rate)
    opt_state = Flux.setup(opt, model)

    # Setup early stopping
    early_stopping = EarlyStopping(patience=config.patience, mode=:max)

    # Setup learning rate scheduler
    lr_scheduler = StepLRScheduler(Float64(config.learning_rate);
                                   step_size=10, gamma=0.5)

    # Training logger
    logger = TrainingLogger()

    # Best model tracking
    best_model_state = nothing
    best_accuracy = 0.0

    # Training loop
    @showprogress "Training" for epoch in 1:config.train_epochs
        epoch_start = time()

        # Training phase
        model = Flux.trainmode!(model)
        train_losses = Float64[]

        for (batch_x, batch_y, batch_mask) in train_loader
            batch_x = device(Float32.(batch_x))
            batch_y = device(batch_y)
            batch_mask = device(batch_mask)

            loss = train_step!(model, opt_state, batch_x, batch_y, batch_mask;
                              loss_fn=loss_fn, max_grad_norm=config.max_grad_norm)
            push!(train_losses, loss)
        end

        train_loss = mean(train_losses)

        # Validation phase
        model = Flux.testmode!(model)
        val_losses = Float64[]
        val_preds = Int[]
        val_labels = Int[]

        for (batch_x, batch_y, batch_mask) in val_loader
            batch_x = device(Float32.(batch_x))
            batch_y = device(batch_y)
            batch_mask = device(batch_mask)

            loss, preds, labels = eval_step(model, batch_x, batch_y, batch_mask;
                                            loss_fn=loss_fn)
            push!(val_losses, loss)
            append!(val_preds, cpu(preds))
            append!(val_labels, cpu(labels))
        end

        val_loss = mean(val_losses)
        val_accuracy = accuracy(val_preds, val_labels)

        # Compute training accuracy
        train_preds = Int[]
        train_labels_all = Int[]
        for (batch_x, batch_y, batch_mask) in train_loader
            batch_x = device(Float32.(batch_x))
            batch_mask = device(batch_mask)
            logits = model(batch_x, batch_mask)
            preds = vec(mapslices(argmax, cpu(softmax(logits, dims=2)), dims=2))
            append!(train_preds, preds)
            append!(train_labels_all, vec(batch_y))
        end
        train_accuracy = accuracy(train_preds, train_labels_all)

        # Get current learning rate
        current_lr = lr_scheduler(epoch)

        # Log epoch
        epoch_time = time() - epoch_start
        log_epoch!(logger, epoch, train_loss, val_loss, train_accuracy,
                   val_accuracy, current_lr)

        # Print epoch summary
        print_epoch_summary(epoch, config.train_epochs, train_loss, val_loss,
                           train_accuracy, val_accuracy, current_lr, epoch_time)

        # Check for improvement
        improved = early_stopping(val_accuracy)
        if improved
            best_accuracy = val_accuracy
            best_model_state = Flux.state(model)
            @info "New best validation accuracy: $(round(best_accuracy, digits=4))"
        end

        # Early stopping check
        if early_stopping.early_stop
            @info "Early stopping at epoch $epoch"
            break
        end

        # Adjust learning rate
        adjust_learning_rate!(opt, current_lr)
    end

    # Restore best model
    if best_model_state !== nothing
        Flux.loadmodel!(model, best_model_state)
        @info "Restored best model with accuracy: $(round(best_accuracy, digits=4))"
    end

    return (
        model = cpu(model),
        logger = logger,
        best_accuracy = best_accuracy
    )
end

# ============================================================================
# Quick Training Function
# ============================================================================

"""
    quick_train(model_name::String, train_data, test_data;
                epochs::Int=50, batch_size::Int=16, lr::Float64=0.0001)

Quick training function with default settings.

# Arguments
- `model_name::String`: Model type
- `train_data::FarmlandDataset`: Training dataset
- `test_data::FarmlandDataset`: Test dataset
- `epochs::Int`: Number of epochs
- `batch_size::Int`: Batch size
- `lr::Float64`: Learning rate

# Returns
- Trained model and results
"""
function quick_train(model_name::String, train_data, test_data;
                     epochs::Int=50, batch_size::Int=16, lr::Float64=0.0001)
    # Create config
    config = Config(
        model_name=model_name,
        seq_len=train_data.seq_len,
        enc_in=train_data.feature_dim,
        num_class=train_data.num_classes,
        train_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr
    )

    # Create model
    model = create_model(config)

    # Create data loaders
    train_loader, test_loader = create_dataloaders(train_data, test_data;
                                                   batch_size=batch_size)

    # Train
    result = train_model!(model, train_loader, test_loader, config)

    return result
end

# ============================================================================
# Cross-Validation
# ============================================================================

"""
    cross_validate(model_name::String, dataset::FarmlandDataset;
                   k::Int=5, config::Config=default_config())

Perform k-fold cross-validation.

# Arguments
- `model_name::String`: Model type
- `dataset::FarmlandDataset`: Full dataset
- `k::Int`: Number of folds
- `config::Config`: Training configuration

# Returns
- CrossValidation results with accuracy for each fold
"""
function cross_validate(model_name::String, dataset::FarmlandDataset;
                        k::Int=5, config::Config=default_config())

    n_samples = length(dataset)
    fold_size = n_samples ÷ k

    accuracies = Float64[]
    f1_scores = Float64[]

    for fold in 1:k
        @info "Fold $fold/$k"

        # Split data
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == k ? n_samples : fold * fold_size

        test_indices = test_start:test_end
        train_indices = setdiff(1:n_samples, test_indices)

        train_data = dataset[collect(train_indices)]
        test_data = dataset[collect(test_indices)]

        # Create model
        config.seq_len = dataset.seq_len
        config.enc_in = dataset.feature_dim
        config.num_class = dataset.num_classes
        model = create_model(model_name, config)

        # Create loaders
        train_loader, test_loader = create_dataloaders(train_data, test_data;
                                                       batch_size=config.batch_size)

        # Train
        result = train_model!(model, train_loader, test_loader, config)

        # Evaluate
        predictions = predict_batch(result.model, test_loader)
        acc = accuracy(predictions.predictions, predictions.labels)
        f1 = f1_score(predictions.predictions, predictions.labels,
                      config.num_class; average=:macro)

        push!(accuracies, acc)
        push!(f1_scores, f1)

        @info "Fold $fold - Accuracy: $(round(acc, digits=4)), F1: $(round(f1, digits=4))"
    end

    mean_acc = mean(accuracies)
    std_acc = std(accuracies)
    mean_f1 = mean(f1_scores)
    std_f1 = std(f1_scores)

    println("\n" * "="^50)
    println("Cross-Validation Results ($k folds)")
    println("="^50)
    @printf("Accuracy: %.4f ± %.4f\n", mean_acc, std_acc)
    @printf("F1 Score: %.4f ± %.4f\n", mean_f1, std_f1)
    println("="^50)

    return (
        accuracies = accuracies,
        f1_scores = f1_scores,
        mean_accuracy = mean_acc,
        std_accuracy = std_acc,
        mean_f1 = mean_f1,
        std_f1 = std_f1
    )
end
