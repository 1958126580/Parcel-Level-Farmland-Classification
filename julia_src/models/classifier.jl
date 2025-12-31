"""
    Classifier Factory and Utilities

This module provides a unified interface for creating different
classification models and common utilities for model management.
"""

using Flux
using Flux: @functor

# ============================================================================
# Model Factory
# ============================================================================

"""
    create_model(model_name::String, config::Config)

Create a classification model by name.

# Arguments
- `model_name::String`: Model type ("TimesNet", "Transformer", "LSTM", "iTransformer")
- `config::Config`: Model configuration

# Returns
- Initialized model

# Example
```julia
config = default_config()
model = create_model("TimesNet", config)
```
"""
function create_model(model_name::String, config::Config)
    model_name = lowercase(model_name)

    if model_name == "timesnet"
        return TimesNet(config)
    elseif model_name == "simpletimesnet"
        return SimpleTimesNet(
            seq_len=config.seq_len,
            enc_in=config.enc_in,
            d_model=config.d_model,
            e_layers=config.e_layers,
            num_class=config.num_class,
            dropout=config.dropout
        )
    elseif model_name == "transformer"
        return TransformerClassifier(config)
    elseif model_name == "itransformer"
        return iTransformer(
            seq_len=config.seq_len,
            enc_in=config.enc_in,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            e_layers=config.e_layers,
            dropout=config.dropout,
            num_class=config.num_class
        )
    elseif model_name == "lstm" || model_name == "lstmattention"
        return LSTMAttentionClassifier(config)
    elseif model_name == "simplelstm"
        return SimpleLSTMClassifier(
            enc_in=config.enc_in,
            d_model=config.d_model,
            dropout=config.dropout,
            num_class=config.num_class
        )
    else
        error("Unknown model: $model_name. Available: TimesNet, Transformer, LSTM, iTransformer")
    end
end

"""
    create_model(config::Config)

Create model from config using the model_name field.
"""
function create_model(config::Config)
    return create_model(config.model_name, config)
end

# ============================================================================
# Ensemble Classifier
# ============================================================================

"""
    EnsembleClassifier

Ensemble of multiple classifiers with voting/averaging.

# Fields
- `models::Vector`: List of models
- `weights::Vector{Float32}`: Model weights for weighted averaging
- `method::Symbol`: Ensemble method (:vote, :mean, :weighted)
"""
struct EnsembleClassifier
    models::Vector{Any}
    weights::Vector{Float32}
    method::Symbol
end

@functor EnsembleClassifier

# Only make the models trainable
Flux.trainable(ec::EnsembleClassifier) = (models = ec.models,)

"""
    EnsembleClassifier(models::Vector; method::Symbol=:mean, weights=nothing)

Create ensemble classifier.

# Arguments
- `models::Vector`: List of models
- `method::Symbol`: Ensemble method
- `weights`: Optional weights for weighted averaging
"""
function EnsembleClassifier(models::Vector; method::Symbol=:mean, weights=nothing)
    n = length(models)
    if weights === nothing
        weights = ones(Float32, n) ./ n
    else
        weights = Float32.(weights)
    end
    return EnsembleClassifier(models, weights, method)
end

"""
    (ensemble::EnsembleClassifier)(x, mask=nothing)

Forward pass of ensemble classifier.
"""
function (ensemble::EnsembleClassifier)(x, mask=nothing)
    n = length(ensemble.models)

    if ensemble.method == :vote
        # Majority voting
        predictions = [argmax(model(x, mask), dims=2) for model in ensemble.models]
        # Count votes for each class
        votes = reduce(+, predictions)
        return Float32.(votes)

    elseif ensemble.method == :mean
        # Simple averaging of probabilities
        outputs = [softmax(model(x, mask), dims=2) for model in ensemble.models]
        return mean(outputs)

    else  # :weighted
        # Weighted averaging
        outputs = [softmax(model(x, mask), dims=2) for model in ensemble.models]
        result = zeros(Float32, size(outputs[1]))
        for (w, out) in zip(ensemble.weights, outputs)
            result .+= w .* out
        end
        return result
    end
end

# ============================================================================
# Model Analysis Utilities
# ============================================================================

"""
    count_parameters(model)

Count total trainable parameters in any model.
"""
function count_parameters(model)
    total = 0
    for p in Flux.params(model)
        total += length(p)
    end
    return total
end

"""
    model_info(model)

Get model information dictionary.
"""
function model_info(model)
    info = Dict{String, Any}()
    info["type"] = typeof(model)
    info["parameters"] = count_parameters(model)

    if hasfield(typeof(model), :seq_len)
        info["seq_len"] = model.seq_len
    end
    if hasfield(typeof(model), :d_model)
        info["d_model"] = model.d_model
    end
    if hasfield(typeof(model), :num_class)
        info["num_class"] = model.num_class
    end

    return info
end

# ============================================================================
# Model I/O
# ============================================================================

"""
    save_model(model, filepath::String; include_config::Bool=true)

Save model to file using BSON.

# Arguments
- `model`: Model to save
- `filepath::String`: Output file path
- `include_config::Bool`: Whether to include configuration
"""
function save_model(model, filepath::String; config=nothing)
    using BSON

    save_dict = Dict(
        :model_state => Flux.state(model),
        :model_type => string(typeof(model))
    )

    if config !== nothing
        save_dict[:config] = config
    end

    BSON.@save filepath save_dict
    @info "Model saved to $filepath"
end

"""
    load_model_state!(model, filepath::String)

Load model state from file.

# Arguments
- `model`: Model to load state into
- `filepath::String`: Input file path
"""
function load_model_state!(model, filepath::String)
    using BSON

    BSON.@load filepath save_dict
    Flux.loadmodel!(model, save_dict[:model_state])
    @info "Model state loaded from $filepath"
    return model
end

# ============================================================================
# Prediction Utilities
# ============================================================================

"""
    predict(model, x; mask=nothing)

Get predictions from model.

# Arguments
- `model`: Classification model
- `x`: Input data (batch_size × seq_len × features)
- `mask`: Optional padding mask

# Returns
- NamedTuple with predictions, probabilities, and logits
"""
function predict(model, x; mask=nothing)
    # Get logits
    logits = model(x, mask)

    # Compute probabilities
    probs = softmax(logits, dims=2)

    # Get predictions (1-indexed class labels)
    predictions = vec(mapslices(argmax, probs, dims=2))

    # Get confidence scores
    confidence = vec(maximum(probs, dims=2))

    return (
        predictions = predictions,
        probabilities = probs,
        confidence = confidence,
        logits = logits
    )
end

"""
    predict_proba(model, x; mask=nothing)

Get class probabilities from model.
"""
function predict_proba(model, x; mask=nothing)
    logits = model(x, mask)
    return softmax(logits, dims=2)
end

"""
    predict_batch(model, dataloader; device=cpu)

Get predictions for entire dataset.

# Arguments
- `model`: Classification model
- `dataloader`: Data loader
- `device`: Device (cpu or gpu)

# Returns
- NamedTuple with all predictions and true labels
"""
function predict_batch(model, dataloader; device=cpu)
    all_preds = Int[]
    all_probs = []
    all_labels = Int[]

    model = device(model)

    for (batch_x, batch_y, batch_mask) in dataloader
        batch_x = device(batch_x)
        batch_mask = device(batch_mask)

        result = predict(model, batch_x; mask=batch_mask)

        append!(all_preds, cpu(result.predictions))
        push!(all_probs, cpu(result.probabilities))
        append!(all_labels, cpu(vec(batch_y)))
    end

    return (
        predictions = all_preds,
        probabilities = vcat(all_probs...),
        labels = all_labels
    )
end

# ============================================================================
# Model Comparison
# ============================================================================

"""
    compare_models(models::Dict{String, Any}, test_loader;
                   device=cpu, verbose::Bool=true)

Compare multiple models on test data.

# Arguments
- `models::Dict{String, Any}`: Dictionary of model_name => model
- `test_loader`: Test data loader
- `device`: Device (cpu or gpu)
- `verbose::Bool`: Whether to print results

# Returns
- DataFrame with comparison results
"""
function compare_models(models::Dict{String, Any}, test_loader;
                        device=cpu, verbose::Bool=true)
    results = DataFrame(
        Model = String[],
        Accuracy = Float64[],
        F1_Macro = Float64[],
        Parameters = Int[]
    )

    for (name, model) in models
        @info "Evaluating $name..."

        result = predict_batch(model, test_loader; device=device)

        acc = accuracy(result.predictions, result.labels)
        num_classes = maximum(result.labels)
        f1 = f1_score(result.predictions, result.labels, num_classes; average=:macro)
        n_params = count_parameters(model)

        push!(results, (name, acc, f1, n_params))
    end

    if verbose
        println("\n" * "="^60)
        println("Model Comparison Results")
        println("="^60)
        println(results)
        println("="^60)
    end

    return results
end
