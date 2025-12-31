"""
    Metrics module for Parcel-Level Farmland Classification

This module provides comprehensive evaluation metrics for assessing
classification performance on farmland crop type prediction.

Implemented metrics:
- Overall Accuracy (OA)
- Per-class Accuracy (PA)
- F1 Score (macro, micro, weighted)
- Precision and Recall
- Confusion Matrix
- Cohen's Kappa
"""

# ============================================================================
# Accuracy Metrics
# ============================================================================

"""
    accuracy(predictions::Vector{Int}, targets::Vector{Int})

Calculate overall accuracy.

# Arguments
- `predictions::Vector{Int}`: Predicted class labels
- `targets::Vector{Int}`: Ground truth labels

# Returns
- `Float64`: Overall accuracy (0-1)

# Example
```julia
preds = [1, 2, 1, 3, 2]
targets = [1, 2, 2, 3, 2]
acc = accuracy(preds, targets)  # 0.8
```
"""
function accuracy(predictions::Vector{Int}, targets::Vector{Int})
    @assert length(predictions) == length(targets) "Predictions and targets must have same length"
    return mean(predictions .== targets)
end

"""
    accuracy(predictions::AbstractMatrix, targets::Vector{Int})

Calculate accuracy from probability matrix.

# Arguments
- `predictions::AbstractMatrix`: Class probabilities (num_samples × num_classes)
- `targets::Vector{Int}`: Ground truth labels

# Returns
- `Float64`: Overall accuracy (0-1)
"""
function accuracy(predictions::AbstractMatrix, targets::Vector{Int})
    pred_labels = vec(mapslices(argmax, predictions, dims=2))
    return accuracy(pred_labels, targets)
end

"""
    per_class_accuracy(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int)

Calculate per-class accuracy (Producer's Accuracy).

# Arguments
- `predictions::Vector{Int}`: Predicted class labels
- `targets::Vector{Int}`: Ground truth labels
- `num_classes::Int`: Total number of classes

# Returns
- `Vector{Float64}`: Accuracy for each class
"""
function per_class_accuracy(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int)
    class_acc = zeros(Float64, num_classes)

    for c in 1:num_classes
        mask = targets .== c
        if sum(mask) > 0
            class_acc[c] = mean(predictions[mask] .== c)
        else
            class_acc[c] = NaN
        end
    end

    return class_acc
end

# ============================================================================
# Confusion Matrix
# ============================================================================

"""
    confusion_matrix(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int)

Compute confusion matrix.

# Arguments
- `predictions::Vector{Int}`: Predicted class labels
- `targets::Vector{Int}`: Ground truth labels
- `num_classes::Int`: Total number of classes

# Returns
- `Matrix{Int}`: Confusion matrix where element (i,j) is count of
  samples with true label i predicted as label j

# Example
```julia
cm = confusion_matrix([1,2,1,2], [1,1,2,2], 2)
# Output: [1 1; 1 1]
```
"""
function confusion_matrix(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int)
    @assert length(predictions) == length(targets) "Predictions and targets must have same length"
    @assert all(1 .<= predictions .<= num_classes) "All predictions must be in [1, num_classes]"
    @assert all(1 .<= targets .<= num_classes) "All targets must be in [1, num_classes]"

    cm = zeros(Int, num_classes, num_classes)

    for (pred, target) in zip(predictions, targets)
        cm[target, pred] += 1
    end

    return cm
end

"""
    normalize_confusion_matrix(cm::Matrix{Int}; mode::Symbol=:row)

Normalize confusion matrix.

# Arguments
- `cm::Matrix{Int}`: Confusion matrix
- `mode::Symbol`: Normalization mode (:row, :col, :all)

# Returns
- `Matrix{Float64}`: Normalized confusion matrix
"""
function normalize_confusion_matrix(cm::Matrix{Int}; mode::Symbol=:row)
    cm_float = Float64.(cm)

    if mode == :row
        # Normalize by row (true labels) - gives recall for each class
        row_sums = sum(cm_float, dims=2)
        row_sums[row_sums .== 0] .= 1.0  # Avoid division by zero
        return cm_float ./ row_sums
    elseif mode == :col
        # Normalize by column (predictions) - gives precision for each class
        col_sums = sum(cm_float, dims=1)
        col_sums[col_sums .== 0] .= 1.0
        return cm_float ./ col_sums
    else  # :all
        # Normalize by total count
        total = sum(cm_float)
        return total > 0 ? cm_float / total : cm_float
    end
end

# ============================================================================
# Precision, Recall, and F1 Score
# ============================================================================

"""
    precision_recall_f1(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int)

Calculate precision, recall, and F1 score for each class.

# Arguments
- `predictions::Vector{Int}`: Predicted class labels
- `targets::Vector{Int}`: Ground truth labels
- `num_classes::Int`: Total number of classes

# Returns
- `NamedTuple`: Contains precision, recall, and f1 vectors for each class
"""
function precision_recall_f1(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int)
    cm = confusion_matrix(predictions, targets, num_classes)

    precision_vec = zeros(Float64, num_classes)
    recall_vec = zeros(Float64, num_classes)
    f1_vec = zeros(Float64, num_classes)

    for c in 1:num_classes
        # True positives
        tp = cm[c, c]

        # False positives (predicted as c but not c)
        fp = sum(cm[:, c]) - tp

        # False negatives (is c but not predicted as c)
        fn = sum(cm[c, :]) - tp

        # Precision: TP / (TP + FP)
        precision_vec[c] = (tp + fp) > 0 ? tp / (tp + fp) : 0.0

        # Recall: TP / (TP + FN)
        recall_vec[c] = (tp + fn) > 0 ? tp / (tp + fn) : 0.0

        # F1: harmonic mean of precision and recall
        p, r = precision_vec[c], recall_vec[c]
        f1_vec[c] = (p + r) > 0 ? 2 * p * r / (p + r) : 0.0
    end

    return (precision=precision_vec, recall=recall_vec, f1=f1_vec)
end

"""
    f1_score(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int;
             average::Symbol=:macro)

Calculate F1 score with different averaging methods.

# Arguments
- `predictions::Vector{Int}`: Predicted class labels
- `targets::Vector{Int}`: Ground truth labels
- `num_classes::Int`: Total number of classes
- `average::Symbol`: Averaging method (:macro, :micro, :weighted, :none)

# Returns
- `Float64` or `Vector{Float64}`: F1 score(s)

# Example
```julia
f1 = f1_score([1,2,1,2], [1,1,2,2], 2; average=:macro)
```
"""
function f1_score(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int;
                  average::Symbol=:macro)
    metrics = precision_recall_f1(predictions, targets, num_classes)

    if average == :none
        return metrics.f1
    elseif average == :macro
        # Simple average of per-class F1 scores
        return mean(filter(!isnan, metrics.f1))
    elseif average == :micro
        # Calculate global TP, FP, FN
        cm = confusion_matrix(predictions, targets, num_classes)
        tp = sum([cm[c, c] for c in 1:num_classes])
        fp = sum([sum(cm[:, c]) - cm[c, c] for c in 1:num_classes])
        fn = sum([sum(cm[c, :]) - cm[c, c] for c in 1:num_classes])

        precision_micro = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
        recall_micro = (tp + fn) > 0 ? tp / (tp + fn) : 0.0

        return (precision_micro + recall_micro) > 0 ?
               2 * precision_micro * recall_micro / (precision_micro + recall_micro) : 0.0
    elseif average == :weighted
        # Weight by support (number of true samples per class)
        weights = [sum(targets .== c) for c in 1:num_classes]
        total = sum(weights)
        if total > 0
            return sum(metrics.f1 .* weights) / total
        else
            return 0.0
        end
    else
        error("Unknown average method: $average. Use :macro, :micro, :weighted, or :none")
    end
end

"""
    precision(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int;
              average::Symbol=:macro)

Calculate precision score.

# Arguments
- Same as `f1_score`

# Returns
- `Float64` or `Vector{Float64}`: Precision score(s)
"""
function precision(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int;
                   average::Symbol=:macro)
    metrics = precision_recall_f1(predictions, targets, num_classes)

    if average == :none
        return metrics.precision
    elseif average == :macro
        return mean(filter(!isnan, metrics.precision))
    elseif average == :weighted
        weights = [sum(targets .== c) for c in 1:num_classes]
        total = sum(weights)
        return total > 0 ? sum(metrics.precision .* weights) / total : 0.0
    else
        error("Unknown average method: $average")
    end
end

"""
    recall(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int;
           average::Symbol=:macro)

Calculate recall score (sensitivity).

# Arguments
- Same as `f1_score`

# Returns
- `Float64` or `Vector{Float64}`: Recall score(s)
"""
function recall(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int;
                average::Symbol=:macro)
    metrics = precision_recall_f1(predictions, targets, num_classes)

    if average == :none
        return metrics.recall
    elseif average == :macro
        return mean(filter(!isnan, metrics.recall))
    elseif average == :weighted
        weights = [sum(targets .== c) for c in 1:num_classes]
        total = sum(weights)
        return total > 0 ? sum(metrics.recall .* weights) / total : 0.0
    else
        error("Unknown average method: $average")
    end
end

# ============================================================================
# Cohen's Kappa
# ============================================================================

"""
    cohens_kappa(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int)

Calculate Cohen's Kappa coefficient.

Cohen's Kappa measures agreement between predictions and targets,
accounting for agreement by chance.

# Arguments
- `predictions::Vector{Int}`: Predicted class labels
- `targets::Vector{Int}`: Ground truth labels
- `num_classes::Int`: Total number of classes

# Returns
- `Float64`: Kappa coefficient (-1 to 1, where 1 is perfect agreement)

# Example
```julia
kappa = cohens_kappa([1,2,1,2], [1,1,2,2], 2)
```
"""
function cohens_kappa(predictions::Vector{Int}, targets::Vector{Int}, num_classes::Int)
    n = length(predictions)
    cm = confusion_matrix(predictions, targets, num_classes)

    # Observed agreement (accuracy)
    po = sum([cm[c, c] for c in 1:num_classes]) / n

    # Expected agreement by chance
    pe = sum([sum(cm[c, :]) * sum(cm[:, c]) for c in 1:num_classes]) / (n^2)

    # Kappa
    if pe == 1.0
        return 1.0
    else
        return (po - pe) / (1 - pe)
    end
end

# ============================================================================
# Classification Report
# ============================================================================

"""
    classification_report(predictions::Vector{Int}, targets::Vector{Int},
                          num_classes::Int; class_names::Vector{String}=String[])

Generate a comprehensive classification report.

# Arguments
- `predictions::Vector{Int}`: Predicted class labels
- `targets::Vector{Int}`: Ground truth labels
- `num_classes::Int`: Total number of classes
- `class_names::Vector{String}`: Optional class names

# Returns
- `DataFrame`: Classification report with precision, recall, F1, and support
"""
function classification_report(predictions::Vector{Int}, targets::Vector{Int},
                               num_classes::Int; class_names::Vector{String}=String[])
    # Get per-class metrics
    metrics = precision_recall_f1(predictions, targets, num_classes)

    # Calculate support (number of samples per class)
    support = [sum(targets .== c) for c in 1:num_classes]

    # Create class names if not provided
    if isempty(class_names)
        class_names = ["Class $c" for c in 1:num_classes]
    end

    # Create DataFrame
    report = DataFrame(
        Class = class_names,
        Precision = round.(metrics.precision, digits=4),
        Recall = round.(metrics.recall, digits=4),
        F1_Score = round.(metrics.f1, digits=4),
        Support = support
    )

    # Add summary rows
    macro_p = mean(filter(!isnan, metrics.precision))
    macro_r = mean(filter(!isnan, metrics.recall))
    macro_f1 = mean(filter(!isnan, metrics.f1))

    weighted_p = sum(metrics.precision .* support) / sum(support)
    weighted_r = sum(metrics.recall .* support) / sum(support)
    weighted_f1 = sum(metrics.f1 .* support) / sum(support)

    # Calculate overall accuracy
    oa = accuracy(predictions, targets)

    println("\n" * "="^70)
    println("Classification Report")
    println("="^70)
    println(report)
    println("-"^70)
    @printf("Accuracy:  %.4f\n", oa)
    @printf("Macro Avg: Precision=%.4f, Recall=%.4f, F1=%.4f\n", macro_p, macro_r, macro_f1)
    @printf("Weighted:  Precision=%.4f, Recall=%.4f, F1=%.4f\n", weighted_p, weighted_r, weighted_f1)
    @printf("Kappa:     %.4f\n", cohens_kappa(predictions, targets, num_classes))
    println("="^70)

    return report
end

# ============================================================================
# Result Storage
# ============================================================================

"""
    EvaluationResults

Structure to store comprehensive evaluation results.

# Fields
- `accuracy::Float64`: Overall accuracy
- `macro_precision::Float64`: Macro-averaged precision
- `macro_recall::Float64`: Macro-averaged recall
- `macro_f1::Float64`: Macro-averaged F1 score
- `kappa::Float64`: Cohen's Kappa coefficient
- `confusion_matrix::Matrix{Int}`: Confusion matrix
- `per_class_metrics::NamedTuple`: Per-class precision, recall, F1
- `class_names::Vector{String}`: Class names
"""
struct EvaluationResults
    accuracy::Float64
    macro_precision::Float64
    macro_recall::Float64
    macro_f1::Float64
    kappa::Float64
    confusion_matrix::Matrix{Int}
    per_class_metrics::NamedTuple
    class_names::Vector{String}
end

"""
    evaluate_predictions(predictions::Vector{Int}, targets::Vector{Int},
                         num_classes::Int; class_names::Vector{String}=String[])

Compute all evaluation metrics and return structured results.

# Arguments
- `predictions::Vector{Int}`: Predicted class labels
- `targets::Vector{Int}`: Ground truth labels
- `num_classes::Int`: Total number of classes
- `class_names::Vector{String}`: Optional class names

# Returns
- `EvaluationResults`: Complete evaluation results
"""
function evaluate_predictions(predictions::Vector{Int}, targets::Vector{Int},
                              num_classes::Int; class_names::Vector{String}=String[])
    if isempty(class_names)
        class_names = ["Class $c" for c in 1:num_classes]
    end

    # Calculate all metrics
    acc = accuracy(predictions, targets)
    metrics = precision_recall_f1(predictions, targets, num_classes)
    cm = confusion_matrix(predictions, targets, num_classes)
    kappa = cohens_kappa(predictions, targets, num_classes)

    macro_p = mean(filter(!isnan, metrics.precision))
    macro_r = mean(filter(!isnan, metrics.recall))
    macro_f1 = mean(filter(!isnan, metrics.f1))

    return EvaluationResults(
        acc, macro_p, macro_r, macro_f1, kappa,
        cm, metrics, class_names
    )
end
