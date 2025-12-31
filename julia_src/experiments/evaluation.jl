"""
    Evaluation Module for Parcel-Level Farmland Classification

This module provides comprehensive evaluation functionality:
- Model performance metrics (accuracy, F1, precision, recall)
- Confusion matrix analysis
- Per-class performance breakdown
- Statistical significance testing
- Results export and reporting
- Comparison with baseline methods
"""

using Statistics
using Printf
using Random
using Dates

# ============================================================================
# Model Evaluation
# ============================================================================

"""
    evaluate_model(model, test_loader; device=cpu, num_classes::Int=8)

Comprehensive model evaluation on test set.

# Arguments
- `model`: Trained classification model
- `test_loader`: Test data loader
- `device`: Computation device
- `num_classes`: Number of classes

# Returns
- NamedTuple with all evaluation metrics
"""
function evaluate_model(model, test_loader; device=cpu, num_classes::Int=8)
    model = device(model)
    model = Flux.testmode!(model)

    all_predictions = Int[]
    all_labels = Int[]
    all_probs = []

    for (batch_x, batch_y, batch_mask) in test_loader
        batch_x = device(Float32.(batch_x))
        batch_mask = device(batch_mask)

        logits = model(batch_x, batch_mask)
        probs = cpu(softmax(logits, dims=2))

        predictions = vec([idx[2] for idx in argmax(probs, dims=2)])

        append!(all_predictions, predictions)
        append!(all_labels, vec(batch_y))
        push!(all_probs, probs)
    end

    probs_matrix = vcat(all_probs...)

    # Compute metrics
    acc = accuracy(all_predictions, all_labels)
    f1_macro = f1_score(all_predictions, all_labels, num_classes; average=:macro)
    f1_weighted = f1_score(all_predictions, all_labels, num_classes; average=:weighted)
    prec = precision_score(all_predictions, all_labels, num_classes; average=:macro)
    rec = recall_score(all_predictions, all_labels, num_classes; average=:macro)
    cm = confusion_matrix(all_predictions, all_labels, num_classes)
    kappa = cohens_kappa(all_predictions, all_labels, num_classes)

    # Per-class metrics
    per_class_metrics = compute_per_class_metrics(all_predictions, all_labels, num_classes)

    return (
        accuracy = acc,
        f1_macro = f1_macro,
        f1_weighted = f1_weighted,
        precision = prec,
        recall = rec,
        kappa = kappa,
        confusion_matrix = cm,
        per_class = per_class_metrics,
        predictions = all_predictions,
        labels = all_labels,
        probabilities = probs_matrix
    )
end

"""
    compute_per_class_metrics(predictions::Vector{Int}, labels::Vector{Int},
                             num_classes::Int)

Compute metrics for each class individually.

# Returns
- Dict with per-class precision, recall, F1, and support
"""
function compute_per_class_metrics(predictions::Vector{Int}, labels::Vector{Int},
                                  num_classes::Int)
    metrics = Dict{Int, NamedTuple}()

    for c in 1:num_classes
        # True positives, false positives, false negatives
        tp = sum((predictions .== c) .& (labels .== c))
        fp = sum((predictions .== c) .& (labels .!= c))
        fn = sum((predictions .!= c) .& (labels .== c))

        # Precision, recall, F1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, eps())

        # Support (number of true samples)
        support = sum(labels .== c)

        metrics[c] = (
            precision = precision,
            recall = recall,
            f1 = f1,
            support = support
        )
    end

    return metrics
end

# ============================================================================
# Results Reporting
# ============================================================================

# Crop class names for Weigan Farmland dataset
const CROP_NAMES = Dict{Int, String}(
    1 => "Cotton",
    2 => "Corn",
    3 => "Pepper",
    4 => "Jujube",
    5 => "Pear",
    6 => "Apricot",
    7 => "Tomato",
    8 => "Others"
)

"""
    print_evaluation_report(results; crop_names::Dict=CROP_NAMES)

Print comprehensive evaluation report.

# Arguments
- `results`: Results from evaluate_model
- `crop_names`: Dictionary mapping class indices to names
"""
function print_evaluation_report(results; crop_names::Dict=CROP_NAMES)
    println("\n" * "="^70)
    println("                    EVALUATION REPORT")
    println("="^70)

    println("\n📊 Overall Metrics:")
    println("-"^40)
    @printf("  Accuracy:          %.4f (%.2f%%)\n", results.accuracy, results.accuracy * 100)
    @printf("  F1 Score (Macro):  %.4f\n", results.f1_macro)
    @printf("  F1 Score (Weighted): %.4f\n", results.f1_weighted)
    @printf("  Precision (Macro): %.4f\n", results.precision)
    @printf("  Recall (Macro):    %.4f\n", results.recall)
    @printf("  Cohen's Kappa:     %.4f\n", results.kappa)

    println("\n📋 Per-Class Performance:")
    println("-"^70)
    @printf("  %-12s  %10s  %10s  %10s  %10s\n",
            "Class", "Precision", "Recall", "F1-Score", "Support")
    println("-"^70)

    for c in sort(collect(keys(results.per_class)))
        class_name = get(crop_names, c, "Class $c")
        m = results.per_class[c]
        @printf("  %-12s  %10.4f  %10.4f  %10.4f  %10d\n",
                class_name, m.precision, m.recall, m.f1, m.support)
    end
    println("-"^70)

    println("\n🔢 Confusion Matrix:")
    println("-"^40)
    print_confusion_matrix(results.confusion_matrix, crop_names)

    println("\n" * "="^70)
end

"""
    print_confusion_matrix(cm::Matrix{Int}, crop_names::Dict)

Print formatted confusion matrix.
"""
function print_confusion_matrix(cm::Matrix{Int}, crop_names::Dict)
    n = size(cm, 1)

    # Print header
    print("           ")
    for j in 1:n
        name = get(crop_names, j, "C$j")
        short_name = length(name) > 6 ? name[1:6] : name
        @printf("%7s", short_name)
    end
    println()

    # Print rows
    for i in 1:n
        name = get(crop_names, i, "C$i")
        short_name = length(name) > 10 ? name[1:10] : name
        @printf("%-10s ", short_name)
        for j in 1:n
            @printf("%7d", cm[i, j])
        end
        println()
    end
end

"""
    generate_latex_table(results; crop_names::Dict=CROP_NAMES)

Generate LaTeX table for paper publication.

# Returns
- LaTeX table string
"""
function generate_latex_table(results; crop_names::Dict=CROP_NAMES)
    table = """
    \\begin{table}[htbp]
    \\centering
    \\caption{Per-class Classification Performance}
    \\label{tab:performance}
    \\begin{tabular}{lcccc}
    \\toprule
    Class & Precision & Recall & F1-Score & Support \\\\
    \\midrule
    """

    for c in sort(collect(keys(results.per_class)))
        class_name = get(crop_names, c, "Class $c")
        m = results.per_class[c]
        table *= @sprintf("    %s & %.4f & %.4f & %.4f & %d \\\\\n",
                         class_name, m.precision, m.recall, m.f1, m.support)
    end

    table *= """
    \\midrule
    \\textbf{Macro Avg} & \\textbf{$(round(results.precision, digits=4))} & \\textbf{$(round(results.recall, digits=4))} & \\textbf{$(round(results.f1_macro, digits=4))} & $(sum(results.labels .> 0)) \\\\
    \\bottomrule
    \\end{tabular}
    \\end{table}
    """

    return table
end

# ============================================================================
# Model Comparison
# ============================================================================

"""
    compare_models(model_results::Dict{String, NamedTuple})

Compare multiple models' performance.

# Arguments
- `model_results`: Dict mapping model names to evaluation results

# Returns
- Comparison summary
"""
function compare_models(model_results::Dict{String, NamedTuple})
    println("\n" * "="^80)
    println("                         MODEL COMPARISON")
    println("="^80)

    @printf("\n%-20s %10s %10s %10s %10s %10s\n",
            "Model", "Accuracy", "F1-Macro", "F1-Wgt", "Precision", "Kappa")
    println("-"^80)

    best_acc = 0.0
    best_model = ""

    for (name, results) in model_results
        @printf("%-20s %10.4f %10.4f %10.4f %10.4f %10.4f\n",
                name, results.accuracy, results.f1_macro, results.f1_weighted,
                results.precision, results.kappa)

        if results.accuracy > best_acc
            best_acc = results.accuracy
            best_model = name
        end
    end

    println("-"^80)
    println("\n🏆 Best Model: $best_model (Accuracy: $(round(best_acc, digits=4)))")
    println("="^80)

    return (best_model = best_model, best_accuracy = best_acc)
end

# ============================================================================
# Statistical Significance Testing
# ============================================================================

"""
    bootstrap_confidence_interval(predictions::Vector{Int}, labels::Vector{Int};
                                  metric_fn=accuracy, n_bootstrap::Int=1000,
                                  confidence::Float64=0.95)

Compute bootstrap confidence interval for a metric.

# Arguments
- `predictions`: Model predictions
- `labels`: True labels
- `metric_fn`: Metric function to compute
- `n_bootstrap`: Number of bootstrap samples
- `confidence`: Confidence level

# Returns
- NamedTuple with (mean, lower, upper, std)
"""
function bootstrap_confidence_interval(predictions::Vector{Int}, labels::Vector{Int};
                                       metric_fn=accuracy, n_bootstrap::Int=1000,
                                       confidence::Float64=0.95)
    n = length(predictions)
    bootstrap_metrics = Float64[]

    for _ in 1:n_bootstrap
        # Sample with replacement
        indices = rand(1:n, n)
        boot_preds = predictions[indices]
        boot_labels = labels[indices]

        metric = metric_fn(boot_preds, boot_labels)
        push!(bootstrap_metrics, metric)
    end

    # Sort for percentile calculation
    sort!(bootstrap_metrics)

    alpha = (1 - confidence) / 2
    lower_idx = max(1, Int(floor(alpha * n_bootstrap)))
    upper_idx = min(n_bootstrap, Int(ceil((1 - alpha) * n_bootstrap)))

    return (
        mean = mean(bootstrap_metrics),
        lower = bootstrap_metrics[lower_idx],
        upper = bootstrap_metrics[upper_idx],
        std = std(bootstrap_metrics)
    )
end

"""
    mcnemar_test(pred1::Vector{Int}, pred2::Vector{Int}, labels::Vector{Int})

Perform McNemar's test to compare two classifiers.

# Arguments
- `pred1`: Predictions from classifier 1
- `pred2`: Predictions from classifier 2
- `labels`: True labels

# Returns
- NamedTuple with test statistic and p-value
"""
function mcnemar_test(pred1::Vector{Int}, pred2::Vector{Int}, labels::Vector{Int})
    # Correct/incorrect for each classifier
    correct1 = pred1 .== labels
    correct2 = pred2 .== labels

    # Count discordant pairs
    # b: correct by 1, incorrect by 2
    # c: incorrect by 1, correct by 2
    b = sum(correct1 .& .!correct2)
    c = sum(.!correct1 .& correct2)

    # McNemar's test statistic (with continuity correction)
    if b + c > 0
        chi2 = (abs(b - c) - 1)^2 / (b + c)
    else
        chi2 = 0.0
    end

    # P-value (approximate using chi-squared distribution with 1 df)
    # For simplicity, use a lookup table for common values
    # chi2 >= 3.84 -> p < 0.05
    # chi2 >= 6.64 -> p < 0.01
    p_value = chi2 >= 6.64 ? "< 0.01" : (chi2 >= 3.84 ? "< 0.05" : "> 0.05")

    return (chi2 = chi2, p_value = p_value, b = b, c = c)
end

# ============================================================================
# Parcel-Level Evaluation
# ============================================================================

"""
    evaluate_parcel_level(predictions::Vector{Int}, labels::Vector{Int},
                         parcel_ids::Vector{Int}; aggregation::Symbol=:majority)

Evaluate classification at the parcel level.

# Arguments
- `predictions`: Pixel-level predictions
- `labels`: Pixel-level labels
- `parcel_ids`: Parcel ID for each pixel
- `aggregation`: How to aggregate pixel predictions (:majority, :weighted)

# Returns
- Parcel-level evaluation metrics
"""
function evaluate_parcel_level(predictions::Vector{Int}, labels::Vector{Int},
                              parcel_ids::Vector{Int}; aggregation::Symbol=:majority)

    unique_parcels = unique(parcel_ids)
    parcel_predictions = Int[]
    parcel_labels = Int[]

    for parcel in unique_parcels
        parcel_mask = parcel_ids .== parcel
        parcel_preds = predictions[parcel_mask]
        parcel_true = labels[parcel_mask]

        if aggregation == :majority
            # Majority voting
            counts = Dict{Int, Int}()
            for p in parcel_preds
                counts[p] = get(counts, p, 0) + 1
            end
            parcel_pred = argmax(counts)[1]
        else
            # Default to majority
            counts = Dict{Int, Int}()
            for p in parcel_preds
                counts[p] = get(counts, p, 0) + 1
            end
            parcel_pred = argmax(counts)[1]
        end

        # True label is the majority of ground truth
        true_counts = Dict{Int, Int}()
        for t in parcel_true
            true_counts[t] = get(true_counts, t, 0) + 1
        end
        parcel_true_label = argmax(true_counts)[1]

        push!(parcel_predictions, parcel_pred)
        push!(parcel_labels, parcel_true_label)
    end

    # Compute parcel-level metrics
    num_classes = maximum(vcat(parcel_predictions, parcel_labels))
    acc = accuracy(parcel_predictions, parcel_labels)
    f1 = f1_score(parcel_predictions, parcel_labels, num_classes; average=:macro)
    kappa = cohens_kappa(parcel_predictions, parcel_labels, num_classes)

    @info "Parcel-Level Evaluation ($(length(unique_parcels)) parcels):"
    @printf("  Accuracy: %.4f\n", acc)
    @printf("  F1 Score: %.4f\n", f1)
    @printf("  Kappa: %.4f\n", kappa)

    return (
        accuracy = acc,
        f1 = f1,
        kappa = kappa,
        n_parcels = length(unique_parcels),
        parcel_predictions = parcel_predictions,
        parcel_labels = parcel_labels
    )
end

# ============================================================================
# Results Export
# ============================================================================

"""
    export_results(results, output_dir::String; format::Symbol=:all)

Export evaluation results to files.

# Arguments
- `results`: Evaluation results
- `output_dir`: Output directory
- `format`: Export format (:csv, :json, :latex, :all)
"""
function export_results(results, output_dir::String; format::Symbol=:all)
    mkpath(output_dir)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

    if format == :csv || format == :all
        # Export metrics to CSV
        csv_path = joinpath(output_dir, "results_$timestamp.csv")
        open(csv_path, "w") do f
            println(f, "Metric,Value")
            println(f, "Accuracy,$(results.accuracy)")
            println(f, "F1_Macro,$(results.f1_macro)")
            println(f, "F1_Weighted,$(results.f1_weighted)")
            println(f, "Precision,$(results.precision)")
            println(f, "Recall,$(results.recall)")
            println(f, "Kappa,$(results.kappa)")
        end
        @info "Results exported to $csv_path"

        # Export per-class metrics
        class_csv_path = joinpath(output_dir, "per_class_results_$timestamp.csv")
        open(class_csv_path, "w") do f
            println(f, "Class,Precision,Recall,F1,Support")
            for c in sort(collect(keys(results.per_class)))
                m = results.per_class[c]
                class_name = get(CROP_NAMES, c, "Class$c")
                println(f, "$class_name,$(m.precision),$(m.recall),$(m.f1),$(m.support)")
            end
        end
        @info "Per-class results exported to $class_csv_path"

        # Export confusion matrix
        cm_csv_path = joinpath(output_dir, "confusion_matrix_$timestamp.csv")
        open(cm_csv_path, "w") do f
            header = join([get(CROP_NAMES, c, "Class$c") for c in 1:size(results.confusion_matrix, 1)], ",")
            println(f, ",$header")
            for i in 1:size(results.confusion_matrix, 1)
                row_name = get(CROP_NAMES, i, "Class$i")
                row_values = join(results.confusion_matrix[i, :], ",")
                println(f, "$row_name,$row_values")
            end
        end
        @info "Confusion matrix exported to $cm_csv_path"
    end

    if format == :latex || format == :all
        # Export LaTeX table
        latex_path = joinpath(output_dir, "results_table_$timestamp.tex")
        open(latex_path, "w") do f
            print(f, generate_latex_table(results))
        end
        @info "LaTeX table exported to $latex_path"
    end

    if format == :json || format == :all
        # Export as JSON-like format
        json_path = joinpath(output_dir, "results_$timestamp.json")
        open(json_path, "w") do f
            println(f, "{")
            println(f, "  \"accuracy\": $(results.accuracy),")
            println(f, "  \"f1_macro\": $(results.f1_macro),")
            println(f, "  \"f1_weighted\": $(results.f1_weighted),")
            println(f, "  \"precision\": $(results.precision),")
            println(f, "  \"recall\": $(results.recall),")
            println(f, "  \"kappa\": $(results.kappa)")
            println(f, "}")
        end
        @info "JSON results exported to $json_path"
    end
end

# ============================================================================
# Benchmark Functions
# ============================================================================

"""
    run_benchmark(model_configs::Vector{Config}, train_loader, test_loader;
                 device=cpu, n_runs::Int=5)

Run benchmark comparison of multiple model configurations.

# Arguments
- `model_configs`: Vector of model configurations
- `train_loader`: Training data loader
- `test_loader`: Test data loader
- `device`: Computation device
- `n_runs`: Number of runs per configuration

# Returns
- Benchmark results with mean and std for each model
"""
function run_benchmark(model_configs::Vector{Config}, train_loader, test_loader;
                      device=cpu, n_runs::Int=5)

    results = Dict{String, NamedTuple}()

    for config in model_configs
        model_name = config.model_name
        println("\n" * "="^50)
        println("Benchmarking: $model_name ($n_runs runs)")
        println("="^50)

        accuracies = Float64[]
        f1_scores = Float64[]
        train_times = Float64[]

        for run in 1:n_runs
            println("\nRun $run/$n_runs")

            # Create model
            model = create_model(config)

            # Train
            start_time = time()
            train_result = train_model!(model, train_loader, test_loader, config;
                                       device=device)
            train_time = time() - start_time

            # Evaluate
            eval_result = evaluate_model(train_result.model, test_loader;
                                        device=device, num_classes=config.num_class)

            push!(accuracies, eval_result.accuracy)
            push!(f1_scores, eval_result.f1_macro)
            push!(train_times, train_time)

            @printf("  Accuracy: %.4f, F1: %.4f, Time: %.1fs\n",
                    eval_result.accuracy, eval_result.f1_macro, train_time)
        end

        results[model_name] = (
            accuracy_mean = mean(accuracies),
            accuracy_std = std(accuracies),
            f1_mean = mean(f1_scores),
            f1_std = std(f1_scores),
            train_time_mean = mean(train_times),
            train_time_std = std(train_times)
        )
    end

    # Print benchmark summary
    println("\n" * "="^80)
    println("                        BENCHMARK SUMMARY")
    println("="^80)
    @printf("\n%-15s %15s %15s %15s\n",
            "Model", "Accuracy", "F1-Macro", "Train Time (s)")
    println("-"^80)

    for (name, res) in results
        @printf("%-15s %7.4f±%.4f %7.4f±%.4f %7.1f±%.1f\n",
                name, res.accuracy_mean, res.accuracy_std,
                res.f1_mean, res.f1_std,
                res.train_time_mean, res.train_time_std)
    end
    println("="^80)

    return results
end

"""
    quick_evaluate(model, test_data::FarmlandDataset; device=cpu)

Quick evaluation function for rapid testing.

# Arguments
- `model`: Trained model
- `test_data`: Test dataset
- `device`: Computation device

# Returns
- Basic evaluation metrics
"""
function quick_evaluate(model, test_data; device=cpu)
    # Create test loader
    batch_size = min(64, length(test_data))
    test_loader = create_dataloaders(test_data, test_data; batch_size=batch_size)[2]

    results = evaluate_model(model, test_loader; device=device,
                            num_classes=test_data.num_classes)

    @printf("\nQuick Evaluation Results:\n")
    @printf("  Accuracy: %.4f\n", results.accuracy)
    @printf("  F1 (Macro): %.4f\n", results.f1_macro)
    @printf("  Kappa: %.4f\n", results.kappa)

    return results
end
