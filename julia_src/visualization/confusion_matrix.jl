"""
    Confusion Matrix Visualization Module

This module provides specialized visualization for confusion matrices:
- High-quality publication-ready confusion matrix plots
- Normalized and absolute value displays
- Per-class accuracy annotations
- Export to various formats
"""

using Plots
using Printf
using Statistics

# Crop class names
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

# ============================================================================
# Confusion Matrix Plotting
# ============================================================================

"""
    plot_confusion_matrix(cm::Matrix{Int};
                         class_names::Vector{String}=String[],
                         normalize::Bool=false,
                         title::String="Confusion Matrix",
                         cmap::Symbol=:Blues,
                         save_path::String="")

Create publication-quality confusion matrix visualization.

# Arguments
- `cm`: Confusion matrix (true labels × predicted labels)
- `class_names`: Names for each class
- `normalize`: If true, normalize rows to percentages
- `title`: Plot title
- `cmap`: Color map
- `save_path`: Path to save figure

# Returns
- Plot object
"""
function plot_confusion_matrix(cm::Matrix{Int};
                              class_names::Vector{String}=String[],
                              normalize::Bool=false,
                              title::String="Confusion Matrix",
                              cmap::Symbol=:Blues,
                              save_path::String="")

    n_classes = size(cm, 1)

    # Default class names
    if isempty(class_names)
        class_names = [get(CROP_NAMES, i, "Class $i") for i in 1:n_classes]
    end

    # Normalize if requested
    if normalize
        cm_display = zeros(Float64, n_classes, n_classes)
        for i in 1:n_classes
            row_sum = sum(cm[i, :])
            if row_sum > 0
                cm_display[i, :] = cm[i, :] ./ row_sum * 100
            end
        end
        value_format = "%.1f%%"
    else
        cm_display = Float64.(cm)
        value_format = "%d"
    end

    # Create heatmap
    p = heatmap(
        1:n_classes, 1:n_classes, cm_display,
        xticks = (1:n_classes, class_names),
        yticks = (1:n_classes, class_names),
        xlabel = "Predicted Label",
        ylabel = "True Label",
        title = title,
        color = cmap,
        aspect_ratio = 1,
        size = (700, 600),
        clims = normalize ? (0, 100) : (0, maximum(cm)),
        yflip = true,  # Put first class at top
        xrotation = 45
    )

    # Add text annotations
    for i in 1:n_classes
        for j in 1:n_classes
            value = cm_display[i, j]
            # Choose text color based on cell brightness
            text_color = value > maximum(cm_display) * 0.5 ? :white : :black

            if normalize
                label = @sprintf("%.1f%%", value)
            else
                label = @sprintf("%d", Int(value))
            end

            annotate!(p, j, i, text(label, 10, text_color, :center))
        end
    end

    if !isempty(save_path)
        savefig(p, save_path)
        @info "Confusion matrix saved to $save_path"
    end

    return p
end

"""
    plot_confusion_matrix_with_metrics(cm::Matrix{Int};
                                      class_names::Vector{String}=String[],
                                      save_path::String="")

Create confusion matrix with per-class precision and recall annotations.

# Arguments
- `cm`: Confusion matrix
- `class_names`: Class names
- `save_path`: Path to save figure

# Returns
- Plot object with additional metrics
"""
function plot_confusion_matrix_with_metrics(cm::Matrix{Int};
                                           class_names::Vector{String}=String[],
                                           save_path::String="")

    n_classes = size(cm, 1)

    if isempty(class_names)
        class_names = [get(CROP_NAMES, i, "Class $i") for i in 1:n_classes]
    end

    # Compute metrics
    precisions = Float64[]
    recalls = Float64[]

    for i in 1:n_classes
        # Precision: TP / (TP + FP)
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        precision = tp / max(tp + fp, 1)
        push!(precisions, precision)

        # Recall: TP / (TP + FN)
        fn = sum(cm[i, :]) - tp
        recall = tp / max(tp + fn, 1)
        push!(recalls, recall)
    end

    # Normalize confusion matrix for display
    cm_norm = zeros(Float64, n_classes, n_classes)
    for i in 1:n_classes
        row_sum = sum(cm[i, :])
        if row_sum > 0
            cm_norm[i, :] = cm[i, :] ./ row_sum * 100
        end
    end

    # Create extended matrix with precision/recall
    n_extended = n_classes + 2
    cm_extended = zeros(Float64, n_extended, n_extended)
    cm_extended[1:n_classes, 1:n_classes] = cm_norm

    # Add precision row
    cm_extended[n_classes+1, 1:n_classes] = precisions .* 100

    # Add recall column
    cm_extended[1:n_classes, n_classes+1] = recalls .* 100

    # Overall accuracy in corner
    overall_acc = sum([cm[i,i] for i in 1:n_classes]) / sum(cm) * 100
    cm_extended[n_classes+1, n_classes+1] = overall_acc

    # Extended labels
    extended_names = vcat(class_names, ["Precision", ""])
    extended_names_y = vcat(class_names, ["Recall", ""])

    p = heatmap(
        1:n_extended, 1:n_extended, cm_extended,
        xticks = (1:n_extended, extended_names),
        yticks = (1:n_extended, extended_names_y),
        xlabel = "Predicted Label",
        ylabel = "True Label",
        title = "Confusion Matrix with Metrics",
        color = :Blues,
        aspect_ratio = 1,
        size = (800, 700),
        yflip = true,
        xrotation = 45
    )

    # Add text annotations
    for i in 1:n_extended
        for j in 1:n_extended
            if i <= n_classes && j <= n_classes
                # Main confusion matrix
                value = cm_extended[i, j]
                text_color = value > 50 ? :white : :black
                annotate!(p, j, i, text(@sprintf("%.1f%%", value), 9, text_color, :center))
            elseif i == n_classes + 1 && j <= n_classes
                # Precision row
                value = cm_extended[i, j]
                annotate!(p, j, i, text(@sprintf("%.1f%%", value), 9, :black, :center))
            elseif j == n_classes + 1 && i <= n_classes
                # Recall column
                value = cm_extended[i, j]
                annotate!(p, j, i, text(@sprintf("%.1f%%", value), 9, :black, :center))
            elseif i == n_classes + 1 && j == n_classes + 1
                # Overall accuracy
                value = cm_extended[i, j]
                annotate!(p, j, i, text(@sprintf("OA: %.1f%%", value), 9, :black, :center))
            end
        end
    end

    if !isempty(save_path)
        savefig(p, save_path)
    end

    return p
end

"""
    plot_dual_confusion_matrix(cm_abs::Matrix{Int}, cm_norm::Matrix{Float64};
                              class_names::Vector{String}=String[],
                              save_path::String="")

Create side-by-side confusion matrices showing absolute counts and percentages.
"""
function plot_dual_confusion_matrix(cm::Matrix{Int};
                                   class_names::Vector{String}=String[],
                                   save_path::String="")

    n_classes = size(cm, 1)

    if isempty(class_names)
        class_names = [get(CROP_NAMES, i, "Class $i") for i in 1:n_classes]
    end

    # Absolute confusion matrix
    p1 = plot_confusion_matrix(cm;
                              class_names=class_names,
                              normalize=false,
                              title="Confusion Matrix (Counts)")

    # Normalized confusion matrix
    p2 = plot_confusion_matrix(cm;
                              class_names=class_names,
                              normalize=true,
                              title="Confusion Matrix (Normalized)")

    combined = plot(p1, p2, layout = (1, 2), size = (1400, 600))

    if !isempty(save_path)
        savefig(combined, save_path)
    end

    return combined
end

# ============================================================================
# Detailed Analysis Plots
# ============================================================================

"""
    plot_class_performance(cm::Matrix{Int};
                          class_names::Vector{String}=String[],
                          save_path::String="")

Create bar chart showing per-class precision, recall, and F1.
"""
function plot_class_performance(cm::Matrix{Int};
                               class_names::Vector{String}=String[],
                               save_path::String="")

    n_classes = size(cm, 1)

    if isempty(class_names)
        class_names = [get(CROP_NAMES, i, "Class $i") for i in 1:n_classes]
    end

    # Compute metrics
    precisions = Float64[]
    recalls = Float64[]
    f1_scores = Float64[]
    supports = Int[]

    for i in 1:n_classes
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, eps())

        push!(precisions, prec)
        push!(recalls, rec)
        push!(f1_scores, f1)
        push!(supports, sum(cm[i, :]))
    end

    # Create grouped bar chart
    p = groupedbar(
        class_names,
        [precisions recalls f1_scores],
        label = ["Precision" "Recall" "F1-Score"],
        xlabel = "Crop Class",
        ylabel = "Score",
        title = "Per-Class Classification Performance",
        color = [:steelblue :coral :seagreen],
        legend = :bottomright,
        xrotation = 45,
        ylims = (0, 1.05),
        size = (900, 500)
    )

    # Add reference lines
    hline!(p, [mean(precisions)], label = "", color = :steelblue, linestyle = :dash, alpha = 0.5)
    hline!(p, [mean(recalls)], label = "", color = :coral, linestyle = :dash, alpha = 0.5)
    hline!(p, [mean(f1_scores)], label = "", color = :seagreen, linestyle = :dash, alpha = 0.5)

    if !isempty(save_path)
        savefig(p, save_path)
    end

    return p
end

"""
    plot_misclassification_analysis(cm::Matrix{Int};
                                   class_names::Vector{String}=String[],
                                   top_n::Int=10,
                                   save_path::String="")

Analyze and visualize the most common misclassifications.
"""
function plot_misclassification_analysis(cm::Matrix{Int};
                                        class_names::Vector{String}=String[],
                                        top_n::Int=10,
                                        save_path::String="")

    n_classes = size(cm, 1)

    if isempty(class_names)
        class_names = [get(CROP_NAMES, i, "Class $i") for i in 1:n_classes]
    end

    # Find top misclassifications (off-diagonal elements)
    misclass = []
    for i in 1:n_classes
        for j in 1:n_classes
            if i != j && cm[i, j] > 0
                push!(misclass, (
                    count = cm[i, j],
                    true_class = class_names[i],
                    pred_class = class_names[j],
                    true_idx = i,
                    pred_idx = j
                ))
            end
        end
    end

    # Sort by count
    sort!(misclass, by = x -> x.count, rev = true)
    top_misclass = misclass[1:min(top_n, length(misclass))]

    # Create bar chart
    labels = ["$(m.true_class) → $(m.pred_class)" for m in top_misclass]
    counts = [m.count for m in top_misclass]

    p = bar(
        labels, counts,
        xlabel = "Misclassification Type",
        ylabel = "Count",
        title = "Top $top_n Misclassifications",
        color = :indianred,
        legend = false,
        xrotation = 45,
        size = (1000, 500)
    )

    if !isempty(save_path)
        savefig(p, save_path)
    end

    return p
end

"""
    plot_confusion_matrix_comparison(cms::Vector{Matrix{Int}}, model_names::Vector{String};
                                    class_names::Vector{String}=String[],
                                    save_path::String="")

Compare confusion matrices from multiple models.
"""
function plot_confusion_matrix_comparison(cms::Vector{Matrix{Int}},
                                         model_names::Vector{String};
                                         class_names::Vector{String}=String[],
                                         save_path::String="")

    n_models = length(cms)
    n_classes = size(cms[1], 1)

    if isempty(class_names)
        class_names = [get(CROP_NAMES, i, "Class $i") for i in 1:n_classes]
    end

    # Per-class accuracy for each model
    accuracies = zeros(n_models, n_classes)

    for (m, cm) in enumerate(cms)
        for i in 1:n_classes
            total = sum(cm[i, :])
            accuracies[m, i] = total > 0 ? cm[i, i] / total * 100 : 0
        end
    end

    # Create grouped bar chart
    p = groupedbar(
        class_names,
        accuracies',
        label = reshape(model_names, 1, :),
        xlabel = "Crop Class",
        ylabel = "Accuracy (%)",
        title = "Per-Class Accuracy Comparison",
        legend = :bottomright,
        xrotation = 45,
        ylims = (0, 105),
        size = (1000, 500)
    )

    if !isempty(save_path)
        savefig(p, save_path)
    end

    return p
end

# ============================================================================
# LaTeX Export
# ============================================================================

"""
    confusion_matrix_to_latex(cm::Matrix{Int};
                             class_names::Vector{String}=String[],
                             normalize::Bool=false)

Generate LaTeX table for confusion matrix.

# Returns
- LaTeX table string
"""
function confusion_matrix_to_latex(cm::Matrix{Int};
                                  class_names::Vector{String}=String[],
                                  normalize::Bool=false)

    n_classes = size(cm, 1)

    if isempty(class_names)
        class_names = [get(CROP_NAMES, i, "Class $i") for i in 1:n_classes]
    end

    # Normalize if requested
    if normalize
        cm_display = zeros(Float64, n_classes, n_classes)
        for i in 1:n_classes
            row_sum = sum(cm[i, :])
            if row_sum > 0
                cm_display[i, :] = cm[i, :] ./ row_sum * 100
            end
        end
    else
        cm_display = Float64.(cm)
    end

    # Build LaTeX table
    col_spec = "l" * repeat("c", n_classes)

    latex = """
    \\begin{table}[htbp]
    \\centering
    \\caption{Confusion Matrix}
    \\label{tab:confusion_matrix}
    \\begin{tabular}{$col_spec}
    \\toprule
    """

    # Header
    header = " & " * join(class_names, " & ") * " \\\\"
    latex *= header * "\n\\midrule\n"

    # Rows
    for i in 1:n_classes
        row = class_names[i]
        for j in 1:n_classes
            if normalize
                row *= @sprintf(" & %.1f\\%%", cm_display[i, j])
            else
                row *= @sprintf(" & %d", Int(cm_display[i, j]))
            end
        end
        row *= " \\\\\n"
        latex *= row
    end

    latex *= """
    \\bottomrule
    \\end{tabular}
    \\end{table}
    """

    return latex
end

# ============================================================================
# Interactive Visualization (for notebooks)
# ============================================================================

"""
    interactive_confusion_matrix(cm::Matrix{Int}; class_names::Vector{String}=String[])

Create interactive confusion matrix visualization.

Note: Full interactivity requires PlotlyJS backend.
"""
function interactive_confusion_matrix(cm::Matrix{Int};
                                     class_names::Vector{String}=String[])
    # For now, return static plot
    # Full interactivity would require PlotlyJS or similar
    return plot_confusion_matrix(cm; class_names=class_names)
end

# ============================================================================
# Summary Statistics
# ============================================================================

"""
    confusion_matrix_summary(cm::Matrix{Int}; class_names::Vector{String}=String[])

Print comprehensive summary statistics from confusion matrix.
"""
function confusion_matrix_summary(cm::Matrix{Int}; class_names::Vector{String}=String[])
    n_classes = size(cm, 1)

    if isempty(class_names)
        class_names = [get(CROP_NAMES, i, "Class $i") for i in 1:n_classes]
    end

    println("\n" * "="^60)
    println("CONFUSION MATRIX SUMMARY")
    println("="^60)

    # Overall accuracy
    total = sum(cm)
    correct = sum([cm[i,i] for i in 1:n_classes])
    overall_acc = correct / total * 100

    println("\n📊 Overall Statistics:")
    println("-"^40)
    @printf("  Total samples: %d\n", total)
    @printf("  Correct predictions: %d\n", correct)
    @printf("  Overall accuracy: %.2f%%\n", overall_acc)

    # Per-class statistics
    println("\n📋 Per-Class Statistics:")
    println("-"^60)
    @printf("  %-12s  %8s  %8s  %8s  %8s  %8s\n",
            "Class", "Support", "TP", "Prec", "Recall", "F1")
    println("-"^60)

    macro_prec = 0.0
    macro_rec = 0.0
    macro_f1 = 0.0

    for i in 1:n_classes
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        support = sum(cm[i, :])

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, eps())

        @printf("  %-12s  %8d  %8d  %7.2f%%  %7.2f%%  %7.2f%%\n",
                class_names[i], support, tp, prec*100, rec*100, f1*100)

        macro_prec += prec
        macro_rec += rec
        macro_f1 += f1
    end

    macro_prec /= n_classes
    macro_rec /= n_classes
    macro_f1 /= n_classes

    println("-"^60)
    @printf("  %-12s  %8s  %8s  %7.2f%%  %7.2f%%  %7.2f%%\n",
            "Macro Avg", "", "", macro_prec*100, macro_rec*100, macro_f1*100)

    # Most confused pairs
    println("\n🔄 Top Misclassifications:")
    println("-"^40)

    misclass = []
    for i in 1:n_classes
        for j in 1:n_classes
            if i != j && cm[i, j] > 0
                push!(misclass, (count=cm[i,j], from=i, to=j))
            end
        end
    end

    sort!(misclass, by = x -> x.count, rev = true)

    for (idx, m) in enumerate(misclass[1:min(5, length(misclass))])
        @printf("  %d. %s → %s: %d samples\n",
                idx, class_names[m.from], class_names[m.to], m.count)
    end

    println("\n" * "="^60)
end
