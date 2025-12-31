"""
    Visualization Module for Parcel-Level Farmland Classification

This module provides visualization functions for:
- Time series data plotting
- Spectral indices visualization
- Training curves
- Model attention weights
- Feature importance
- NDVI temporal profiles
"""

using Plots
using Statistics
using Printf

# Set default plot settings
default(
    fontfamily = "Computer Modern",
    linewidth = 2,
    framestyle = :box,
    grid = true,
    gridalpha = 0.3,
    legend = :topright,
    size = (800, 600),
    dpi = 300
)

# Color palette for crop classes (matching paper)
const CROP_COLORS = Dict(
    1 => :blue,       # Cotton
    2 => :green,      # Corn
    3 => :red,        # Pepper
    4 => :purple,     # Jujube
    5 => :orange,     # Pear
    6 => :brown,      # Apricot
    7 => :pink,       # Tomato
    8 => :gray        # Others
)

const CROP_NAMES = Dict(
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
# Time Series Visualization
# ============================================================================

"""
    plot_time_series(data::AbstractMatrix, labels::AbstractVector;
                    feature_idx::Int=1, n_samples::Int=5, title::String="")

Plot time series data with class labels.

# Arguments
- `data`: Time series data (n_samples × seq_len × features)
- `labels`: Class labels
- `feature_idx`: Which feature to plot
- `n_samples`: Number of samples per class to plot
- `title`: Plot title

# Returns
- Plot object
"""
function plot_time_series(data::AbstractArray{T, 3}, labels::AbstractVector;
                         feature_idx::Int=1, n_samples::Int=5,
                         title::String="Time Series by Crop Type") where T

    seq_len = size(data, 2)
    time_axis = 1:seq_len

    p = plot(
        xlabel = "Time Step",
        ylabel = "Feature Value",
        title = title,
        legend = :outerright
    )

    unique_classes = sort(unique(labels))

    for class in unique_classes
        class_indices = findall(labels .== class)
        n_plot = min(n_samples, length(class_indices))

        for i in 1:n_plot
            idx = class_indices[i]
            series = data[idx, :, feature_idx]

            plot!(p, time_axis, series,
                  color = get(CROP_COLORS, class, :black),
                  label = i == 1 ? get(CROP_NAMES, class, "Class $class") : "",
                  alpha = 0.7)
        end
    end

    return p
end

"""
    plot_ndvi_profiles(dataset; n_samples_per_class::Int=10)

Plot NDVI temporal profiles for all crop classes.

This replicates the NDVI profile visualization from the paper.

# Arguments
- `dataset`: FarmlandDataset with computed NDVI
- `n_samples_per_class`: Number of samples to plot per class

# Returns
- Multi-panel plot with NDVI profiles
"""
function plot_ndvi_profiles(data::AbstractArray{T, 3}, labels::AbstractVector;
                           n_samples_per_class::Int=10,
                           ndvi_idx::Int=1) where T

    unique_classes = sort(unique(labels))
    n_classes = length(unique_classes)
    seq_len = size(data, 2)

    # Create time axis (approximate DOY for growing season)
    # Assuming 24 time steps from March to November
    doy_start = 60  # Early March
    doy_end = 330   # Late November
    time_axis = range(doy_start, doy_end, length=seq_len)

    # Create subplot for each class
    plots_list = []

    for (i, class) in enumerate(unique_classes)
        class_name = get(CROP_NAMES, class, "Class $class")
        class_color = get(CROP_COLORS, class, :black)

        class_indices = findall(labels .== class)
        n_plot = min(n_samples_per_class, length(class_indices))

        p = plot(
            xlabel = i > n_classes - 2 ? "Day of Year" : "",
            ylabel = "NDVI",
            title = class_name,
            ylims = (-0.2, 1.0),
            legend = false
        )

        # Plot individual samples
        for j in 1:n_plot
            idx = class_indices[j]
            ndvi = data[idx, :, ndvi_idx]
            plot!(p, time_axis, ndvi, color = class_color, alpha = 0.3)
        end

        # Plot mean NDVI
        all_ndvi = data[class_indices, :, ndvi_idx]
        mean_ndvi = vec(mean(all_ndvi, dims=1))
        plot!(p, time_axis, mean_ndvi, color = class_color, linewidth = 3)

        push!(plots_list, p)
    end

    # Arrange in grid
    n_cols = 4
    n_rows = ceil(Int, n_classes / n_cols)

    combined = plot(plots_list...,
                   layout = (n_rows, n_cols),
                   size = (1200, 300 * n_rows),
                   dpi = 300)

    return combined
end

"""
    plot_spectral_indices(data::AbstractArray, labels::AbstractVector;
                         indices::Vector{String}=["NDVI", "MNDWI", "NDVIRE1"])

Plot multiple spectral indices for comparison.

# Arguments
- `data`: Time series data with multiple spectral indices
- `labels`: Class labels
- `indices`: Names of spectral indices to plot

# Returns
- Multi-panel plot comparing spectral indices
"""
function plot_spectral_indices(data::AbstractArray{T, 3}, labels::AbstractVector;
                              indices::Vector{String}=["NDVI", "MNDWI", "NDVIRE1"],
                              class_to_plot::Int=1) where T

    n_indices = min(length(indices), size(data, 3))
    seq_len = size(data, 2)
    time_axis = 1:seq_len

    class_indices = findall(labels .== class_to_plot)
    class_name = get(CROP_NAMES, class_to_plot, "Class $class_to_plot")

    plots_list = []

    for i in 1:n_indices
        index_name = i <= length(indices) ? indices[i] : "Feature $i"

        p = plot(
            xlabel = i == n_indices ? "Time Step" : "",
            ylabel = index_name,
            title = "$index_name - $class_name",
            legend = false
        )

        # Plot samples
        for idx in class_indices[1:min(10, length(class_indices))]
            series = data[idx, :, i]
            plot!(p, time_axis, series,
                  color = get(CROP_COLORS, class_to_plot, :blue),
                  alpha = 0.5)
        end

        # Plot mean
        all_data = data[class_indices, :, i]
        mean_data = vec(mean(all_data, dims=1))
        plot!(p, time_axis, mean_data,
              color = get(CROP_COLORS, class_to_plot, :blue),
              linewidth = 3)

        push!(plots_list, p)
    end

    combined = plot(plots_list..., layout = (1, n_indices), size = (400 * n_indices, 400))
    return combined
end

# ============================================================================
# Training Visualization
# ============================================================================

"""
    plot_training_curves(logger::TrainingLogger; save_path::String="")

Plot training and validation curves.

# Arguments
- `logger`: TrainingLogger with training history
- `save_path`: Optional path to save figure

# Returns
- Combined plot with loss and accuracy curves
"""
function plot_training_curves(logger; save_path::String="")
    epochs = logger.epochs

    # Loss plot
    p1 = plot(
        epochs, logger.train_losses,
        label = "Train Loss",
        xlabel = "Epoch",
        ylabel = "Loss",
        title = "Training and Validation Loss",
        color = :blue,
        linewidth = 2
    )
    plot!(p1, epochs, logger.val_losses,
          label = "Val Loss",
          color = :red,
          linewidth = 2)

    # Accuracy plot
    p2 = plot(
        epochs, logger.train_accuracies,
        label = "Train Accuracy",
        xlabel = "Epoch",
        ylabel = "Accuracy",
        title = "Training and Validation Accuracy",
        color = :blue,
        linewidth = 2
    )
    plot!(p2, epochs, logger.val_accuracies,
          label = "Val Accuracy",
          color = :red,
          linewidth = 2)

    # Learning rate plot
    p3 = plot(
        epochs, logger.learning_rates,
        label = "Learning Rate",
        xlabel = "Epoch",
        ylabel = "Learning Rate",
        title = "Learning Rate Schedule",
        color = :green,
        linewidth = 2,
        yscale = :log10
    )

    # Combine plots
    combined = plot(p1, p2, p3, layout = (1, 3), size = (1500, 400))

    if !isempty(save_path)
        savefig(combined, save_path)
        @info "Training curves saved to $save_path"
    end

    return combined
end

"""
    plot_learning_rate_schedule(scheduler, n_epochs::Int; save_path::String="")

Visualize learning rate schedule.
"""
function plot_learning_rate_schedule(scheduler, n_epochs::Int; save_path::String="")
    epochs = 1:n_epochs
    lrs = [scheduler(e) for e in epochs]

    p = plot(
        epochs, lrs,
        xlabel = "Epoch",
        ylabel = "Learning Rate",
        title = "Learning Rate Schedule",
        color = :blue,
        linewidth = 2,
        legend = false,
        yscale = :log10
    )

    if !isempty(save_path)
        savefig(p, save_path)
    end

    return p
end

# ============================================================================
# Self-Training Visualization
# ============================================================================

"""
    plot_self_training_progress(history::Dict; save_path::String="")

Visualize self-training iteration progress.

# Arguments
- `history`: Training history from iterative_self_training
- `save_path`: Optional path to save figure

# Returns
- Plot showing accuracy and pseudo-label growth
"""
function plot_self_training_progress(history::Dict; save_path::String="")
    iterations = 1:length(history["val_accuracy"])

    # Accuracy plot
    p1 = plot(
        iterations, history["val_accuracy"],
        label = "Validation Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy",
        title = "Self-Training Progress",
        color = :blue,
        linewidth = 2,
        marker = :circle
    )

    # Pseudo-label count
    p2 = plot(
        iterations, history["pseudo_count"],
        label = "Total Pseudo-Labels",
        xlabel = "Iteration",
        ylabel = "Count",
        title = "Pseudo-Label Accumulation",
        color = :green,
        linewidth = 2,
        marker = :square
    )

    # Threshold decay
    p3 = plot(
        iterations, history["threshold"],
        label = "Confidence Threshold",
        xlabel = "Iteration",
        ylabel = "Threshold",
        title = "Threshold Decay",
        color = :red,
        linewidth = 2,
        marker = :diamond
    )

    combined = plot(p1, p2, p3, layout = (1, 3), size = (1500, 400))

    if !isempty(save_path)
        savefig(combined, save_path)
    end

    return combined
end

# ============================================================================
# Feature Importance Visualization
# ============================================================================

"""
    plot_feature_importance(importance::Vector{Float64}, feature_names::Vector{String};
                           top_n::Int=20, save_path::String="")

Plot feature importance bar chart.
"""
function plot_feature_importance(importance::Vector{Float64},
                                feature_names::Vector{String};
                                top_n::Int=20,
                                save_path::String="")

    # Sort by importance
    sorted_idx = sortperm(importance, rev=true)
    top_idx = sorted_idx[1:min(top_n, length(importance))]

    top_names = feature_names[top_idx]
    top_importance = importance[top_idx]

    p = bar(
        top_names, top_importance,
        xlabel = "Feature",
        ylabel = "Importance",
        title = "Top $top_n Feature Importance",
        color = :steelblue,
        legend = false,
        xrotation = 45,
        size = (1000, 600)
    )

    if !isempty(save_path)
        savefig(p, save_path)
    end

    return p
end

# ============================================================================
# Attention Visualization
# ============================================================================

"""
    plot_attention_weights(attention::AbstractMatrix; save_path::String="")

Visualize attention weights as heatmap.

# Arguments
- `attention`: Attention weight matrix (seq_len × seq_len)
- `save_path`: Optional path to save figure

# Returns
- Heatmap plot
"""
function plot_attention_weights(attention::AbstractMatrix; save_path::String="")
    seq_len = size(attention, 1)

    p = heatmap(
        1:seq_len, 1:seq_len, attention,
        xlabel = "Key Position",
        ylabel = "Query Position",
        title = "Attention Weights",
        color = :viridis,
        aspect_ratio = 1
    )

    if !isempty(save_path)
        savefig(p, save_path)
    end

    return p
end

"""
    plot_multi_head_attention(attentions::AbstractArray{T, 3}; save_path::String="") where T

Visualize attention weights from multiple heads.

# Arguments
- `attentions`: Attention weights (n_heads × seq_len × seq_len)
- `save_path`: Optional path to save figure
"""
function plot_multi_head_attention(attentions::AbstractArray{T, 3};
                                   save_path::String="") where T
    n_heads = size(attentions, 1)
    seq_len = size(attentions, 2)

    plots_list = []

    for h in 1:min(n_heads, 8)  # Limit to 8 heads for display
        p = heatmap(
            1:seq_len, 1:seq_len, attentions[h, :, :],
            title = "Head $h",
            color = :viridis,
            aspect_ratio = 1,
            colorbar = false
        )
        push!(plots_list, p)
    end

    n_cols = min(4, n_heads)
    n_rows = ceil(Int, min(n_heads, 8) / n_cols)

    combined = plot(plots_list..., layout = (n_rows, n_cols),
                   size = (300 * n_cols, 300 * n_rows))

    if !isempty(save_path)
        savefig(combined, save_path)
    end

    return combined
end

# ============================================================================
# Model Comparison Plots
# ============================================================================

"""
    plot_model_comparison(results::Dict{String, NamedTuple}; save_path::String="")

Create bar chart comparing model performance.
"""
function plot_model_comparison(results::Dict{String, NamedTuple}; save_path::String="")
    model_names = collect(keys(results))
    accuracies = [results[m].accuracy for m in model_names]
    f1_scores = [results[m].f1_macro for m in model_names]

    p = groupedbar(
        model_names,
        [accuracies f1_scores],
        label = ["Accuracy" "F1-Macro"],
        xlabel = "Model",
        ylabel = "Score",
        title = "Model Performance Comparison",
        color = [:steelblue :coral],
        legend = :topright
    )

    if !isempty(save_path)
        savefig(p, save_path)
    end

    return p
end

"""
    plot_benchmark_results(benchmark_results::Dict; save_path::String="")

Visualize benchmark results with error bars.
"""
function plot_benchmark_results(benchmark_results::Dict; save_path::String="")
    model_names = collect(keys(benchmark_results))
    n_models = length(model_names)

    acc_means = [benchmark_results[m].accuracy_mean for m in model_names]
    acc_stds = [benchmark_results[m].accuracy_std for m in model_names]
    f1_means = [benchmark_results[m].f1_mean for m in model_names]
    f1_stds = [benchmark_results[m].f1_std for m in model_names]

    x = 1:n_models

    p = scatter(
        x .- 0.15, acc_means,
        yerror = acc_stds,
        label = "Accuracy",
        xlabel = "Model",
        ylabel = "Score",
        title = "Benchmark Results (Mean ± Std)",
        color = :steelblue,
        markersize = 8,
        xticks = (x, model_names)
    )

    scatter!(p, x .+ 0.15, f1_means,
            yerror = f1_stds,
            label = "F1-Macro",
            color = :coral,
            markersize = 8)

    if !isempty(save_path)
        savefig(p, save_path)
    end

    return p
end

# ============================================================================
# Parcel Map Visualization
# ============================================================================

"""
    plot_classification_map(predictions::Matrix{Int}, geo_transform;
                           crop_names::Dict=CROP_NAMES, save_path::String="")

Create classification map visualization.

# Arguments
- `predictions`: 2D array of predicted class labels
- `geo_transform`: Geographic transformation parameters
- `crop_names`: Dictionary of crop class names
- `save_path`: Path to save figure
"""
function plot_classification_map(predictions::Matrix{Int};
                                crop_names::Dict=CROP_NAMES,
                                save_path::String="")

    rows, cols = size(predictions)
    unique_classes = sort(unique(predictions[predictions .> 0]))

    # Create color matrix
    colors = zeros(RGB{Float64}, rows, cols)

    color_map = Dict(
        1 => RGB(0.0, 0.0, 1.0),      # Cotton - Blue
        2 => RGB(0.0, 0.8, 0.0),      # Corn - Green
        3 => RGB(1.0, 0.0, 0.0),      # Pepper - Red
        4 => RGB(0.5, 0.0, 0.5),      # Jujube - Purple
        5 => RGB(1.0, 0.5, 0.0),      # Pear - Orange
        6 => RGB(0.6, 0.3, 0.0),      # Apricot - Brown
        7 => RGB(1.0, 0.7, 0.7),      # Tomato - Pink
        8 => RGB(0.5, 0.5, 0.5),      # Others - Gray
        0 => RGB(1.0, 1.0, 1.0)       # Background - White
    )

    for i in 1:rows
        for j in 1:cols
            class = predictions[i, j]
            colors[i, j] = get(color_map, class, RGB(0.0, 0.0, 0.0))
        end
    end

    p = plot(colors,
            title = "Farmland Classification Map",
            axis = false,
            aspect_ratio = :equal)

    if !isempty(save_path)
        savefig(p, save_path)
        @info "Classification map saved to $save_path"
    end

    return p
end

# ============================================================================
# Generate All Paper Figures
# ============================================================================

"""
    generate_paper_figures(results, output_dir::String)

Generate all figures as presented in the paper.

# Arguments
- `results`: All evaluation and training results
- `output_dir`: Directory to save figures
"""
function generate_paper_figures(results, output_dir::String)
    mkpath(output_dir)

    @info "Generating paper figures in $output_dir"

    # Figure 1: Study area (requires GIS data - placeholder)
    @info "Figure 1: Study area visualization (requires external GIS data)"

    # Figure 2: NDVI temporal profiles
    if haskey(results, :ndvi_data) && haskey(results, :labels)
        fig2 = plot_ndvi_profiles(results.ndvi_data, results.labels)
        savefig(fig2, joinpath(output_dir, "fig2_ndvi_profiles.png"))
        @info "Figure 2: NDVI profiles saved"
    end

    # Figure 3: Model architecture (requires manual diagram)
    @info "Figure 3: Model architecture (create using diagram tool)"

    # Figure 4: Training curves
    if haskey(results, :logger)
        fig4 = plot_training_curves(results.logger)
        savefig(fig4, joinpath(output_dir, "fig4_training_curves.png"))
        @info "Figure 4: Training curves saved"
    end

    # Figure 5: Confusion matrix (handled in confusion_matrix.jl)
    @info "Figure 5: Confusion matrix (see plot_confusion_matrix function)"

    # Figure 6: Self-training progress
    if haskey(results, :self_training_history)
        fig6 = plot_self_training_progress(results.self_training_history)
        savefig(fig6, joinpath(output_dir, "fig6_self_training.png"))
        @info "Figure 6: Self-training progress saved"
    end

    # Figure 7: Model comparison
    if haskey(results, :model_comparison)
        fig7 = plot_model_comparison(results.model_comparison)
        savefig(fig7, joinpath(output_dir, "fig7_model_comparison.png"))
        @info "Figure 7: Model comparison saved"
    end

    # Figure 8: Classification map
    if haskey(results, :classification_map)
        fig8 = plot_classification_map(results.classification_map)
        savefig(fig8, joinpath(output_dir, "fig8_classification_map.png"))
        @info "Figure 8: Classification map saved"
    end

    @info "All paper figures generated successfully"
end
