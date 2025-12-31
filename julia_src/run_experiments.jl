#!/usr/bin/env julia
"""
    Main Experiment Runner for Parcel-Level Farmland Classification

This script reproduces all experiments and figures from the paper:
"Parcel-Level Farmland Classification via Deep Heterogeneous Feature Integration
and Iterative Self-Optimization"

Usage:
    julia run_experiments.jl [--data-path PATH] [--output-dir DIR] [--gpu]

Arguments:
    --data-path PATH    Path to the data directory (default: ./data)
    --output-dir DIR    Directory for output results (default: ./results)
    --gpu               Enable GPU acceleration if available
    --quick             Run quick experiments for testing
    --full              Run full experiments as in paper

Example:
    julia run_experiments.jl --data-path ./data --output-dir ./results --gpu

Author: Julia Implementation
Date: December 2025
"""

# ============================================================================
# Imports and Setup
# ============================================================================

using Pkg
# Activate the project environment
Pkg.activate(@__DIR__)

using ArgParse
using Dates
using Printf
using Random
using Statistics

# Include the main module
include("ParcelFarmlandClassification.jl")
using .ParcelFarmlandClassification

# ============================================================================
# Command Line Arguments
# ============================================================================

function parse_commandline()
    s = ArgParseSettings(
        description = "Parcel-Level Farmland Classification Experiments",
        commands_are_required = false
    )

    @add_arg_table! s begin
        "--data-path"
            help = "Path to data directory"
            arg_type = String
            default = "data"
        "--output-dir"
            help = "Output directory for results"
            arg_type = String
            default = "results"
        "--gpu"
            help = "Enable GPU acceleration"
            action = :store_true
        "--quick"
            help = "Run quick experiments for testing"
            action = :store_true
        "--full"
            help = "Run full experiments as in paper"
            action = :store_true
        "--seed"
            help = "Random seed for reproducibility"
            arg_type = Int
            default = 2021
        "--model"
            help = "Model to train (timesnet, transformer, lstm, all)"
            arg_type = String
            default = "all"
    end

    return parse_args(s)
end

# ============================================================================
# Experiment Configurations
# ============================================================================

"""
Create configuration for quick testing
"""
function quick_config()
    return Config(
        model_name = "TimesNet",
        seq_len = 24,
        enc_in = 4,  # NDVI, MNDWI, NDVIRE1, EVI
        num_class = 8,
        d_model = 32,
        d_ff = 64,
        n_heads = 4,
        e_layers = 1,
        top_k = 3,
        dropout = 0.1,
        train_epochs = 10,
        batch_size = 32,
        learning_rate = 0.001,
        patience = 5
    )
end

"""
Create configuration for full experiments (as in paper)
"""
function full_config()
    return Config(
        model_name = "TimesNet",
        seq_len = 24,
        enc_in = 4,
        num_class = 8,
        d_model = 64,
        d_ff = 256,
        n_heads = 8,
        e_layers = 2,
        top_k = 5,
        dropout = 0.1,
        train_epochs = 100,
        batch_size = 16,
        learning_rate = 0.0001,
        patience = 15
    )
end

# ============================================================================
# Main Experiment Functions
# ============================================================================

"""
    run_data_exploration(data_path::String, output_dir::String)

Explore and visualize the dataset.
"""
function run_data_exploration(data_path::String, output_dir::String)
    println("\n" * "="^60)
    println("DATA EXPLORATION")
    println("="^60)

    # Create output directory
    figures_dir = joinpath(output_dir, "figures", "exploration")
    mkpath(figures_dir)

    # Load data
    @info "Loading dataset from $data_path"
    train_data, test_data = load_farmland_data(data_path)

    @info "Dataset Statistics:"
    @printf("  Training samples: %d\n", length(train_data))
    @printf("  Test samples: %d\n", length(test_data))
    @printf("  Sequence length: %d\n", train_data.seq_len)
    @printf("  Feature dimension: %d\n", train_data.feature_dim)
    @printf("  Number of classes: %d\n", train_data.num_classes)

    # Class distribution
    println("\nClass Distribution (Training):")
    for c in 1:train_data.num_classes
        count = sum(train_data.labels .== c)
        @printf("  Class %d (%s): %d samples (%.1f%%)\n",
                c, get(CROP_NAMES, c, "Unknown"),
                count, 100 * count / length(train_data))
    end

    # Visualize NDVI profiles
    @info "Generating NDVI profile visualization..."
    fig_ndvi = plot_ndvi_profiles(train_data.data, train_data.labels)
    savefig(fig_ndvi, joinpath(figures_dir, "ndvi_profiles.png"))

    # Visualize spectral indices
    @info "Generating spectral indices visualization..."
    fig_spectral = plot_spectral_indices(train_data.data, train_data.labels)
    savefig(fig_spectral, joinpath(figures_dir, "spectral_indices.png"))

    # Time series examples
    @info "Generating time series examples..."
    fig_ts = plot_time_series(train_data.data, train_data.labels;
                             title="Time Series Examples by Crop Type")
    savefig(fig_ts, joinpath(figures_dir, "time_series_examples.png"))

    @info "Data exploration complete. Figures saved to $figures_dir"

    return train_data, test_data
end

"""
    run_model_training(train_data, test_data, config::Config, output_dir::String;
                      device=cpu, model_name::String="TimesNet")

Train a single model and save results.
"""
function run_model_training(train_data, test_data, config::Config, output_dir::String;
                           device=cpu, model_name::String="TimesNet")

    println("\n" * "="^60)
    println("TRAINING: $model_name")
    println("="^60)

    # Create output directories
    models_dir = joinpath(output_dir, "models")
    figures_dir = joinpath(output_dir, "figures", lowercase(model_name))
    mkpath(models_dir)
    mkpath(figures_dir)

    # Update config with model name
    config = Config(config; model_name=model_name)

    # Create data loaders
    train_loader, test_loader = create_dataloaders(train_data, test_data;
                                                   batch_size=config.batch_size)

    # Create model
    @info "Creating $model_name model..."
    model = create_model(config)
    @info "Model parameters: $(sum(length, Flux.params(model)))"

    # Train model
    @info "Starting training..."
    train_result = train_model!(model, train_loader, test_loader, config;
                               device=device)

    # Plot training curves
    fig_curves = plot_training_curves(train_result.logger)
    savefig(fig_curves, joinpath(figures_dir, "training_curves.png"))

    # Evaluate model
    @info "Evaluating model..."
    eval_result = evaluate_model(train_result.model, test_loader;
                                device=device, num_classes=config.num_class)

    # Print evaluation report
    print_evaluation_report(eval_result)

    # Plot confusion matrix
    fig_cm = plot_confusion_matrix(eval_result.confusion_matrix;
                                  normalize=true,
                                  title="$model_name Confusion Matrix")
    savefig(fig_cm, joinpath(figures_dir, "confusion_matrix.png"))

    # Plot per-class performance
    fig_class = plot_class_performance(eval_result.confusion_matrix)
    savefig(fig_class, joinpath(figures_dir, "class_performance.png"))

    # Export results
    export_results(eval_result, joinpath(output_dir, "results", lowercase(model_name)))

    # Save model checkpoint
    save_checkpoint(train_result.model, eval_result,
                   joinpath(models_dir, "$(lowercase(model_name))_best.bson"))

    @info "Training complete for $model_name"

    return (model = train_result.model, results = eval_result, logger = train_result.logger)
end

"""
    run_self_training(train_data, unlabeled_data, test_data, config::Config,
                     output_dir::String; device=cpu)

Run iterative self-training experiment.
"""
function run_self_training(train_data, unlabeled_data, test_data, config::Config,
                          output_dir::String; device=cpu)

    println("\n" * "="^60)
    println("ITERATIVE SELF-TRAINING")
    println("="^60)

    figures_dir = joinpath(output_dir, "figures", "self_training")
    mkpath(figures_dir)

    # Create data loaders
    train_loader, test_loader = create_dataloaders(train_data, test_data;
                                                   batch_size=config.batch_size)
    unlabeled_loader = create_unlabeled_dataloader(unlabeled_data;
                                                  batch_size=config.batch_size)

    # Self-training configuration
    st_config = SelfTrainingConfig(
        initial_threshold = 0.95,
        final_threshold = 0.85,
        max_iterations = 10,
        min_new_samples = 50
    )

    # Model creation function
    model_fn() = create_model(config)

    # Run self-training
    @info "Starting iterative self-training..."
    st_result = iterative_self_training(model_fn, train_loader, unlabeled_loader,
                                       test_loader, config, st_config; device=device)

    # Plot self-training progress
    fig_st = plot_self_training_progress(st_result.history)
    savefig(fig_st, joinpath(figures_dir, "self_training_progress.png"))

    # Final evaluation
    eval_result = evaluate_model(st_result.model, test_loader;
                                device=device, num_classes=config.num_class)
    print_evaluation_report(eval_result)

    @info "Self-training complete"
    @printf("Best accuracy: %.4f\n", st_result.best_accuracy)
    @printf("Total pseudo-labels: %d\n", st_result.total_pseudo_labels)

    return st_result
end

"""
    run_model_comparison(train_data, test_data, config::Config, output_dir::String;
                        device=cpu)

Compare all model architectures.
"""
function run_model_comparison(train_data, test_data, config::Config, output_dir::String;
                             device=cpu)

    println("\n" * "="^60)
    println("MODEL COMPARISON")
    println("="^60)

    figures_dir = joinpath(output_dir, "figures", "comparison")
    mkpath(figures_dir)

    models = ["TimesNet", "Transformer", "LSTMAttention"]
    all_results = Dict{String, NamedTuple}()
    all_cms = Matrix{Int}[]

    for model_name in models
        result = run_model_training(train_data, test_data, config, output_dir;
                                   device=device, model_name=model_name)
        all_results[model_name] = result.results
        push!(all_cms, result.results.confusion_matrix)
    end

    # Compare models
    compare_models(all_results)

    # Plot comparison
    fig_comp = plot_model_comparison(all_results)
    savefig(fig_comp, joinpath(figures_dir, "model_comparison.png"))

    # Plot confusion matrix comparison
    fig_cm_comp = plot_confusion_matrix_comparison(all_cms, models)
    savefig(fig_cm_comp, joinpath(figures_dir, "confusion_matrix_comparison.png"))

    return all_results
end

"""
    run_benchmark(train_data, test_data, config::Config, output_dir::String;
                 device=cpu, n_runs::Int=5)

Run full benchmark with multiple runs.
"""
function run_benchmark(train_data, test_data, config::Config, output_dir::String;
                      device=cpu, n_runs::Int=5)

    println("\n" * "="^60)
    println("BENCHMARK ($n_runs runs per model)")
    println("="^60)

    figures_dir = joinpath(output_dir, "figures", "benchmark")
    mkpath(figures_dir)

    # Create data loaders
    train_loader, test_loader = create_dataloaders(train_data, test_data;
                                                   batch_size=config.batch_size)

    # Define model configurations
    model_configs = [
        Config(config; model_name="TimesNet"),
        Config(config; model_name="Transformer"),
        Config(config; model_name="LSTMAttention")
    ]

    # Run benchmark
    benchmark_results = run_benchmark(model_configs, train_loader, test_loader;
                                     device=device, n_runs=n_runs)

    # Plot benchmark results
    fig_bench = plot_benchmark_results(benchmark_results)
    savefig(fig_bench, joinpath(figures_dir, "benchmark_results.png"))

    return benchmark_results
end

# ============================================================================
# Main Entry Point
# ============================================================================

function main()
    println("\n" * "="^70)
    println("  PARCEL-LEVEL FARMLAND CLASSIFICATION")
    println("  Julia Implementation of Deep Heterogeneous Feature Integration")
    println("="^70)
    println("  Start time: $(Dates.now())")
    println("="^70 * "\n")

    # Parse command line arguments
    args = parse_commandline()

    # Set random seed
    Random.seed!(args["seed"])
    @info "Random seed: $(args["seed"])"

    # Setup device
    device = cpu
    if args["gpu"]
        if CUDA.functional()
            device = gpu
            @info "Using GPU acceleration"
        else
            @warn "GPU requested but CUDA not available. Using CPU."
        end
    end

    # Setup paths
    data_path = args["data-path"]
    output_dir = args["output-dir"]
    mkpath(output_dir)

    # Select configuration
    config = args["quick"] ? quick_config() : full_config()
    @info "Configuration: $(args["quick"] ? "Quick" : "Full")"

    # Run experiments
    try
        # Step 1: Data Exploration
        train_data, test_data = run_data_exploration(data_path, output_dir)

        # Step 2: Model Training
        if args["model"] == "all" || args["model"] == "timesnet"
            run_model_training(train_data, test_data, config, output_dir;
                              device=device, model_name="TimesNet")
        end

        if args["model"] == "all" || args["model"] == "transformer"
            run_model_training(train_data, test_data, config, output_dir;
                              device=device, model_name="Transformer")
        end

        if args["model"] == "all" || args["model"] == "lstm"
            run_model_training(train_data, test_data, config, output_dir;
                              device=device, model_name="LSTMAttention")
        end

        # Step 3: Model Comparison (if running all models)
        if args["model"] == "all"
            run_model_comparison(train_data, test_data, config, output_dir;
                               device=device)
        end

        # Step 4: Benchmark (if full experiments)
        if args["full"]
            run_benchmark(train_data, test_data, config, output_dir;
                         device=device, n_runs=5)
        end

        println("\n" * "="^70)
        println("  EXPERIMENTS COMPLETED SUCCESSFULLY")
        println("  End time: $(Dates.now())")
        println("  Results saved to: $output_dir")
        println("="^70 * "\n")

    catch e
        @error "Experiment failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
