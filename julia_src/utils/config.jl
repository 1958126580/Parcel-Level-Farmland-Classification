"""
    Configuration module for Parcel-Level Farmland Classification

This module provides configuration management for model training and evaluation.
All hyperparameters and settings are centralized here for easy experimentation.
"""

# ============================================================================
# Configuration Structure
# ============================================================================

"""
    Config

Configuration structure holding all hyperparameters and settings for the
farmland classification pipeline.

# Fields
- `task_name::String`: Classification task type ("classification")
- `model_name::String`: Model architecture ("TimesNet", "Transformer", "LSTM")
- `data_path::String`: Path to the dataset directory
- `seq_len::Int`: Input sequence length (number of time steps)
- `pred_len::Int`: Prediction length (0 for classification)
- `enc_in::Int`: Number of input features/channels
- `num_class::Int`: Number of output classes
- `d_model::Int`: Model embedding dimension
- `n_heads::Int`: Number of attention heads
- `e_layers::Int`: Number of encoder layers
- `d_ff::Int`: Feed-forward network dimension
- `dropout::Float64`: Dropout probability
- `top_k::Int`: Number of top periods for TimesNet
- `num_kernels::Int`: Number of inception kernels
- `batch_size::Int`: Training batch size
- `learning_rate::Float64`: Optimizer learning rate
- `train_epochs::Int`: Maximum training epochs
- `patience::Int`: Early stopping patience
- `use_gpu::Bool`: Whether to use GPU acceleration
- `seed::Int`: Random seed for reproducibility
"""
Base.@kwdef mutable struct Config
    # Task configuration
    task_name::String = "classification"
    model_name::String = "TimesNet"
    data_path::String = "./data/WeiganFarmland"
    output_path::String = "./results"

    # Sequence configuration
    seq_len::Int = 24           # Input sequence length (24 time observations)
    pred_len::Int = 0           # Prediction length (0 for classification)
    label_len::Int = 48         # Label length for decoder

    # Feature configuration
    enc_in::Int = 1             # Number of input features per time step
    dec_in::Int = 1             # Decoder input size
    c_out::Int = 1              # Output size

    # Classification configuration
    num_class::Int = 8          # Number of crop classes

    # Model architecture
    d_model::Int = 128          # Embedding dimension
    n_heads::Int = 8            # Number of attention heads
    e_layers::Int = 3           # Number of encoder layers
    d_layers::Int = 1           # Number of decoder layers
    d_ff::Int = 256             # Feed-forward network dimension
    dropout::Float64 = 0.1      # Dropout probability
    activation::String = "gelu" # Activation function

    # TimesNet specific
    top_k::Int = 5              # Number of top periods to consider
    num_kernels::Int = 6        # Number of inception kernels

    # Training configuration
    batch_size::Int = 16        # Batch size
    learning_rate::Float64 = 0.0001  # Learning rate
    train_epochs::Int = 100     # Maximum training epochs
    patience::Int = 10          # Early stopping patience
    max_grad_norm::Float64 = 4.0 # Gradient clipping norm

    # Hardware configuration
    use_gpu::Bool = true        # Use GPU if available
    num_workers::Int = 4        # DataLoader workers
    seed::Int = 2021            # Random seed

    # Self-optimization configuration
    self_training_rounds::Int = 5   # Number of self-training iterations
    confidence_threshold::Float64 = 0.9  # Pseudo-label confidence threshold
    pseudo_label_ratio::Float64 = 0.3    # Ratio of pseudo-labels to add

    # Crop type labels (for visualization)
    class_names::Vector{String} = [
        "Cotton", "Corn", "Pepper", "Jujube",
        "Pear", "Apricot", "Tomato", "Others"
    ]
end

# ============================================================================
# Configuration Factory Functions
# ============================================================================

"""
    default_config()

Create a default configuration with standard hyperparameters.

# Returns
- `Config`: Default configuration object

# Example
```julia
config = default_config()
config.batch_size = 32  # Modify as needed
```
"""
function default_config()
    return Config()
end

"""
    timesnet_config()

Create configuration optimized for TimesNet model.

# Returns
- `Config`: Configuration optimized for TimesNet
"""
function timesnet_config()
    config = Config()
    config.model_name = "TimesNet"
    config.d_model = 32
    config.d_ff = 32
    config.e_layers = 2
    config.top_k = 5
    config.num_kernels = 6
    return config
end

"""
    transformer_config()

Create configuration optimized for Transformer model.

# Returns
- `Config`: Configuration optimized for Transformer
"""
function transformer_config()
    config = Config()
    config.model_name = "Transformer"
    config.d_model = 128
    config.d_ff = 256
    config.e_layers = 3
    config.n_heads = 8
    return config
end

"""
    lstm_config()

Create configuration optimized for LSTM with Attention model.

# Returns
- `Config`: Configuration optimized for LSTM+Attention
"""
function lstm_config()
    config = Config()
    config.model_name = "LSTM"
    config.d_model = 128
    config.d_ff = 256
    config.e_layers = 2
    return config
end

# ============================================================================
# Configuration I/O
# ============================================================================

"""
    save_config(config::Config, filepath::String)

Save configuration to a JSON file.

# Arguments
- `config::Config`: Configuration to save
- `filepath::String`: Output file path
"""
function save_config(config::Config, filepath::String)
    # Convert config to dictionary
    config_dict = Dict{String, Any}()
    for field in fieldnames(Config)
        config_dict[String(field)] = getfield(config, field)
    end

    # Write to JSON file
    open(filepath, "w") do f
        JSON3.pretty(f, config_dict)
    end
    @info "Configuration saved to $filepath"
end

"""
    load_config(filepath::String)

Load configuration from a JSON file.

# Arguments
- `filepath::String`: Input file path

# Returns
- `Config`: Loaded configuration object
"""
function load_config(filepath::String)
    # Read JSON file
    config_dict = JSON3.read(read(filepath, String), Dict{String, Any})

    # Create config with loaded values
    config = Config()
    for (key, value) in config_dict
        field = Symbol(key)
        if hasfield(Config, field)
            setfield!(config, field, value)
        end
    end

    @info "Configuration loaded from $filepath"
    return config
end

"""
    print_config(config::Config)

Print configuration in a formatted manner.

# Arguments
- `config::Config`: Configuration to print
"""
function print_config(config::Config)
    println("\n" * "="^60)
    println("Configuration Settings")
    println("="^60)

    println("\n[Task]")
    println("  Task Name:      $(config.task_name)")
    println("  Model:          $(config.model_name)")
    println("  Data Path:      $(config.data_path)")

    println("\n[Sequence]")
    println("  Sequence Length: $(config.seq_len)")
    println("  Input Features:  $(config.enc_in)")
    println("  Num Classes:     $(config.num_class)")

    println("\n[Model Architecture]")
    println("  d_model:        $(config.d_model)")
    println("  n_heads:        $(config.n_heads)")
    println("  e_layers:       $(config.e_layers)")
    println("  d_ff:           $(config.d_ff)")
    println("  Dropout:        $(config.dropout)")

    println("\n[Training]")
    println("  Batch Size:     $(config.batch_size)")
    println("  Learning Rate:  $(config.learning_rate)")
    println("  Epochs:         $(config.train_epochs)")
    println("  Patience:       $(config.patience)")
    println("  Use GPU:        $(config.use_gpu)")

    println("\n[Self-Optimization]")
    println("  Rounds:         $(config.self_training_rounds)")
    println("  Confidence:     $(config.confidence_threshold)")

    println("="^60 * "\n")
end

# ============================================================================
# Validation Functions
# ============================================================================

"""
    validate_config(config::Config)

Validate configuration parameters.

# Arguments
- `config::Config`: Configuration to validate

# Returns
- `Bool`: True if valid, throws error otherwise
"""
function validate_config(config::Config)
    # Validate sequence length
    @assert config.seq_len > 0 "Sequence length must be positive"

    # Validate model dimensions
    @assert config.d_model > 0 "Model dimension must be positive"
    @assert config.d_model % config.n_heads == 0 "d_model must be divisible by n_heads"

    # Validate training parameters
    @assert config.batch_size > 0 "Batch size must be positive"
    @assert 0.0 < config.learning_rate < 1.0 "Learning rate must be between 0 and 1"
    @assert 0.0 <= config.dropout < 1.0 "Dropout must be between 0 and 1"

    # Validate self-optimization parameters
    @assert 0.0 <= config.confidence_threshold <= 1.0 "Confidence threshold must be between 0 and 1"

    return true
end
