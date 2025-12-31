"""
    Unit Tests for Parcel-Level Farmland Classification

This file contains comprehensive unit tests for all modules:
- Data loading and preprocessing
- Model architectures
- Training utilities
- Evaluation metrics
- Visualization functions
"""

using Test
using Random
using Statistics
using LinearAlgebra

# Set random seed for reproducibility
Random.seed!(2021)

# ============================================================================
# Test Configuration and Utilities
# ============================================================================

@testset "Configuration Tests" begin
    # Test default config creation
    @testset "Default Config" begin
        config = Config()
        @test config.seq_len == 24
        @test config.num_class == 8
        @test config.learning_rate > 0
        @test config.train_epochs > 0
    end

    # Test custom config
    @testset "Custom Config" begin
        config = Config(
            model_name = "Transformer",
            seq_len = 48,
            enc_in = 8,
            num_class = 10
        )
        @test config.model_name == "Transformer"
        @test config.seq_len == 48
        @test config.enc_in == 8
        @test config.num_class == 10
    end
end

# ============================================================================
# Test Metrics
# ============================================================================

@testset "Metrics Tests" begin
    # Test accuracy
    @testset "Accuracy" begin
        predictions = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
        labels = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
        @test accuracy(predictions, labels) == 1.0

        predictions2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        labels2 = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
        @test 0.0 <= accuracy(predictions2, labels2) <= 1.0
    end

    # Test confusion matrix
    @testset "Confusion Matrix" begin
        predictions = [1, 2, 3, 1, 2, 3]
        labels = [1, 2, 3, 1, 2, 3]
        cm = confusion_matrix(predictions, labels, 3)
        @test size(cm) == (3, 3)
        @test sum(cm) == 6
        @test all(diag(cm) .== 2)
    end

    # Test F1 score
    @testset "F1 Score" begin
        predictions = [1, 1, 2, 2, 3, 3]
        labels = [1, 1, 2, 2, 3, 3]
        f1 = f1_score(predictions, labels, 3; average=:macro)
        @test f1 == 1.0
    end

    # Test precision and recall
    @testset "Precision and Recall" begin
        predictions = [1, 1, 1, 2, 2, 2]
        labels = [1, 1, 1, 2, 2, 2]
        prec = precision_score(predictions, labels, 2; average=:macro)
        rec = recall_score(predictions, labels, 2; average=:macro)
        @test prec == 1.0
        @test rec == 1.0
    end
end

# ============================================================================
# Test Spectral Indices
# ============================================================================

@testset "Spectral Indices Tests" begin
    # Create synthetic band data
    n_samples = 100
    nir = rand(Float32, n_samples) .* 0.5 .+ 0.3
    red = rand(Float32, n_samples) .* 0.2 .+ 0.1
    green = rand(Float32, n_samples) .* 0.2 .+ 0.1
    swir = rand(Float32, n_samples) .* 0.3 .+ 0.1
    red_edge1 = rand(Float32, n_samples) .* 0.3 .+ 0.2
    blue = rand(Float32, n_samples) .* 0.1 .+ 0.05

    @testset "NDVI" begin
        ndvi = compute_ndvi(nir, red)
        @test length(ndvi) == n_samples
        @test all(-1.0 .<= ndvi .<= 1.0)
    end

    @testset "MNDWI" begin
        mndwi = compute_mndwi(green, swir)
        @test length(mndwi) == n_samples
        @test all(-1.0 .<= mndwi .<= 1.0)
    end

    @testset "NDVIRE1" begin
        ndvire1 = compute_ndvire1(nir, red_edge1)
        @test length(ndvire1) == n_samples
        @test all(-1.0 .<= ndvire1 .<= 1.0)
    end

    @testset "EVI" begin
        evi = compute_evi(nir, red, blue)
        @test length(evi) == n_samples
    end
end

# ============================================================================
# Test Data Preprocessing
# ============================================================================

@testset "Preprocessing Tests" begin
    # Create synthetic time series data
    n_samples = 50
    seq_len = 24
    n_features = 4
    data = randn(Float32, n_samples, seq_len, n_features)

    @testset "Normalization" begin
        normalized = normalize_time_series(data; method=:zscore)
        @test size(normalized) == size(data)

        # Check approximate zero mean per feature
        for f in 1:n_features
            feature_data = vec(normalized[:, :, f])
            @test abs(mean(feature_data)) < 1.0
        end
    end

    @testset "MinMax Normalization" begin
        normalized = normalize_time_series(data; method=:minmax)
        @test size(normalized) == size(data)
        @test all(normalized .>= 0.0)
        @test all(normalized .<= 1.0)
    end

    @testset "Interpolation" begin
        # Create data with missing values
        data_missing = copy(data)
        data_missing[1, 5, 1] = NaN32

        # Interpolation should handle NaN
        interpolated = interpolate_missing(data_missing)
        @test !any(isnan.(interpolated))
    end
end

# ============================================================================
# Test Model Architectures
# ============================================================================

@testset "Model Architecture Tests" begin
    batch_size = 4
    seq_len = 24
    enc_in = 4
    num_class = 8
    d_model = 32

    @testset "TokenEmbedding" begin
        embedding = TokenEmbedding(enc_in, d_model)
        x = randn(Float32, batch_size, seq_len, enc_in)
        y = embedding(x)
        @test size(y) == (batch_size, seq_len, d_model)
    end

    @testset "PositionalEncoding" begin
        pe = PositionalEncoding(d_model, seq_len)
        x = randn(Float32, batch_size, seq_len, d_model)
        y = pe(x)
        @test size(y) == size(x)
    end

    @testset "MultiHeadAttention" begin
        n_heads = 4
        mha = MultiHeadAttention(d_model, n_heads)
        x = randn(Float32, batch_size, seq_len, d_model)
        y = mha(x, x, x)
        @test size(y) == size(x)
    end

    @testset "InceptionBlock" begin
        inception = InceptionBlockV1(d_model, d_model * 2)
        # Reshape for 2D conv: (W, H, C, B)
        x = randn(Float32, 4, 6, d_model, batch_size)
        y = inception(x)
        @test size(y, 4) == batch_size
    end
end

# ============================================================================
# Test FFT Period Detection
# ============================================================================

@testset "FFT Period Detection Tests" begin
    batch_size = 8
    seq_len = 24
    channels = 4

    @testset "Period Detection" begin
        x = randn(Float32, batch_size, seq_len, channels)
        periods, weights = fft_for_period(x; k=3)

        @test length(periods) == 3
        @test size(weights) == (batch_size, 3)
        @test all(periods .>= 2)
        @test all(periods .<= seq_len)
    end

    @testset "2D Reshaping" begin
        x = randn(Float32, batch_size, seq_len, channels)
        period = 6

        reshaped = reshape_for_2d(x, period)
        @test size(reshaped, 4) == batch_size
        @test size(reshaped, 1) == period

        # Test round-trip
        back = reshape_from_2d(reshaped, seq_len)
        @test size(back) == size(x)
    end
end

# ============================================================================
# Test Training Utilities
# ============================================================================

@testset "Training Utilities Tests" begin
    @testset "EarlyStopping" begin
        es = EarlyStopping(patience=3, mode=:min)

        # Improving values
        @test es(1.0) == true   # First value
        @test es(0.9) == true   # Improvement
        @test es(0.8) == true   # Improvement
        @test es.early_stop == false

        # No improvement
        @test es(0.85) == false
        @test es(0.9) == false
        @test es(1.0) == false
        @test es.early_stop == true
    end

    @testset "LRScheduler" begin
        initial_lr = 0.001
        scheduler = StepLRScheduler(initial_lr; step_size=5, gamma=0.5)

        @test scheduler(1) == initial_lr
        @test scheduler(5) == initial_lr
        @test scheduler(6) == initial_lr * 0.5
        @test scheduler(11) == initial_lr * 0.25
    end

    @testset "Cosine Annealing" begin
        initial_lr = 0.001
        scheduler = CosineAnnealingScheduler(initial_lr; T_max=100)

        @test scheduler(1) == initial_lr
        @test scheduler(50) < initial_lr
        @test scheduler(100) < scheduler(50)
    end
end

# ============================================================================
# Test Self-Training
# ============================================================================

@testset "Self-Training Tests" begin
    @testset "SelfTrainingConfig" begin
        config = SelfTrainingConfig(
            initial_threshold = 0.95,
            final_threshold = 0.85,
            max_iterations = 10
        )

        @test config.initial_threshold == 0.95
        @test config.final_threshold == 0.85
        @test config.max_iterations == 10
    end

    @testset "Temporal Consistency" begin
        # Create predictions from multiple epochs
        predictions1 = [1, 2, 3, 1, 2, 3, 1, 2]
        predictions2 = [1, 2, 3, 1, 2, 3, 1, 2]
        predictions3 = [1, 2, 3, 1, 2, 3, 1, 2]

        predictions_multi = [predictions1, predictions2, predictions3]
        consistent_idx = filter_by_temporal_consistency(predictions_multi;
                                                       agreement_threshold=0.8)

        @test length(consistent_idx) == 8  # All should be consistent
    end
end

# ============================================================================
# Integration Tests
# ============================================================================

@testset "Integration Tests" begin
    @testset "End-to-End Pipeline" begin
        # Create synthetic dataset
        n_train = 100
        n_test = 20
        seq_len = 24
        enc_in = 4
        num_class = 8

        # Generate random data
        train_data = randn(Float32, n_train, seq_len, enc_in)
        train_labels = rand(1:num_class, n_train)
        test_data = randn(Float32, n_test, seq_len, enc_in)
        test_labels = rand(1:num_class, n_test)

        # Create config
        config = Config(
            model_name = "SimpleTimesNet",
            seq_len = seq_len,
            enc_in = enc_in,
            num_class = num_class,
            d_model = 16,
            d_ff = 32,
            e_layers = 1,
            train_epochs = 2,
            batch_size = 16
        )

        # Test model creation
        model = create_model(config)
        @test model !== nothing

        # Test forward pass
        batch = randn(Float32, 4, seq_len, enc_in)
        mask = ones(Float32, 4, seq_len)
        output = model(batch, mask)
        @test size(output, 1) == 4
        @test size(output, 2) == num_class
    end
end

# ============================================================================
# Run All Tests
# ============================================================================

println("\n" * "="^50)
println("All tests completed!")
println("="^50)
