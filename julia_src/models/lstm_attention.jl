"""
    LSTM with Attention Model Implementation

This module implements an LSTM model with attention mechanism for
time series classification. The attention mechanism allows the model
to focus on the most relevant time steps for classification.

Architecture:
1. LSTM Encoder: Bidirectional LSTM for sequence encoding
2. Attention: Self-attention over LSTM outputs
3. Classification Head: Attended features → class predictions
"""

using Flux
using Flux: @functor, Dense, LSTM, Dropout, Chain
using Statistics

# ============================================================================
# Attention Layer
# ============================================================================

"""
    LSTMAttention

Attention mechanism for LSTM outputs.

Computes attention weights over sequence and creates weighted representation.

# Fields
- `query::Dense`: Query projection
- `key::Dense`: Key projection
- `value::Dense`: Value projection
- `output::Dense`: Output projection
- `d_model::Int`: Model dimension
"""
struct LSTMAttention
    query::Dense
    key::Dense
    value::Dense
    output::Dense
    d_model::Int
end

@functor LSTMAttention

"""
    LSTMAttention(d_model::Int)

Create attention layer.
"""
function LSTMAttention(d_model::Int)
    query = Dense(d_model, d_model)
    key = Dense(d_model, d_model)
    value = Dense(d_model, d_model)
    output = Dense(d_model, d_model)
    return LSTMAttention(query, key, value, output, d_model)
end

"""
    (attn::LSTMAttention)(x; mask=nothing)

Apply attention to LSTM outputs.

# Arguments
- `x`: LSTM outputs (batch_size × seq_len × d_model)
- `mask`: Optional attention mask

# Returns
- Attended output (batch_size × d_model)
"""
function (attn::LSTMAttention)(x; mask=nothing)
    B, T, D = size(x)

    # Project to Q, K, V
    x_flat = reshape(x, B * T, D)

    Q = attn.query(x_flat')  # (D, B*T)
    K = attn.key(x_flat')
    V = attn.value(x_flat')

    # Reshape: (D, B*T) -> (B, T, D)
    Q = permutedims(reshape(Q, D, B, T), (2, 3, 1))
    K = permutedims(reshape(K, D, B, T), (2, 3, 1))
    V = permutedims(reshape(V, D, B, T), (2, 3, 1))

    # Compute attention scores: (B, T, T)
    scores = batched_mul(Q, batched_transpose(K)) ./ sqrt(Float32(D))

    # Apply mask if provided
    if mask !== nothing
        mask_expanded = reshape(mask, B, 1, T)
        scores = ifelse.(mask_expanded, scores, Float32(-1e9))
    end

    # Softmax
    attention_weights = softmax(scores, dims=3)

    # Apply attention to values: (B, T, D)
    attended = batched_mul(attention_weights, V)

    # Global pooling - weighted average
    if mask !== nothing
        mask_expanded = reshape(Float32.(mask), B, T, 1)
        attended = attended .* mask_expanded
        pooled = sum(attended, dims=2) ./ max.(sum(mask_expanded, dims=2), 1f0)
    else
        pooled = mean(attended, dims=2)
    end

    pooled = dropdims(pooled, dims=2)  # (B, D)

    # Output projection
    output = attn.output(pooled')  # (D, B)
    return permutedims(output)  # (B, D)
end

# ============================================================================
# Bidirectional LSTM
# ============================================================================

"""
    BiLSTM

Bidirectional LSTM layer.

# Fields
- `forward_lstm::LSTM`: Forward LSTM
- `backward_lstm::LSTM`: Backward LSTM
"""
struct BiLSTM
    forward_lstm::LSTM
    backward_lstm::LSTM
    hidden_size::Int
end

@functor BiLSTM

"""
    BiLSTM(input_size::Int, hidden_size::Int)

Create bidirectional LSTM.
"""
function BiLSTM(input_size::Int, hidden_size::Int)
    forward_lstm = LSTM(input_size, hidden_size)
    backward_lstm = LSTM(input_size, hidden_size)
    return BiLSTM(forward_lstm, backward_lstm, hidden_size)
end

"""
    (bilstm::BiLSTM)(x)

Apply bidirectional LSTM.

# Arguments
- `x`: Input (batch_size × seq_len × input_size)

# Returns
- Concatenated forward and backward outputs (batch_size × seq_len × 2*hidden_size)
"""
function (bilstm::BiLSTM)(x)
    B, T, C = size(x)

    # Reset LSTM states
    Flux.reset!(bilstm.forward_lstm)
    Flux.reset!(bilstm.backward_lstm)

    # Forward pass
    forward_outputs = []
    for t in 1:T
        xt = x[:, t, :]'  # (C, B)
        out = bilstm.forward_lstm(xt)  # (hidden_size, B)
        push!(forward_outputs, out)
    end

    # Backward pass
    backward_outputs = []
    for t in T:-1:1
        xt = x[:, t, :]'  # (C, B)
        out = bilstm.backward_lstm(xt)  # (hidden_size, B)
        pushfirst!(backward_outputs, out)
    end

    # Concatenate outputs
    result = zeros(Float32, B, T, 2 * bilstm.hidden_size)
    for t in 1:T
        fwd = permutedims(forward_outputs[t])   # (B, hidden_size)
        bwd = permutedims(backward_outputs[t])  # (B, hidden_size)
        result[:, t, :] = cat(fwd, bwd, dims=2)
    end

    return result
end

# ============================================================================
# LSTM Classifier with Attention
# ============================================================================

"""
    LSTMAttentionClassifier

LSTM with attention mechanism for time series classification.

# Fields
- `embedding::Dense`: Input embedding
- `lstm::BiLSTM`: Bidirectional LSTM
- `attention::LSTMAttention`: Attention layer
- `projection::Dense`: Classification head
- `dropout::Dropout`: Dropout layer
"""
struct LSTMAttentionClassifier
    embedding::Dense
    lstm::BiLSTM
    attention::LSTMAttention
    projection::Dense
    dropout::Dropout
    seq_len::Int
    d_model::Int
    num_class::Int
end

@functor LSTMAttentionClassifier

"""
    LSTMAttentionClassifier(; kwargs...)

Create LSTM classifier with attention.

# Keyword Arguments
- `seq_len::Int=24`: Sequence length
- `enc_in::Int=1`: Number of input features
- `d_model::Int=128`: Model dimension
- `e_layers::Int=2`: Number of LSTM layers
- `dropout::Float64=0.1`: Dropout probability
- `num_class::Int=8`: Number of output classes
"""
function LSTMAttentionClassifier(; seq_len::Int=24, enc_in::Int=1, d_model::Int=128,
                                   e_layers::Int=2, dropout::Float64=0.1, num_class::Int=8)
    # Input embedding
    embedding = Dense(enc_in, d_model)

    # Bidirectional LSTM
    lstm = BiLSTM(d_model, d_model ÷ 2)

    # Attention layer
    attention = LSTMAttention(d_model)

    # Classification head
    projection = Dense(d_model, num_class)

    dropout_layer = Dropout(dropout)

    return LSTMAttentionClassifier(
        embedding, lstm, attention, projection, dropout_layer,
        seq_len, d_model, num_class
    )
end

"""
    LSTMAttentionClassifier(config::Config)

Create LSTM classifier from configuration.
"""
function LSTMAttentionClassifier(config::Config)
    return LSTMAttentionClassifier(
        seq_len=config.seq_len,
        enc_in=config.enc_in,
        d_model=config.d_model,
        e_layers=config.e_layers,
        dropout=config.dropout,
        num_class=config.num_class
    )
end

"""
    (model::LSTMAttentionClassifier)(x, mask=nothing)

Forward pass of LSTM classifier.

# Arguments
- `x`: Input tensor (batch_size × seq_len × enc_in)
- `mask`: Optional padding mask

# Returns
- Logits tensor (batch_size × num_class)
"""
function (model::LSTMAttentionClassifier)(x, mask=nothing)
    B, T, C = size(x)

    # 1. Embed input
    x_flat = reshape(x, B * T, C)
    embedded = model.embedding(x_flat')  # (d_model, B*T)
    embedded = permutedims(reshape(embedded, model.d_model, B, T), (2, 3, 1))  # (B, T, d_model)

    # 2. LSTM encoding
    lstm_out = model.lstm(embedded)  # (B, T, d_model)

    # 3. Attention
    attended = model.attention(lstm_out; mask=mask)  # (B, d_model)

    # 4. Dropout
    output = model.dropout(gelu.(attended))

    # 5. Classification
    logits = model.projection(output')  # (num_class, B)

    return permutedims(logits)  # (B, num_class)
end

# ============================================================================
# Simple LSTM Classifier (without Attention)
# ============================================================================

"""
    SimpleLSTMClassifier

Simple LSTM classifier using last hidden state.

# Fields
- `embedding::Dense`: Input embedding
- `lstm::LSTM`: LSTM layer
- `projection::Dense`: Classification head
"""
struct SimpleLSTMClassifier
    embedding::Dense
    lstm::LSTM
    projection::Dense
    dropout::Dropout
    d_model::Int
    num_class::Int
end

@functor SimpleLSTMClassifier

"""
    SimpleLSTMClassifier(; kwargs...)

Create simple LSTM classifier.
"""
function SimpleLSTMClassifier(; enc_in::Int=1, d_model::Int=128,
                                dropout::Float64=0.1, num_class::Int=8)
    embedding = Dense(enc_in, d_model)
    lstm = LSTM(d_model, d_model)
    projection = Dense(d_model, num_class)
    dropout_layer = Dropout(dropout)

    return SimpleLSTMClassifier(embedding, lstm, projection, dropout_layer,
                                 d_model, num_class)
end

"""
    (model::SimpleLSTMClassifier)(x, mask=nothing)

Forward pass of simple LSTM classifier.
"""
function (model::SimpleLSTMClassifier)(x, mask=nothing)
    B, T, C = size(x)

    # Reset LSTM state
    Flux.reset!(model.lstm)

    # Embed and process sequence
    outputs = []
    for t in 1:T
        xt = x[:, t, :]'  # (C, B)
        embedded = model.embedding(xt)  # (d_model, B)
        out = model.lstm(embedded)  # (d_model, B)
        push!(outputs, out)
    end

    # Use last output
    last_output = outputs[end]  # (d_model, B)
    last_output = permutedims(last_output)  # (B, d_model)

    # Dropout and project
    output = model.dropout(gelu.(last_output))
    logits = model.projection(output')  # (num_class, B)

    return permutedims(logits)  # (B, num_class)
end

# ============================================================================
# Model Utilities
# ============================================================================

"""
    count_parameters(model::LSTMAttentionClassifier)

Count trainable parameters.
"""
function count_parameters(model::LSTMAttentionClassifier)
    total = 0
    for p in Flux.params(model)
        total += length(p)
    end
    return total
end

"""
    model_summary(model::LSTMAttentionClassifier)

Print model summary.
"""
function model_summary(model::LSTMAttentionClassifier)
    println("\n" * "="^50)
    println("LSTM + Attention Classifier Summary")
    println("="^50)
    println("Sequence Length:   $(model.seq_len)")
    println("Model Dimension:   $(model.d_model)")
    println("Number of Classes: $(model.num_class)")
    println("Total Parameters:  $(count_parameters(model))")
    println("="^50 * "\n")
end
