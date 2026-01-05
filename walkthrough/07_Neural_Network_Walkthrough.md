# Neural Network - Walkthrough

## Overview
Okay, this one's complex. We're talking multi-task learning (classification AND regression), batch norm, dropout, multiple optimizers, Bayesian hyperparameter tuning - basically everything. This is production-grade stuff that took a while to get right. Buckle up!

## Architecture Config

### Network Configuration (`config` dict)

#### 1. **hidden_units**
- **Type**: List of integers
- **Example**: `[256, 128, 64]`
- **Purpose**: Defines network architecture (layer sizes)
- **Effect**:
  - More layers: Deeper network, more capacity
  - Wider layers: More parameters per layer
  - Deep & narrow: Better for hierarchical features
  - Shallow & wide: Better for simple patterns
- **Typical**: 2-4 hidden layers, 64-512 units per layer

#### 2. **initializer**
- **Type**: String
- **Options**: `'he_normal'`, `'he_uniform'`, `'xavier_normal'`, `'xavier_uniform'`
- **Purpose**: How to initialize weights
- **He initialization**: For ReLU activations
  - `he_normal`: Gaussian with σ = √(2/fan_in)
  - `he_uniform`: Uniform [-√(6/fan_in), √(6/fan_in)]
- **Xavier/Glorot**: For sigmoid/tanh
  - `xavier_normal`: σ = √(2/(fan_in + fan_out))
  - `xavier_uniform`: [-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))]
- **Critical**: Prevents vanishing/exploding gradients

#### 3. **batch_norm**
- **Type**: Boolean
- **Default**: False
- **Purpose**: Normalize layer inputs during training
- **Benefits**:
  - Faster convergence
  - Higher learning rates possible
  - Reduces internal covariate shift
  - Slight regularization effect
- **Cost**: Additional parameters (γ, β per layer)
- **When to use**: Deep networks (>3 layers), slow convergence

#### 4. **dropout_rate**
- **Type**: Float
- **Range**: 0.0 to 0.8
- **Default**: 0.0 (no dropout)
- **Purpose**: Randomly drop neurons during training
- **Effect**:
  - 0.0: No regularization
  - 0.2-0.5: Moderate regularization
  - >0.6: Heavy regularization, may underfit
- **Key insight**: Only active during training!

#### 5. **l1, l2** (Regularization)
- **Type**: Float
- **Range**: 0.0 to 0.1
- **Purpose**: Penalize large weights
- **L1** (Lasso): Encourages sparsity
- **L2** (Ridge): Weight shrinkage
- **Formula**: `loss += l1*|W| + l2*W²`

#### 6. **binary_cls**
- **Type**: Boolean
- **Purpose**: Binary classification (1 output) vs. multi-class (K outputs)
- **Effect on architecture**: Changes output layer size and activation

---

## Optimizer Parameters

### HybridOptimizer

**Available Optimizers**: `'sgd'`, `'momentum'`, `'adagrad'`, `'rmsprop'`, `'adam'`

#### 1. **SGD** (Stochastic Gradient Descent)
```python
weights -= learning_rate * gradients
```
- Simplest
- No memory of past gradients
- Can be unstable

#### 2. **Momentum**
```python
velocity = momentum * velocity - learning_rate * gradients
weights += velocity
```
- Accelerates in consistent directions
- Dampens oscillations
- Typical momentum: 0.9

#### 3. **AdaGrad**
```python
cache += gradients ** 2
weights -= learning_rate * gradients / (sqrt(cache) + epsilon)
```
- Adapts learning rate per parameter
- Problem: Learning rate decays too aggressively

#### 4. **RMSprop**
```python
cache = beta * cache + (1-beta) * gradients ** 2
weights -= learning_rate * gradients / (sqrt(cache) + epsilon)
```
- Fixes AdaGrad's aggressive decay
- Uses moving average
- Good for RNNs

#### 5. **Adam** (Adaptive Moment Estimation)
```python
m = beta1 * m + (1-beta1) * gradients  # First moment (mean)
v = beta2 * v + (1-beta2) * gradients**2  # Second moment (variance)
m_hat = m / (1 - beta1**t)  # Bias correction
v_hat = v / (1 - beta2**t)
weights -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```
- Combines momentum + RMSprop
- Adapts learning rate per parameter
- Most popular for deep learning
- Defaults: beta1=0.9, beta2=0.999, epsilon=1e-8

---

## Layer Types

### 1. **Dense (Fully Connected)**

**Forward**:
```python
output = input @ weights + biases
```

**Backward**:
```python
dweights = input.T @ output_gradient
dbiases = sum(output_gradient, axis=0)
input_gradient = output_gradient @ weights.T
```

**With Regularization**:
```python
dweights += l2 * 2 * weights + l1 * sign(weights)
```

**Key Insight**: Backpropagation is just chain rule with matrices!

---

### 2. **Batch Normalization**

**Forward (Training)**:
```python
batch_mean = mean(input, axis=0)
batch_var = var(input, axis=0)
normalized = (input - batch_mean) / sqrt(batch_var + epsilon)
output = gamma * normalized + beta

# Update running statistics
running_mean = momentum * running_mean + (1-momentum) * batch_mean
running_var = momentum * running_var + (1-momentum) * batch_var
```

**Forward (Inference)**:
```python
normalized = (input - running_mean) / sqrt(running_var + epsilon)
output = gamma * normalized + beta
```

**Backward** (complex!):
```python
dgamma = sum(output_gradient * normalized, axis=0)
dbeta = sum(output_gradient, axis=0)

# Gradient of normalized
dx_norm = output_gradient * gamma

# Gradient of variance
dvar = sum(dx_norm * (input - batch_mean) * -0.5 * (batch_var + epsilon)**(-1.5), axis=0)

# Gradient of mean
dmean = sum(dx_norm * -1/sqrt(batch_var + epsilon), axis=0) + dvar * sum(-2*(input-batch_mean), axis=0) / N

# Final input gradient
input_gradient = dx_norm/sqrt(batch_var+epsilon) + dvar*2*(input-batch_mean)/N + dmean/N
```

**Why Complex**: Must backpropagate through normalization, which depends on all samples in batch!

**Common Bug**: Forgetting to store `input` during forward pass (needed for backward)

---

### 3. **Activation Functions**

**ReLU**:
```python
forward: output = max(0, input)
backward: gradient = gradient * (input > 0)
```
- Pros: Simple, fast, no vanishing gradient
- Cons: Dead neurons (always output 0)

**Sigmoid**:
```python
forward: output = 1 / (1 + exp(-input))
backward: gradient = gradient * output * (1 - output)
```
- Pros: Smooth, bounded [0,1]
- Cons: Vanishing gradient, not zero-centered

**Softmax** (for multi-class):
```python
forward: 
    exp_values = exp(input - max(input, axis=1))
    output = exp_values / sum(exp_values, axis=1)
backward: 
    # Combined with cross-entropy: gradient = prediction - target
```
- **Key**: Subtract max for numerical stability!

---

### 4. **Dropout**

**Forward (Training)**:
```python
mask = random_binomial(1, 1-dropout_rate, shape) / (1-dropout_rate)
output = input * mask
```

**Forward (Inference)**:
```python
output = input  # No dropout!
```

**Backward**:
```python
input_gradient = output_gradient * mask
```

**Inverted Dropout**: Division by (1-rate) ensures expected value unchanged
- Without: Test outputs are (1-rate) × training outputs
- With: Outputs match in expectation

**Common Bug**: Applying dropout during inference!

---

## Loss Functions

### 1. **Categorical Cross-Entropy**

**Formula**:
```python
loss = -mean(sum(true * log(predicted), axis=1))
```

**Gradient** (with softmax):
```python
gradient = (predicted - true) / batch_size
```

**Numerical Stability**:
```python
predicted = clip(predicted, 1e-15, 1)  # Avoid log(0)
```

---

### 2. **Sparse Categorical Cross-Entropy**

**Difference**: True labels are indices, not one-hot vectors

**Formula**:
```python
loss = -mean(log(predicted[range(N), true_indices]))
```

**Gradient**:
```python
gradient = predicted.copy()
gradient[range(N), true_indices] -= 1
gradient /= batch_size
```

**Advantage**: Memory efficient for many classes

---

### 3. **Binary Cross-Entropy**

**Formula**:
```python
loss = -mean(true*log(predicted) + (1-true)*log(1-predicted))
```

**Gradient**:
```python
gradient = (predicted - true) / batch_size
```

---

### 4. **MSE (Regression)**

**Formula**:
```python
loss = mean((predicted - true)**2)
```

**Gradient**:
```python
gradient = 2 * (predicted - true) / batch_size
```

---

### 5. **Huber Loss (Robust Regression)**

**Formula**:
```python
error = abs(true - predicted)
loss = where(error <= delta, 
             0.5 * error**2,
             delta * (error - 0.5*delta))
```

**Gradient**:
```python
diff = predicted - true
gradient = where(abs(diff) <= delta,
                 diff,
                 delta * sign(diff)) / batch_size
```

**Benefit**: Combines MSE (small errors) + MAE (large errors)
- Less sensitive to outliers than MSE

---

## Bayesian Hyperparameter Optimization

### Search Space Definition

```python
space = {
    'learning_rate': [1e-4, 1e-2],  # Log-uniform
    'batch_size': [32, 64, 128],     # Categorical
    'dropout_rate': [0.1, 0.5],      # Uniform
    'optimizer': ['adam', 'rmsprop', 'momentum'],  # Categorical
    'batch_norm': [0, 1]  # Boolean (encoded as float)
}
```

---

### 1. **Gaussian Process (GP) Surrogate**

**Idea**: Model objective function as Gaussian process

**Kernel**: RBF (Radial Basis Function)
```python
K(x, x') = exp(-||x - x'||² / (2 * length_scale²))
```

**Prediction**:
```python
# Fit GP on observed points
GP.fit(X_observed, y_observed)

# Predict mean and variance at new points
mean, variance = GP.predict(X_candidates)
```

**Acquisition Function**: Upper Confidence Bound (UCB)
```python
score = mean + kappa * sqrt(variance)
# Pick candidate with highest score
```

**Balances**:
- **Exploitation**: Pick where mean is high
- **Exploration**: Pick where variance is high (uncertainty)

**Limitation**: Assumes smooth objective (may not hold for neural networks)

---

### 2. **Tree-structured Parzen Estimator (TPE)**

**Idea**: Model P(x|y) instead of P(y|x)

**Algorithm**:
1. Split observations into "good" (top γ quantile) and "bad"
2. Build density models l(x) for good, g(x) for bad
3. Acquisition: maximize l(x)/g(x)

**Densities**: Parzen windows (kernel density estimation)
```python
def parzen_window(x, samples, bandwidth):
    kernel = exp(-0.5 * ||x - samples||² / bandwidth²)
    return mean(kernel)
```

**Why Better for Neural Networks**:
- Handles categorical variables naturally
- No smoothness assumption
- More robust to noise

**Trade-off**: Needs more samples to build good densities

---

### 3. **Optimization Loop**

```python
bayesian_optimizer = BayesianOptimizer(param_space, sampler='tpe')

for iteration in range(n_iterations):
    # Suggest next configuration
    config, vector = bayesian_optimizer.suggest()
    
    # Train model with config
    model = NeuralNetwork(input_dim, n_classes, config)
    optimizer = HybridOptimizer(config['optimizer'], config['learning_rate'])
    history = train_model(model, X_train, y_train, X_val, y_val, optimizer, config)
    
    # Evaluate (objective to maximize)
    score = max(history['val_f1']) + max(history['val_r2'])
    
    # Register result
    bayesian_optimizer.register(vector, score)
    
    # Update best
    if score > best_score:
        best_score = score
        best_config = config
        best_model = model
```

**Key Insight**: Each iteration is expensive (train full model), so we want to minimize iterations!

---

## Critical Implementation Insights

### 1. **Weight Initialization - Make or Break**

**Problem**: Poor initialization causes vanishing/exploding gradients

**Example** (all zeros):
```python
weights = np.zeros((fan_in, fan_out))
# All neurons output same value
# All gradients identical
# Neurons never differentiate → symmetry breaking fails!
```

**Example** (too large):
```python
weights = np.random.randn(fan_in, fan_out) * 10
# Activations saturate (sigmoid → 0 or 1, ReLU → huge)
# Gradients vanish or explode
```

**Solution**: He initialization (for ReLU)
```python
weights = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
```

**Derivation**:
- Goal: Preserve variance across layers
- Variance(output) = Variance(input)
- For ReLU: E[ReLU(x)²] = 0.5 * E[x²] (half inputs zeroed)
- Need Var(w) = 2 / fan_in to compensate

**Xavier** (for sigmoid/tanh):
```python
weights = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))
```

---

### 2. **Forward Pass - Training vs. Inference**

**Training Mode**:
```python
def forward(self, X, training=True):
    out = X
    for layer in self.layers:
        out = layer.forward(out, training=True)  # Dropout active, BN uses batch stats
    return out
```

**Inference Mode**:
```python
def forward(self, X, training=False):
    out = X
    for layer in self.layers:
        out = layer.forward(out, training=False)  # No dropout, BN uses running stats
    return out
```

**Common Bug**: Using training=True during prediction → random dropout causes inconsistent outputs!

---

### 3. **Backpropagation - The Chain Rule**

**Conceptual**:
```
Loss → Output Layer → Hidden Layer 2 → Hidden Layer 1 → Input
  ↓         ↓                ↓                  ↓
∂L/∂o     ∂L/∂h2          ∂L/∂h1            ∂L/∂x
```

**Implementation**:
```python
def backward(self, gradient_cls, gradient_reg, optimizer):
    # Backprop through classification head
    g_cls = gradient_cls
    for layer in reversed(self.head_cls):
        if not isinstance(layer, (Softmax, Sigmoid)):  # Skip activation
            g_cls = layer.backward(g_cls)
            optimizer.update(layer)
    
    # Backprop through regression head
    g_reg = gradient_reg
    for layer in reversed(self.head_reg):
        g_reg = layer.backward(g_reg)
        optimizer.update(layer)
    
    # Combine gradients from both heads
    g_trunk = g_cls + g_reg
    
    # Backprop through shared trunk
    for layer in reversed(self.layers):
        g_trunk = layer.backward(g_trunk)
        optimizer.update(layer)
```

**Key Insight**: Multi-task learning shares trunk, splits into task-specific heads

---

### 4. **Mini-Batch Training**

**Why Batches?**
1. **Efficiency**: Vectorized operations faster than loops
2. **Memory**: Can't fit entire dataset in GPU
3. **Generalization**: Noise in batches acts as regularization
4. **Batch Normalization**: Requires batch statistics

**Implementation**:
```python
batch_size = 32
for epoch in range(n_epochs):
    # Shuffle data
    perm = np.random.permutation(n_samples)
    X_shuffled = X[perm]
    y_shuffled = y[perm]
    
    # Process batches
    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # Forward pass
        predictions = model.forward(X_batch, training=True)
        
        # Compute loss and gradients
        loss = compute_loss(predictions, y_batch)
        gradients = compute_gradients(predictions, y_batch)
        
        # Backward pass
        model.backward(gradients, optimizer)
```

**Batch Size Trade-offs**:
- **Small (16-32)**: More noise, better generalization, slower
- **Large (256-512)**: Less noise, faster, may overfit

---

### 5. **Gradient Clipping (Not Implemented, But Important)**

**Problem**: Exploding gradients in deep networks

**Solution**: Clip gradient norm
```python
def clip_gradients(gradients, max_norm=5.0):
    grad_norm = np.sqrt(np.sum(gradients ** 2))
    if grad_norm > max_norm:
        gradients *= max_norm / grad_norm
    return gradients
```

---

## The Hard Parts

### 1. **Vanishing Gradients**
Early layers barely learn. Deep network + sigmoid activations = gradients die.

**Fix**: Use ReLU, He initialization, batch normalization

### 2. **Exploding Gradients**
Weights blow up to infinity, loss becomes NaN.

**Fix**: Lower learning rate, gradient clipping, batch norm

### 3. **Dead ReLUs**
Neuron always outputs 0, gradient always 0, stuck forever.

**Fix**: Better initialization, lower learning rate, or try Leaky ReLU

### 4. **Overfitting**
Train loss down, val loss up.

**Fix**: Dropout (0.3-0.5), L2 regularization, more data, early stopping

---

## Advanced Techniques (Referenced in Code)

### 1. **Multi-Task Learning**

**Architecture**:
```
       Input
         |
    Shared Trunk (Dense layers)
       / | \
      /  |  \
   Task1 Task2 ...
```

**Benefits**:
- Shared representations
- Regularization through multiple objectives
- Better generalization

**Loss**:
```python
total_loss = loss_cls + loss_reg
```

---

### 2. **Macro F1 Score**

**Why Not Accuracy?**
- Imbalanced classes: 95% class 0, 5% class 1
- Accuracy 95% by always predicting class 0!
- F1 balances precision and recall

**Implementation**:
```python
def macro_f1_score(y_true, y_pred):
    f1_scores = []
    for cls in unique_classes:
        tp = sum((y_pred == cls) & (y_true == cls))
        fp = sum((y_pred == cls) & (y_true != cls))
        fn = sum((y_pred != cls) & (y_true == cls))
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        f1_scores.append(f1)
    
    return mean(f1_scores)  # Average across classes
```

---

### 3. **Learning Rate Schedules**

**Step Decay**:
```python
lr = initial_lr * (decay_rate ** (epoch // drop_every))
```

**Exponential Decay**:
```python
lr = initial_lr * exp(-decay_rate * epoch)
```

**Cosine Annealing**:
```python
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * epoch / total_epochs))
```

---

## Debugging Checklist

### Training Not Starting
- [ ] Learning rate too small (<1e-5)?
- [ ] Weights initialized correctly?
- [ ] Gradients flowing (check backward pass)?
- [ ] Data preprocessed (scaled, no NaN)?

### Loss Exploding
- [ ] Learning rate too high?
- [ ] Gradients clipped?
- [ ] Batch normalization used?
- [ ] Check for NaN in data

### Poor Accuracy
- [ ] Model capacity sufficient?
- [ ] Overfitting (train vs. val gap)?
- [ ] Underfitting (both low)?
- [ ] Data imbalance?
- [ ] Learning rate schedule?

### Slow Convergence
- [ ] Learning rate too small?
- [ ] Batch size too small?
- [ ] Batch normalization helps?
- [ ] Better optimizer (Adam vs. SGD)?

---

## Key Takeaways

✅ **Flexible**: Handles classification + regression simultaneously
✅ **Modern Techniques**: Batch norm, dropout, multiple optimizers
✅ **Hyperparameter Optimization**: TPE/GP for automatic tuning
✅ **Production-Ready**: Proper train/inference modes, regularization
✅ **Interpretable Training**: Loss curves, metrics tracking

❌ **Complex**: Many moving parts, hard to debug
❌ **Hyperparameter Sensitive**: Requires careful tuning
❌ **Computationally Expensive**: Training takes time
❌ **Black Box**: Hard to interpret learned features

**When to Use**:
- Complex non-linear patterns
- Large datasets (>10K samples)
- Need multi-task learning
- High accuracy priority
- Computational resources available

**When to Avoid**:
- Small datasets (<1K samples)
- Need interpretability
- Simple patterns (use linear/tree models)
- Limited computational resources
- Fast prototyping needed

**Best Practices**:
1. Start simple (2 layers, no regularization)
2. Verify gradient flow (small synthetic data)
3. Use Adam optimizer (usually best default)
4. Monitor train/val curves (detect over/underfitting)
5. Use Bayesian optimization for hyperparameters
6. Always use batch normalization for deep networks
7. Apply dropout for regularization
8. Save best model during training
9. Use early stopping
10. Visualize predictions and mistakes

This implementation represents months of ML engineering compressed into production-grade code!
