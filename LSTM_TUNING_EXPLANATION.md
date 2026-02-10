# 🧠 LSTM Hyperparameter Tuning - Complete Explanation

## Overview

LSTM (Long Short-Term Memory) networks are powerful deep learning models for time-series forecasting. However, they have many hyperparameters that significantly affect performance. This guide explains:

1. **What each hyperparameter does**
2. **Why we're testing specific values**
3. **The tuning strategy**
4. **How to interpret results**

---

## 📊 Hyperparameter Explanations

### 1. **Sequence Length (seq_length)** [7, 14, 21, 30]

**What it is**: Number of past days the LSTM "looks back" to predict the next day

```
seq_length = 7
Today = Day 8
↑
Uses Days 1,2,3,4,5,6,7

seq_length = 30  
Today = Day 31
↑
Uses Days 1-30
```

**Why test different values**:
- **Too short (7 days)**: Models misses long-term patterns, prone to noise
  - ❌ Can't catch monthly trends or seasonal cycles
  - ✅ Faster training, less computation

- **Medium (14 days)**: Captures weekly patterns (good for fibre subscriptions!)
  - ✅ Weekly seasonality is natural in telecom data
  - ✅ Balanced training time

- **Longer (21-30 days)**: Captures 3-4 weeks of history
  - ✅ More context, better for complex patterns
  - ❌ Slower training, more parameters to learn

**Intuition for Fibre Data**: Telecom subscriptions have weekly patterns (business hours vs weekends), so 14-21 days is likely optimal.

---

### 2. **Learning Rate (lr)** [0.0001, 0.001, 0.01]

**What it is**: How aggressively the neural network updates its weights during training

```
Loss Function (Cost)
        ↓
        ↓ Learning Rate = 0.0001 (small steps - slow)
        ↓ 
    ╱ ╲ ╱ ╲ ╱ ╲      Takes many updates to reach bottom
   ╱   ╲╱   ╲╱ ╲
  ╱    Minimum

        ↓
        ↓ Learning Rate = 0.001 (medium steps - balanced)
        ↓
    ╱   ╲╱        Reaches bottom efficiently
   ╱     ╲
  ╱      Minimum

        ↓
        ↓ Learning Rate = 0.01 (large steps - risky)
        ↓
    ╱     ╲ ╱     Might overshoot or diverge!
   ╱       ✗       
  ╱
```

**Why test different values**:
- **Too small (0.0001)**: 
  - ✅ More stable, less risk of diverging
  - ❌ Takes forever to train, might not converge
  
- **Medium (0.001)**: **Default for Adam optimizer**
  - ✅ Good for most cases
  - ✅ Balanced convergence speed
  
- **Larger (0.01)**:
  - ✅ Faster training
  - ❌ Risk of overshooting optimal weights, unstable convergence

**For Time-Series**: Usually 0.001 or 0.0005 works best. We test to find the sweet spot.

---

### 3. **Batch Size** [8, 16, 32]

**What it is**: Number of training samples processed before updating weights

```
Batch Size = 8:
Update 1 (8 samples)  →  Update weights
Update 2 (8 samples)  →  Update weights
Update 3 (8 samples)  →  Update weights
(16 updates per epoch)

Batch Size = 16:
Update 1 (16 samples)  →  Update weights
Update 2 (16 samples)  →  Update weights
(8 updates per epoch)

Batch Size = 32:
Update 1 (32 samples)  →  Update weights
Update 2 (32 samples)  →  Update weights
(4 updates per epoch)
```

**Why test different values**:
- **Small (8)**:
  - ✅ More frequent weight updates = more learning
  - ✅ Better for noisy data
  - ❌ Noisier gradients, can be unstable
  - ❌ Slower overall (more iterations)
  
- **Medium (16)**: **Good balance**
  - ✅ Moderate noise, stable convergence
  - ✅ Fast enough training
  
- **Large (32)**:
  - ✅ Smooth gradient estimates
  - ✅ Faster (fewer iterations)
  - ❌ Might miss nuances in data
  - ❌ Get stuck in local minima

**For Fibre Data**: Small batch (8-16) usually better for capturing daily variations.

---

### 4. **Dropout Rate** [0.1, 0.2, 0.3]

**What it is**: Randomly disable a percentage of neurons during training to prevent overfitting

```
Normal LSTM Layer:
[o] → [o] → [o] → [o] → [o]

Dropout = 0.1 (10% disabled):
[o] → [✗] → [o] → [o] → [o]  (randomly ~10% are "dropped out")

Dropout = 0.2 (20% disabled):
[o] → [✗] → [o] → [✗] → [o]  (randomly ~20% are "dropped out")

Dropout = 0.3 (30% disabled):
[✗] → [o] → [✗] → [o] → [✗]  (randomly ~30% are "dropped out")
```

**Why test different values**:
- **Too low (0.1)**:
  - ❌ Less regularization, higher risk of overfitting
  - ✅ Model can learn complex patterns
  
- **Medium (0.2)**: **Common default**
  - ✅ Balances generalization and learning
  - ✅ Prevents overfitting without losing capacity
  
- **Too high (0.3)**:
  - ✅ Strong regularization, less overfitting
  - ❌ Model underfits, can't learn complex patterns

**For Fibre Data**: 0.15-0.25 usually optimal. We test to find exact sweet spot.

---

### 5. **LSTM Units (Layer 1 & 2)**

**What it is**: Number of internal "memory cells" in each LSTM layer

```
Current Architecture:
Input → [LSTM: 64 units] → [LSTM: 32 units] → Output

Each LSTM unit = a small neural network with memory

64 units = 64 parallel "learners"
32 units = 32 parallel "learners" (fewer, more compressed)
```

**Why test different values**:
- **Fewer units (32, 16)**:
  - ✅ Faster training
  - ✅ Less overfitting risk
  - ❌ Less capacity to learn complex patterns
  
- **More units (64, 128)**:
  - ✅ Greater model capacity
  - ✅ Can capture complex temporal patterns
  - ❌ Slower training
  - ❌ Higher overfitting risk
  - ❌ Need more data

**For Fibre Data**: 64 → 32 (current) is good. We test if 32→16 (smaller) or 128→64 (larger) is better.

---

### 6. **Dense Layer Units** [8, 16, 32]

**What it is**: Number of neurons in the final fully-connected layer before output

```
LSTM Output → [Dense Layer: 16 units] → [1 unit] → Prediction

Fewer (8):     Faster, less capacity
Medium (16):   Good balance
More (32):     Slower, more capacity
```

**Why test different values**:
- Small dense layer forces efficient representation
- Large dense layer can extract more patterns
- Usually 16 is optimal for fibre data

---

## 🎯 Tuning Strategy Used

I'm using a **Strategic Grid Search** (not full brute-force) to keep computation time reasonable:

### Strategy:
1. **Fixed baseline**: Test defaults first
2. **One-at-a-time variation**: Change one hyperparameter, keep others fixed
3. **Independent evaluation**: Understand how each affects accuracy
4. **Combined optimization**: Blend best findings into final config

### Why this approach?
- ✅ **Feasible**: ~15 configs instead of 1,000+
- ✅ **Interpretable**: See which params matter most
- ✅ **Efficient**: Find good solutions quickly
- ⚠️ **May miss interactions**: Some params work better together

---

## 📈 What We're Optimizing For

### Primary Metric: **MAPE** (Mean Absolute Percentage Error)
- Lower is better (goal: < 10% for time-series)
- Measures accuracy as a percentage

### Secondary Metrics:
- **MAE**: Average prediction error in actual units
- **RMSE**: Penalizes large errors more
- **Training Time**: Computational efficiency

### Trade-offs:
```
Accuracy vs Speed:
- More epochs → Better accuracy but slower
- Larger batch → Faster but less learning opportunity
- More units → Better accuracy but slower
```

---

## 🚀 How to Run the Tuning

```bash
cd /home/habib/fibre_data_project/projet-fibre-forecast

# Run the tuning script (will test 15 different configurations)
python tune_lstm_hyperparams.py
```

**Expected output**:
- Tests each configuration with timing
- Displays top 5 best configurations
- Shows best hyperparameters
- Calculates improvement over baseline
- Saves JSON and CSV results to `outputs/lstm_tuning/`

---

## 📊 How to Interpret Results

### Example Output:
```
🥇 BEST CONFIGURATION

Configuration Name: Seq Length: 14 days

Hyperparameters:
   • Sequence Length:     14 days
   • LSTM Layer 1 Units:  64
   • LSTM Layer 2 Units:  32
   • Dropout Rate:        0.2
   • Batch Size:          16
   • Learning Rate:       0.001

Results:
   • MAPE:   8.45%          ← Lower than baseline (12.75%) = Better!
   • MAE:    152.34
   • RMSE:   189.23
   • Training Time: 15.42s

📈 IMPROVEMENT vs BASELINE:
   Baseline MAPE: 12.75%
   Best MAPE:     8.45%
   Improvement:   ✅ 33.7% better
```

### Interpreting the Numbers:
- **MAPE 8.45%**: On average, predictions are off by 8.45% of actual values
- **33.7% better**: This config is 33.7% more accurate than the original
- **Training Time**: How long it took to train this specific configuration

---

## 🔍 Common Patterns to Watch For

### Sign of Overfitting:
- Training loss decreases, but test MAPE increases
- Solution: Increase dropout or reduce complexity

### Sign of Underfitting:
- Both training and test losses high
- Solution: Increase model capacity (more units) or train longer

### Good Fit:
- Training loss decreases smoothly
- Test MAPE is low and stable

---

## Next Steps After Tuning

1. **If improvement > 20%**: Use new hyperparameters → create new run_lstm_model.py
2. **If improvement 5-20%**: Conditional - use if deployment priority is accuracy
3. **If improvement < 5%**: Stick with baseline (simpler to maintain)
4. **If TensorFlow still unavailable**: Skip LSTM, use other 4 models

---

## Key Takeaway

Architecture choices matter, but time-series forecasting is about:
1. **Right algorithm choice** (SARIMA beats LSTM for fibre data typically)
2. **Quality of features/data** (more important than hyperparameters)
3. **Proper train/test split** (temporal ordering matters!)
4. **Domain knowledge** (weekly patterns in telecom)

LSTM is powerful but needs careful tuning and data. For your fibre dataset, SARIMA (5.38% MAPE) might still outperform tuned LSTM unless we have much more data.

