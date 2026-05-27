# ML From Scratch 🧠📐

![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy_Only-green)
![Paradigm](https://img.shields.io/badge/Paradigm-Bare_Metal_Math-orange)
![Focus](https://img.shields.io/badge/Focus-Algorithm_Architecture-purple)

A comprehensive, bare-metal implementation of foundational Machine Learning, Deep Learning, and Natural Language Processing algorithms written entirely in pure Python and NumPy. 

This repository serves as a mathematical playground and proof-of-concept proving that modern AI architectures can be built, trained, and understood without relying on black-box frameworks like PyTorch, TensorFlow, or Scikit-Learn.

**Author:** Tanmay Janak

---

## 🚀 Repository Philosophy & Engineering Standards

* **Zero-Framework Dependency:** The core logic for forward passes, backpropagation, and sequence attention relies strictly on NumPy matrix operations.
* **Vectorized Execution:** Avoids standard Python `for` loops in favor of highly optimized tensor broadcasting and dot-products (e.g., executing $O(1)$ spatial distance calculations and cross-correlation sliding windows).
* **Numerical Stability:** Implements production-grade mathematical safeguards, including LogSumExp tricks for Softmax, Laplace smoothing for TF-IDF, and gradient clipping for RNNs.
* **Architectural Modularity:** Deep learning components (Optimizers, Layers, Activations) are built as independent, composable classes mimicking modern framework APIs.

---

## 📂 Implementation Roadmap

### Week 1: Classical Machine Learning
Foundational statistical learning and tree-based ensembles.
* `linear_regression.py` - Gradient descent optimization for continuous targets.
* `logistic_regression.py` - Binary classification utilizing Sigmoid activations and log-loss.
* `knn.py` - Non-parametric lazy learning using Euclidean distance mapping.
* `decision_tree.py` - Recursive dataset partitioning optimized via Shannon Entropy and Information Gain.
* `random_forest.py` - Variance reduction via Bootstrap Aggregating (Bagging) custom tree modules.

### Week 2: Deep Learning Mechanics
Feed-forward architectures, spatial convolutions, and parameter optimization routines.
* `mlp.py` - Binary Multi-Layer Perceptron utilizing explicit Chain Rule Backpropagation.
* `multiclass_mlp.py` - Advanced classification using ReLU, Softmax, and Categorical Cross-Entropy with He/Kaiming initialization.
* `conv2d.py` - Bare-metal spatial sliding windows, cross-correlation mapping, and multi-channel backpropagation.
* `adam.py` - Adaptive Moment Estimation optimizer featuring momentum, RMSProp, and bias correction.
* `dropout.py` - Network regularization utilizing Inverted Dropout scaling to maintain inference integrity.

### Week 3: Natural Language Processing & Advanced Metrics
Sequence modeling, attention mechanisms, and few-shot spatial clustering.
* `tfidf.py` - Term Frequency-Inverse Document Frequency vectorizer with L2 normalization.
* `word2vec.py` - Skip-Gram neural network architecture for extracting dense semantic word embeddings.
* `rnn.py` - Recurrent neural network implementing Backpropagation Through Time (BPTT) with exploding gradient prevention.
* `self_attention.py` - Scaled Dot-Product Attention mechanism (the core of the Transformer), vectorizing Query, Key, and Value projections.
* `prototypical_net.py` - N-way K-shot episodic metric learning for few-shot classification in latent geometric spaces.

---

## 💻 Local Setup & Execution

### Prerequisites
* Python 3.10+
* NumPy

### Installation
Clone the repository and install the minimal requirements:
```bash
git clone [https://github.com/yourusername/ml-from-scratch.git](https://github.com/yourusername/ml-from-scratch.git)
cd ml-from-scratch
pip install numpy
```

### Running the Implementations
Every algorithm is designed as a standalone module with an integrated `__main__` test block. Simply execute the file to watch the algorithm initialize, train on simulated data, and output its mathematical evaluations.

Example:
```bash
python multiclass_mlp.py
```
*Output:*
```text
📊 Generating 3-Class non-linear spiral distribution...
🧠 Initializing deep Multi-Class Network architecture...
🔄 Epoch    0 | Loss: 1.1084 | Training Accuracy: 33.3%
...
🔄 Epoch 5000 | Loss: 0.1421 | Training Accuracy: 96.7%
✅ Optimization Complete. Final Target Accuracy mapping: 96.67%
```

---
*Note: This repository is built for educational, structural, and mathematical transparency. For massive-scale production deployments utilizing distributed GPU clusters, refer to compiled frameworks (PyTorch/JAX).*