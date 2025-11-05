# Astrocyte-Inspired Hierarchical Routing for MoE Models

Official implementation of **"Astrocyte-Inspired Hierarchical Routing for Enhanced Expert Specialization in Mixture-of-Experts Models"** (under review at TMLR).

## Overview

This repository implements **Astrocyte-Hierarchical Routing (AHR)**, a novel bio-inspired routing mechanism for Mixture-of-Experts (MoE) models that addresses the persistent challenge of cultivating genuine expert specialization. 

### Key Innovation

Traditional MoE models struggle with the "specialization paradox" where load-balancing losses inhibit expert differentiation. AHR solves this by:

- **Hierarchical Routing**: Conditioning token-level routing decisions on global sequence context (derived from [CLS] token)
- **Developmental Trajectory**: Fostering generalist experts in early layers that feed into highly specialized experts in later layers
- **Preserved Performance**: Achieving 82% (0.1088 - 0.0598) / 0.0598) increase in last-layer specialization with no accuracy degradation

### Results Summary

| Model | Accuracy | Last Layer Specialization |
|-------|----------|--------------------------|
| Dense Baseline | 94.31% | N/A |
| Softmax MoE | 94.15% | 0.0598 |
| **AHR (Ours)** | **94.23%** | **0.1088** ⭐ |

## Architecture

```
Input Sequence → [CLS] Token Extraction
                      ↓
              Global Context (Wa)
                      ↓
    ┌─────────────────┴─────────────────┐
    ↓                                   ↓
Token Routing (Wr)              Additive Bias
    ↓                                   ↓
    └─────────────────┬─────────────────┘
                      ↓
            Modified Logits (Lmod)
                      ↓
              Top-K Selection → Experts
```

## Installation

### Requirements

```bash
# Create conda environment
conda create -n astrocyte-moe python=3.8
conda activate astrocyte-moe

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0
evaluate>=0.4.0
scikit-learn>=1.3.0
pandas>=2.0.0
scipy>=1.10.0
tqdm>=4.65.0
```

## Quick Start

### 1. Basic Training

Run the full ablation study with default settings:

```bash
python run_final_experiment.py
```

This will train and evaluate all model variants across 5 random seeds.

### 2. Debug Mode

Test the pipeline quickly with minimal data:

```python
# In run_final_experiment.py, set:
DEBUG = True  # Uses 200 train/100 eval samples, 1 epoch
```

```bash
python run_final_experiment.py
```

### 3. Custom Configuration

Modify hyperparameters in the script:

```python
# Model configuration
ablation_configs = {
    "astrocyte-hierarchical": {
        "is_moe": True,
        "router_type": "astrocyte-hierarchical",
        "num_experts": 8,
        "energy_loss_alpha": 0.01,
        "alpha_balance": 0.01,
    }
}

# Training configuration
NUM_SEEDS = 5  # Number of random seeds
base_training_args = TrainingArguments(
    num_train_epochs=6,
    per_device_train_batch_size=192,
    learning_rate=5e-5,
    # ... other args
)
```

## Model Variants

The codebase supports comprehensive ablation studies:

| Variant | Description | Key Features |
|---------|-------------|--------------|
| `dense` | Standard BERT baseline | No MoE, pre-trained weights |
| `dense-random-ffn` | Dense with reset FFNs | Fair cold-start comparison |
| `softmax-moe` | Standard MoE | Top-K softmax gating |
| `softmax-moe-energy` | Softmax + energy loss | Additional L2 regularization |
| `astrocyte` | Multiplicative modulation | [CLS] sigmoid gating |
| `astrocyte-no-energy` | Astrocyte w/o energy | Only load balance loss |
| `astrocyte-unleashed` | Extreme specialization | No balance, high energy |
| **`astrocyte-hierarchical`** | **AHR (proposed)** | **Additive [CLS] bias** ⭐ |
| `astrocyte-meanpool` | Mean-pool ablation | Global context via averaging |

## Project Structure

```
.
├── run_final_experiment.py      # Main experiment script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── results/                      # Model checkpoints & outputs
│   ├── astrocyte-hierarchical-E8-LB0.01-EN0.01-seed0/
│   │   ├── final-model/         # Trained model weights
│   │   ├── analysis_plots/      # Per-run visualizations
│   │   └── checkpoint-*/        # Training checkpoints
│   └── [other configs]/
├── logs/                         # TensorBoard logs
│   └── [model-config]/
└── ablation_plots/               # Comparative analysis
    ├── ablation_accuracy_comparison.png
    ├── ablation_specialization_comparison_first_layer.png
    └── ablation_specialization_comparison_last_layer.png
```

## Outputs & Analysis

### 1. Training Logs

View real-time training with TensorBoard:

```bash
tensorboard --logdir=./logs
```

Navigate to `http://localhost:6006` to see:
- Loss curves (classification, auxiliary, total)
- Evaluation accuracy
- Expert utilization histograms

### 2. Model Checkpoints

Each run saves:
- `final-model/`: Best model weights
- `eval_results.json`: Final metrics
- Training state for resumption

### 3. Visualizations

#### Per-Model Analysis (`results/[config]/analysis_plots/`)

- **`topic_utilization_last_layer.png`**: Heatmap showing expert specialization by topic
- **`topic_utilization_all_layers.png`**: Layer-wise specialization evolution
- Qualitative token routing examples (console output)

#### Ablation Comparison (`ablation_plots/`)

- **Accuracy Comparison**: Bar chart with statistical significance (t-tests)
- **Specialization Comparison**: First/last layer metrics across all variants
- Raw results saved to `ablation_all_results.json`

## Key Concepts

### Specialization Score

Quantifies expert differentiation:

```
Score = mean(std(expert_utilization_by_topic))
```

- **High score**: Expert activates selectively for specific topics
- **Low score**: Expert processes all topics uniformly

### Hierarchical Routing Mechanism

```python
# Standard routing
Ltoken = X · Wr  # Token-level logits

# AHR augmentation
xcls = X[0]                    # Extract [CLS] token
Lglobal = xcls · Wa            # Global context bias
Lmod = Ltoken + Lglobal        # Additive modulation

# Top-K selection
weights = softmax(Lmod)
top_k_experts = select_topk(weights, k=2)
```

### Biological Inspiration

| Biological Component | Computational Analog |
|---------------------|---------------------|
| Astrocyte cell | Global context module (Wa) |
| Synapses | Token-level routers (Wr) |
| Gliotransmitters | Additive bias (Lglobal) |
| Tripartite synapse | Hierarchical routing (token + context) |

## Reproducibility

### Deterministic Training

The code sets all random seeds for reproducibility:

```python
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
training_args.seed = seed
```

### Statistical Validation

- All experiments run across **N=5 random seeds**
- Results reported as **mean ± standard error**
- Significance tested via **Welch's t-test** (p < 0.05)

### Computational Requirements

**Minimal Configuration** (DEBUG mode):
- GPU: 8GB VRAM (e.g., RTX 3070)
- RAM: 16GB
- Time: ~5 minutes

**Full Experiment** (5 seeds × 9 configs):
- GPU: 12GB+ VRAM (e.g., RTX 3080)
- RAM: 32GB recommended
- Time: ~4-6 hours

### Optimization Tips

For RTX 30-series or newer:
```python
torch.backends.cuda.matmul.allow_tf32 = True  # Enabled by default
torch.backends.cudnn.benchmark = True
```

## Extending the Code

### Adding New Router Types

1. Define router logic in `MoELayer.__init__()`:
```python
if self.config.router_type == "my-router":
    self.my_module = nn.Linear(...)
```

2. Implement forward pass in `MoELayer.forward()`:
```python
elif self.config.router_type == "my-router":
    router_logits = self.compute_my_routing(hidden_states)
```

3. Add to ablation configs:
```python
"my-router": {
    "is_moe": True,
    "router_type": "my-router",
    # ... hyperparams
}
```

### Custom Datasets

Replace AG News with your dataset:
```python
script_args = ScriptArguments(
    dataset_name="your-dataset",  # HuggingFace dataset name
    model_name_or_path="bert-base-uncased",
    # ... other args
)
```

Ensure dataset has:
- `text` field (input)
- `label` field (target)

## Citation

```bibtex
@article{astrocyte-moe-2025,
  title={Astrocyte-Inspired Hierarchical Routing for Enhanced Expert Specialization in Mixture-of-Experts Models},
  author={Anonymous},
  journal={Under review at Transactions on Machine Learning Research (TMLR)},
  year={2025}
}
```

## Hypotheses & Scientific Method

This codebase follows rigorous scientific principles:

**Hypothesis 1**: Hierarchical routing (global context + local decisions) promotes layer-wise specialization development.

**Validation**: ✅ Confirmed via quantitative specialization scores (0.0139 → 0.1088 from first to last layer)

**Hypothesis 2**: AHR achieves superior specialization without accuracy trade-off.

**Validation**: ✅ Confirmed via statistical comparison (94.23% accuracy, p=0.298 vs dense baseline)

**Hypothesis 3**: Additive bias outperforms multiplicative gating.

**Testing**: Compare `astrocyte-hierarchical` vs `astrocyte-meanpool` results

## Known Limitations

1. **Scale**: Tested on 4-layer, 8-expert models. Large-scale validation needed.
2. **Task Scope**: Evaluated on text classification only. Generalization to generation/multimodal tasks unverified.
3. **Decoder Adaptation**: Current [CLS] approach requires modification for causal LM architectures.

See paper Section 6.3 for detailed discussion.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Base MoE implementation inspired by [HuggingFace Transformers](https://github.com/huggingface/transformers)
- Biological concepts from neuroscience literature (Oschmann et al., 2018; Kozachkov et al., 2025)
- AG News dataset from [Zhang et al., 2015](https://arxiv.org/abs/1509.01626)

## Contact

For questions or issues, please open a GitHub issue or contact [anonymous for review].

---

⭐ **Star this repo** if you find it useful for your research!
