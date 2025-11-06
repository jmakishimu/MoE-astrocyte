# Astrocyte-Inspired Hierarchical Routing for MoE Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Astrocyte-Inspired Hierarchical Routing for Enhanced Expert Specialization in Mixture-of-Experts Models"** (under review at TMLR).

## TL;DR

**Problem**: MoE models struggle with expert specialization due to load-balancing constraints.  
**Solution**: Bio-inspired hierarchical routing that conditions token-level decisions on global sequence context.  
**Result**: **82% increase** in expert specialization with **no accuracy loss** (94.23% vs 94.31% baseline).

## Key Results

| Model | Accuracy | Last Layer Specialization | Improvement |
|-------|----------|--------------------------|-------------|
| Dense Baseline | 94.31% ± 0.05% | N/A | - |
| Softmax MoE (best) | 94.11% ± 0.08% | 0.0598 | baseline |
| **AHR (Ours)** | **94.23% ± 0.07%** | **0.1088** | **+82%** ⭐ |

## Architecture

```
Input Sequence: X ∈ ℝ^(S×d)  [S tokens, d dimensions]
         |
         ├──────────────────────────────┐
         ↓                              ↓
    [CLS] Token                    All Tokens
    X[0] ∈ ℝ^d                     X ∈ ℝ^(S×d)
         ↓                              ↓
    Wa: ℝ^d → ℝ^N                  Wr: ℝ^d → ℝ^N
    (astrocyte)                    (synapses)
         ↓                              ↓
    Lglobal ∈ ℝ^N              Ltoken ∈ ℝ^(S×N)
    (gliotransmitter)          (neurotransmitters)
         |                              |
         └────────(broadcast)───────────┤
                                        ↓
                    Lmod = Ltoken + Lglobal  (tripartite synapse)
                                        ↓
                                Softmax + Top-K=2
                                        ↓
                              Weighted Expert Outputs
```

**Key Innovation**: Global [CLS]-based bias additively modulates all token routing decisions, creating consistent pathways that promote specialization.

## Installation

```bash
conda create -n astrocyte-moe python=3.8
conda activate astrocyte-moe
pip install torch>=2.0.0 transformers>=4.30.0 datasets evaluate \
    matplotlib seaborn tensorboard pandas scipy tqdm
```

## Quick Start

### Full Experiment (5 seeds × 9 configs)

```bash
python run_final_experiment.py
```

**Runtime**: ~4-6 hours (RTX 3080, 12GB VRAM)

### Debug Mode (Quick Test)

```python
# In run_final_experiment.py, line 83:
DEBUG = True
```

```bash
python run_final_experiment.py
```

**Runtime**: ~5-10 minutes

## Model Variants

| Variant | Router Type | Purpose |
|---------|-------------|---------|
| `dense` | N/A | Baseline performance |
| `softmax-moe` | Standard softmax | Standard MoE baseline |
| **`astrocyte-hierarchical`** | **AHR (additive [CLS] bias)** | **Main contribution** ⭐ |
| `astrocyte-meanpool` | Mean-pooled context | Ablation: context source |
| `astrocyte` | Multiplicative gating | Ablation: modulation type |

See paper Section 4.2 for full list.

## Key Implementation

```python
# Standard routing (baseline)
router_logits = hidden_states · Wr  # (B, S, E)

# AHR routing (proposed)
cls_token = hidden_states[:, 0, :]              # (B, H)
global_bias = cls_token · Wa                     # (B, E)
router_logits = (hidden_states · Wr) + global_bias  # Broadcast
```

## Outputs

### Training Logs
```bash
tensorboard --logdir=./logs
```

### Results Structure
```
results/
├── [config]-seed[N]/
│   ├── final-model/              # Best checkpoint
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── eval_results.json
│   └── analysis_plots/           # Per-model heatmaps
ablation_plots/
├── ablation_accuracy_comparison.png
├── ablation_specialization_comparison_last_layer.png
└── ablation_all_results.json
```

### Key Visualizations

**Last Layer Specialization Heatmap**: Shows expert-topic affinity (e.g., Expert 5 → 59.6% Sports)  
**Layer-wise Evolution**: Demonstrates generalist (Layer 0) → specialist (Layer 3) trajectory

## Specialization Score

Quantifies expert differentiation:

```python
# For each layer:
U[topic, expert] = % of topic's tokens routed to expert
Specialization = mean(std(U, axis=topics))
```

**Interpretation**:
- High score → Expert specializes in specific topics
- Low score → Expert generalizes across all topics

## Requirements

**Minimal** (DEBUG mode):
- GPU: 8GB VRAM (RTX 3070)
- RAM: 16GB
- Time: ~10 minutes

**Full Experiment**:
- GPU: 12GB VRAM (RTX 3080)
- RAM: 32GB
- Time: ~4-6 hours
- Storage: 50GB

## Biological Metaphor

| Biology | Computation |
|---------|-------------|
| Astrocyte (integrates ~100K synapses) | [CLS] token → global bias |
| Synapses (local connections) | Token-level routing |
| Gliotransmitters (modulators) | Additive bias (Lglobal) |
| Tripartite synapse | Local + global integration |

## Extending the Code

### Add New Router

```python
# In MoELayer.__init__():
if self.config.router_type == "my-router":
    self.my_module = nn.Linear(...)

# In MoELayer.forward():
elif self.config.router_type == "my-router":
    router_logits = self.compute_my_routing(hidden_states)
```

### Use Different Dataset

```python
base_run_config = {
    "dataset_name": "your-dataset",  # Must have 'text' and 'label'
    # ...
}
```

## Troubleshooting

**OOM Errors**: Reduce `per_device_train_batch_size` to 96 or 128  
**No Plots**: Check `tensorboard --logdir=./logs` path  
**Slow Training**: Enable `torch.backends.cuda.matmul.allow_tf32 = True` (line 631)

## Citation

```bibtex
@article{astrocyte-hierarchical-routing-2025,
  title={Astrocyte-Inspired Hierarchical Routing for Enhanced Expert 
         Specialization in Mixture-of-Experts Models},
  author={Anonymous},
  journal={Under review at TMLR},
  year={2025}
}
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contact

Under double-blind review. Please open a GitHub issue for questions.

---

⭐ **Star this repo** if you find it useful!
