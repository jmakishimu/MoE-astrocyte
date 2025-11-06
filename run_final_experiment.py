# -*- coding: utf-8 -*-;
"""
Astrocyte-Inspired Hierarchical Routing for Enhanced Expert Specialization in Mixture-of-Experts Models

1.  Full Ablation Runner:
    - The main script now runs a full suite of ablation models:
      - 'dense' (baseline BERT)
      - 'softmax-moe' (standard MoE)
      - 'softmax-moe-energy' (standard MoE + energy loss)
      - 'astrocyte' (Astrocyte MoE + energy loss)
      - 'astrocyte-no-energy' (Astrocyte MoE, no energy loss)
      - 'dense-random-ffn' (Baseline BERT w/ re-initialized FFNs)
      - 'astrocyte-unleashed' (Strategy 1: No CV loss, high energy loss)
      - 'astrocyte-hierarchical' (Strategy 3A: Global context as bias)
      - 'astrocyte-meanpool' (Strategy 3B: Mean-pooled global context)
2.  Multi-Seed Execution:
    - Runs the entire ablation suite `NUM_SEEDS` times (default: 3)
      to gather robust statistics.
    - Sets all random seeds (torch, numpy) for each run.
3.  Memory Management:
    - Calls `gc.collect()` and `torch.cuda.empty_cache()`
      between each model run to prevent VRAM accumulation.
4.  Statistical Comparison Plot:
    - Gathers 'eval_accuracy' from all runs.
    - Generates a final bar plot (`ablation_accuracy_comparison.png`)
      showing the mean and standard error for each model.
    - Includes statistical significance (t-test p-value) for each
      MoE model compared to the 'dense' baseline.
5.  Checkpoint Reusability (Preserved):
    - All checkpointing logic is preserved *per model, per seed*.
    - e.g., './results/astrocyte-E8...-seed0/'
    - The script skips any run that already has a 'final-model'
      and resumes from the last checkpoint if one is found.
6.  Specialization Score Calculation:
    - Adds new function `calculate_specialization_score`.
    - This metric is the *average standard deviation of expert
      utilization across topics*.
    - A high score = high specialization (experts focus on
      specific topics).
    - A low score = generalist experts (experts are used
      equally for all topics).
    - This is calculated for the first and last layers of
      each MoE model.
7.  Specialization Comparison Plot:
    - Adds new function `plot_specialization_results`.
    - Generates two new plots:
      - `ablation_specialization_comparison_first_layer.png`
      - `ablation_specialization_comparison_last_layer.png`
    - These plots compare the specialization score for all
      MoE models, with error bars and significance
      (t-test) relative to the 'softmax-moe' baseline.

===================================================================

Usage:
1.  Set your configuration in the `if __name__ == "__main__":` block.
    (e.g., `NUM_SEEDS`, `DEBUG`, model configs).
2.  Run: `python run_local_research_LIGHTWEIGHT.py`
3.  View logs: `tensorboard --logdir=./logs`
4.  View final plots:
    - `./ablation_plots/ablation_accuracy_comparison.png`
    - `./ablation_plots/ablation_specialization_comparison_first_layer.png`
    - `./ablation_plots/ablation_specialization_comparison_last_layer.png`
"""

# --- 0. Setup & Imports ---
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Optional, Tuple
from tqdm import tqdm
import shutil
import evaluate
import json
import gc  # ⬅️ NEW IMPORT for memory clearing
import copy  # ⬅️ NEW IMPORT for config deepcopy
import pandas as pd  # ⬅️ NEW IMPORT for plotting
from scipy import stats  # ⬅️ NEW IMPORT for plotting

# --- Hugging Face Imports ---
from transformers import (
    BertConfig,
    BertPreTrainedModel,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    TrainerCallback,
    DataCollatorWithPadding,
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertPooler,
    BertAttention,
)
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import (
    get_last_checkpoint,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn import LayerNorm, CrossEntropyLoss


# Set to True for a fast pipeline check (few samples, 1 epoch, fast analysis)
# Set to False for a full training run
DEBUG = False # ⬅️ Set to False for your 94.1% model example


print("--- GPU Check ---")
print(f"Is CUDA available?    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count:        {torch.cuda.device_count()}")
    print(f"Current device:    {torch.cuda.current_device()}")
    print(f"Device name:         {torch.cuda.get_device_name(0)}")
print("---------------------")
# --- 1. Modularization: Custom MoE Config ---
# =================================================================
class MoEBertConfig(BertConfig):
    """
    Custom config class for MoE-BERT. Inherits from BertConfig and adds
    MoE-specific hyperparameters.
    """

    model_type = "moe-bert"

    def __init__(
        self,
        num_experts=8,
        alpha_balance=0.01,
        energy_loss_alpha=0.0,
        router_type="softmax",  # "softmax", "astrocyte", "astrocyte-hierarchical", "astrocyte-meanpool"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.alpha_balance = alpha_balance
        self.energy_loss_alpha = energy_loss_alpha
        self.router_type = router_type
        # Ensure this class is recognized by AutoConfig
        self.model_type = "moe-bert"


# --- 2. Modularization: Custom MoE Components ---
# =================================================================
class Expert(nn.Module):
    """A simple feed-forward expert network."""

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        h = self.dense1(hidden_states)
        h = self.intermediate_act_fn(h)
        h = self.dense2(h)
        h = self.dropout(h)
        return h


class MoELayer(nn.Module):
    """
    The core MoE layer (Phase 1) with Astrocyte routing (Phase 3).

    Supports:
    - 'softmax': Standard token-level softmax routing.
    - 'astrocyte': Token-level routing modulated by [CLS] token context (multiplicative).
    - 'astrocyte-hierarchical': (Strategy 3A) Token-level routing biased by [CLS] token context (additive).
    - 'astrocyte-meanpool': (Strategy 3B) Token-level routing modulated by mean-pooled token context.
    """

    def __init__(self, config: MoEBertConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = 2  #  Set K=2

        # Standard token-level router
        self.router = nn.Linear(config.hidden_size, self.num_experts)

        #  Astrocyte Modulator (for [CLS] token)
        if self.config.router_type == "astrocyte":
            self.modulator = nn.Sequential(
                nn.Linear(config.hidden_size, self.num_experts), nn.Sigmoid()
            )
        # (Strategy 3A)
        elif self.config.router_type == "astrocyte-hierarchical":
            # Just a linear layer, no sigmoid. Output will be added as logits.
            self.modulator = nn.Linear(config.hidden_size, self.num_experts)
        # (Strategy 3B)
        elif self.config.router_type == "astrocyte-meanpool":
            # Same structure as 'astrocyte', but will be fed different input
            self.modulator = nn.Sequential(
                nn.Linear(config.hidden_size, self.num_experts), nn.Sigmoid()
            )

        # List of experts
        self.experts = nn.ModuleList(
            [Expert(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_size)

        # --- 1. Get Router Logits ---
        token_router_logits = self.router(hidden_states)  # (B, S, E)

        # Astrocyte Gating Logic (Phase 3)
        if self.config.router_type == "astrocyte":
            cls_embedding = hidden_states[:, 0, :]  # (B, H)
            modulation_weights = self.modulator(cls_embedding)  # (B, E)
            modulation_weights = modulation_weights.unsqueeze(1)  # (B, 1, E)
            router_logits = token_router_logits * modulation_weights  # (B, S, E)

        # (Strategy 3A)
        elif self.config.router_type == "astrocyte-hierarchical":
            cls_embedding = hidden_states[:, 0, :] # (B, H)
            global_logits = self.modulator(cls_embedding) # (B, E)
            global_logits = global_logits.unsqueeze(1) # (B, 1, E)
            # ADD the global bias to the token logits
            router_logits = token_router_logits + global_logits # (B, S, E)

        # (Strategy 3B)
        elif self.config.router_type == "astrocyte-meanpool":
            # Get all token embeddings, ignoring [CLS]
            token_embeddings = hidden_states[:, 1:, :] # (B, S-1, H)
            # Calculate the mean across the sequence dimension
            global_context = torch.mean(token_embeddings, dim=1) # (B, H)
            modulation_weights = self.modulator(global_context) # (B, E)
            modulation_weights = modulation_weights.unsqueeze(1) # (B, 1, E)
            router_logits = token_router_logits * modulation_weights # (B, S, E)

        else:
            # This covers "softmax" router_type
            router_logits = token_router_logits

        # --- 2. Get Top-K Gating Weights ---

        # 2a. Find the top k logits and their indices
        # We take top_k logits for stable softmax
        top_k_logits, top_k_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # (B, S, 2)

        # 2b. Apply softmax to just the top-k logits
        top_k_weights = F.softmax(top_k_logits, dim=-1) # (B, S, 2)

        # 2c. Create a full-size weight tensor (mostly zeros)
        # This is (B, S, 8)
        gating_weights = torch.zeros_like(router_logits)

        # 2d. "Scatter" the top-k weights into the zero tensor
        # This operation puts the top_k_weights into the
        # correct expert "slots" (indices) in the (B, S, 8) tensor
        gating_weights.scatter_(
           dim=-1, index=top_k_indices, src=top_k_weights.to(gating_weights.dtype)
        )
        # --- 3. Calculate Auxiliary Losses ---
        # (This section is identical to before)

        # Loss 1: Load Balancing Loss (CV Loss)
        # Note: This loss now correctly encourages balancing
        # the *chosen* Top-K experts.
        importance = gating_weights.sum(dim=(0, 1))  # (E)
        mean_importance = importance.mean()
        std_importance = importance.std()
        # This is the (CV)^2 loss
        load_balancing_loss = self.config.alpha_balance * (
            std_importance**2 / (mean_importance**2 + 1e-8)
        )

        # Loss 2: Energy Loss (L2 of mean probabilities)
        token_probs_mean_over_batch_seq = gating_weights.mean(dim=(0, 1))  # (E)
        param_weighted_loss_raw = torch.sum(token_probs_mean_over_batch_seq**2)
        energy_loss = self.config.energy_loss_alpha * param_weighted_loss_raw

        # Combine them into a single auxiliary loss
        aux_loss = load_balancing_loss + energy_loss

        # --- 4. Combine Expert Outputs ---
        # (This section is identical to before)
        # This is still computationally dense, but mathematically sparse.
        # It's the simplest way to get sparse specialization.
        expert_outputs = torch.stack(
            [expert(hidden_states) for expert in self.experts], dim=2
        )  # (B, S, E, H)

        gating_weights_expanded = gating_weights.unsqueeze(2)  # (B, S, 1, E)

        # This matmul now multiplies 6 experts by 0,
        # effectively selecting only the Top-K.
        weighted_output_bmm = torch.matmul(
            gating_weights_expanded, expert_outputs
        )  # (B, S, 1, H)
        final_output = weighted_output_bmm.squeeze(2)  # (B, S, H)

        return final_output, aux_loss, gating_weights


class MoEBertLayer(nn.Module):
    """A custom BertLayer that replaces the standard FFN with our MoELayer."""

    def __init__(self, config: MoEBertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.layer_norm_after_attention = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.moe_layer = MoELayer(config)
        self.layer_norm_after_moe = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_router_weights=False,
    ):
        # 1. Self-Attention
        attention_outputs = self.attention(
            hidden_states, attention_mask, output_attentions=output_attentions
        )
        self_attention_output = attention_outputs[0]

        # 2. Add & Norm
        hidden_states_attn = self.layer_norm_after_attention(
            self_attention_output + hidden_states
        )

        # 3. MoE Layer
        moe_outputs = self.moe_layer(hidden_states_attn)
        moe_output, aux_loss, router_weights = (  # aux_loss is now the combined loss
            moe_outputs[0],
            moe_outputs[1],
            moe_outputs[2],
        )

        # 4. Add & Norm
        hidden_states_moe = self.layer_norm_after_moe(moe_output + hidden_states_attn)

        # 5. Package outputs
        outputs = (hidden_states_moe, aux_loss)  # Pass the combined aux_loss up
        if output_attentions:
            outputs += (attention_outputs[1],)
        if output_router_weights:
            outputs += (router_weights,)

        return outputs


class MoEBertEncoder(nn.Module):
    """Custom BertEncoder that stacks our MoEBertLayer."""

    def __init__(self, config: MoEBertConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [MoEBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        output_router_weights=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_weights = () if output_router_weights else None

        # This var now holds the sum of *all* auxiliary losses
        total_aux_loss = 0.0

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions,
                output_router_weights,
            )

            hidden_states = layer_outputs[0]
            total_aux_loss += layer_outputs[1]  # Add the (combined) aux_loss

            if output_attentions:
                all_attentions += (layer_outputs[2],)
            if output_router_weights:
                all_router_weights += (layer_outputs[-1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        avg_aux_loss = total_aux_loss / len(self.layer)

        return (
            hidden_states,
            avg_aux_loss,  # Return the averaged combined aux_loss
            all_hidden_states,
            all_attentions,
            all_router_weights,
        )


# --- 3. Modularization: Main MoE Model (Phase 1) ---
# =================================================================
class MoEBertForSequenceClassification(BertPreTrainedModel):
    """
    The main MoE-BERT model for sequence classification.
    """

    config_class = MoEBertConfig  # Links to our custom config

    def __init__(self, config: MoEBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = MoEBertEncoder(config)  # Use our custom encoder
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_weights=None,  # Our custom flag
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )

        # 🚀 Pass through MoE Encoder
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_weights=output_router_weights,
        )

        sequence_output = encoder_outputs[0]
        # This is now the combined (avg_lb_loss + avg_energy_loss)
        avg_aux_loss = encoder_outputs[1]

        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # 🚀 Calculate Losses (Phase 2)
        total_loss = None
        classification_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            classification_loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1)
            )
            total_loss = classification_loss + avg_aux_loss

        # Return a dictionary. Trainer will auto-log all scalar values.
        output_dict = {
            "logits": logits,
        }
        if total_loss is not None:
            output_dict["loss"] = total_loss  # This is the total_loss
            output_dict["classification_loss"] = classification_loss
            output_dict["aux_loss"] = avg_aux_loss

        if output_hidden_states:
            output_dict["hidden_states"] = encoder_outputs[2]
        if output_attentions:
            output_dict["attentions"] = encoder_outputs[3]
        if output_router_weights:
            output_dict["all_router_weights"] = encoder_outputs[4]

        return output_dict


# Register the custom config and model
AutoConfig.register("moe-bert", MoEBertConfig)
AutoModelForSequenceClassification.register(
    MoEBertConfig, MoEBertForSequenceClassification
)


# --- 4. TensorBoard Integration (Phase 2) ---
# =================================================================
class MoETensorBoardCallback(TensorBoardCallback):
    """
    Custom TensorBoard callback to log expert utilization histograms.
    This is already lightweight as it only uses one batch.
    """

    def __init__(self, *args, **kwargs):
        # Pop our custom kwarg before passing to parent
        self.eval_dataset = kwargs.pop("eval_dataset", None)
        super().__init__(*args, **kwargs)
        self.eval_dataloader = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # We need access to the eval_dataset to create a dataloader
        if self.eval_dataset is not None:
            # We create this dataloader *without* the 'text' column,
            # which was removed in run_training().
            self.eval_dataloader = DataLoader(
                self.eval_dataset.with_format("torch"),
                batch_size=args.per_device_eval_batch_size,
            )
        super().on_train_begin(args, state, control, model=model, **kwargs)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # First, let the default callback log all the scalar metrics
        super().on_evaluate(args, state, control, model=model, **kwargs)

        # Now, add our custom histogram
        if self.tb_writer is None:
            return  # TensorBoard writer not initialized

        print("Logging expert utilization histogram to TensorBoard...")

        if self.eval_dataloader is None:
            print("Warning: No eval_dataloader found, skipping histogram.")
            return

        try:
            batch = next(iter(self.eval_dataloader))
        except StopIteration:
            print("Warning: Could not get a batch for expert histogram.")
            return

        # Move batch to device
        batch = {
            k: v.to(args.device)
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "token_type_ids", "labels"]
        }

        model.eval()
        with torch.no_grad():
            outputs = model(**batch, output_router_weights=True)

        # Get weights from the last layer
        last_layer_weights = outputs["all_router_weights"][-1].cpu()  # (B, S, E)

        # Get the top-1 expert chosen for each token
        top1_expert = torch.argmax(last_layer_weights, dim=-1)  # (B, S)

        # Flatten to get a 1D array of expert choices
        expert_choices = top1_expert.view(-1).numpy()

        # Log as a histogram to TensorBoard
        self.tb_writer.add_histogram(
            "eval/expert_utilization",
            expert_choices,
            global_step=state.global_step,
            bins=model.config.num_experts,
        )


# --- 5. Main Training Function ---
# =================================================================
@dataclass
class ScriptArguments:
    """Arguments for the training script."""

    model_name_or_path: str = field(
        default="google/bert_uncased_L-4_H-512_A-8",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    dataset_name: str = field(
        default="ag_news", metadata={"help": "The name of the dataset to use."}
    )
    is_moe: bool = field(
        default=True, metadata={"help": "Whether to use MoE layers or dense FFNs."}
    )  
    num_experts: int = field(
        default=8, metadata={"help": "Number of experts in MoE layers."}
    )
    alpha_balance: float = field(
        default=0.01, metadata={"help": "Weight for the load balancing (CV) loss."}
    )
    energy_loss_alpha: float = field(
        default=0.0, metadata={"help": "Weight for the energy (L2 prob) loss."}
    )
    router_type: str = field(
        default="softmax",
        metadata={"help": "Type of router to use: 'softmax' or 'astrocyte'."},
    )
    max_seq_length: int = field(
        default=128, metadata={"help": "Maximum sequence length."}
    )


def run_training(
    script_args: ScriptArguments,
    training_args: TrainingArguments,
    seed: Optional[int] = None,
):  # seed argument
    """
    Main training and evaluation function.
    Returns: (model_path, output_dir, eval_results, model_base_name)
    """
    if script_args.is_moe:
        print(f"🚀 Initializing {script_args.router_type.upper()} MoE-BERT Training...")
    else:
        print(f"🚀 Initializing DENSE Baseline Training...")
    print(f"Using base model: {script_args.model_name_or_path}")

    # 1. Setup logging
    # run_name logic for ablations
    if script_args.is_moe:
        model_base_name = (
            f"{script_args.router_type}-E{script_args.num_experts}"
            f"-LB{script_args.alpha_balance}-EN{script_args.energy_loss_alpha}"
        )
    else:
        # This will be "dense" or "dense-random-ffn"
        model_base_name = config_name # Relies on config_name from __main__

    run_name = model_base_name
    if DEBUG:
        print(" RUNNING IN DEBUG MODE ")
        print("    (Using 200 train/100 eval samples, 1 epoch)")
        training_args.num_train_epochs = 1
        training_args.logging_steps = 1
        training_args.eval_steps = 10
        training_args.save_strategy = "no"  # Don't save checkpoints
        training_args.load_best_model_at_end = False
        run_name = f"DEBUG-{run_name}"

    # ⬅️ NEW: Add seed to run_name for unique directories
    if seed is not None:
        run_name = f"{run_name}-seed{seed}"

    # Point logging_dir and output_dir to a subdirectory for this specific run
    training_args.logging_dir = os.path.join(training_args.logging_dir, run_name)
    training_args.output_dir = os.path.join(training_args.output_dir, run_name)

    # 3. NEW CHECKPOINT LOGIC (START)
    # Check if a *final model* already exists for this config.
    final_model_path = os.path.join(training_args.output_dir, "final-model")
    if os.path.exists(os.path.join(final_model_path, "pytorch_model.bin")):
        print(f" Found fully trained 'final-model' at: {final_model_path}")
        print("Skipping training. Moving directly to analysis.")
        # ⬅️ NEW: Load final eval results if they exist
        eval_results_path = os.path.join(final_model_path, "eval_results.json")
        if os.path.exists(eval_results_path):
            with open(eval_results_path, "r") as f:
                eval_results = json.load(f)
            print(f"Loaded existing eval results: {eval_results['eval_accuracy']:.4f}")
        else:
            print("Warning: No eval_results.json found. Returning empty dict.")
            eval_results = {}
        return final_model_path, training_args.output_dir, eval_results, model_base_name

    print(f"No 'final-model' found. Will train/resume in: {training_args.output_dir}")
    # ⬆️ 3. NEW CHECKPOINT LOGIC (END)

    # 2. Load Tokenizer & Dataset
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    dataset = load_dataset(script_args.dataset_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=script_args.max_seq_length,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4)
    num_labels = dataset["train"].features["label"].num_classes

    if DEBUG:
        train_dataset = (
            tokenized_datasets["train"].shuffle(seed=training_args.seed).select(range(200))
        )
        eval_dataset = tokenized_datasets["test"].shuffle(seed=training_args.seed).select(range(100))
    else:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["test"]

    # 3. Load Model (MoE or Dense)
    # ⬅️ NEW: Conditional model loading
    if script_args.is_moe:
        print(f" Initializing MoE model with WARM START...")

        # 1. Load the DENSE baseline model's config and weights first
        print("    Loading pre-trained dense weights for initialization...")
        dense_config = AutoConfig.from_pretrained(
            script_args.model_name_or_path,
            num_labels=num_labels,
        )
        dense_model_for_weights = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name_or_path,
            config=dense_config,
        )

        # 2. Now, create the MoE config
        print(f"    Instantiating MoE structure with router: {script_args.router_type}")
        moe_config = MoEBertConfig.from_pretrained(
            script_args.model_name_or_path,
            num_labels=num_labels,
            num_experts=script_args.num_experts,
            alpha_balance=script_args.alpha_balance,
            energy_loss_alpha=script_args.energy_loss_alpha,
            router_type=script_args.router_type,
        )

        # 3. Instantiate the MoE model structure FROM THE CONFIG
        # This creates the model with random weights
        model = MoEBertForSequenceClassification(config=moe_config)

        # ---
        # --- START OF CORRECTION  ---
        # ---
        # 4. Manually copy all matching weights from the dense model.
        # This is required because the MoE model has a different
        # structure (e.g., no 'bert' wrapper) and different
        # LayerNorm names within the BertLayer.
        print("      Manually copying weights from dense model...")

        # 4a. Copy Embeddings
        model.embeddings.load_state_dict(dense_model_for_weights.bert.embeddings.state_dict())
        print("      Copied Embeddings.")

        # 4b. Copy Pooler
        model.pooler.load_state_dict(dense_model_for_weights.bert.pooler.state_dict())
        print("      Copied Pooler.")

        # 4c. Copy Classifier
        model.classifier.load_state_dict(dense_model_for_weights.classifier.state_dict())
        print("      Copied Classifier.")

        # 4d. Loop through layers to copy Attention, LayerNorms, and FFNs-to-Experts
        print(f"      Copying layer weights (Attention, LayerNorms, and FFNs-to-Experts)...")
        noise_std = 1e-4  # Small noise to break symmetry

        for layer_idx in range(moe_config.num_hidden_layers):
            dense_layer = dense_model_for_weights.bert.encoder.layer[layer_idx]
            moe_layer = model.encoder.layer[layer_idx]

            # --- Copy Attention ---
            # MoEBertLayer uses BertAttention directly, so this is a 1:1 copy
            moe_layer.attention.load_state_dict(dense_layer.attention.state_dict())

            # --- Copy LayerNorms (Handles Name Mismatch) ---
            # MoE LayerNorm 1 ('layer_norm_after_attention') maps to
            # Dense LayerNorm 1 ('attention.output.LayerNorm')
            moe_layer.layer_norm_after_attention.load_state_dict(
                dense_layer.attention.output.LayerNorm.state_dict()
            )

            # MoE LayerNorm 2 ('layer_norm_after_moe') maps to
            # Dense LayerNorm 2 ('output.LayerNorm')
            moe_layer.layer_norm_after_moe.load_state_dict(
                dense_layer.output.LayerNorm.state_dict()
            )

            # --- Copy FFNs into Experts (Original Logic) ---
            # Get the pre-trained FFN weights from the dense model
            dense_ffn_w1 = dense_layer.intermediate.dense.weight
            dense_ffn_b1 = dense_layer.intermediate.dense.bias
            dense_ffn_w2 = dense_layer.output.dense.weight
            dense_ffn_b2 = dense_layer.output.dense.bias

            # Get the experts in our MoE model's layer
            moe_experts = moe_layer.moe_layer.experts

            for expert in moe_experts:
                # Copy weights + add small random noise to break symmetry
                expert.dense1.weight.data = dense_ffn_w1.clone() + (torch.randn_like(dense_ffn_w1) * noise_std)
                expert.dense1.bias.data = dense_ffn_b1.clone() + (torch.randn_like(dense_ffn_b1) * noise_std)
                expert.dense2.weight.data = dense_ffn_w2.clone() + (torch.randn_like(dense_ffn_w2) * noise_std)
                expert.dense2.bias.data = dense_ffn_b2.clone() + (torch.randn_like(dense_ffn_b2) * noise_std)
        # ---
        # ---  END OF CORRECTION  ---
        # ---

        print("      All experts initialized from pre-trained weights.")
        del dense_model_for_weights # Free up VRAM
        gc.collect()
        torch.cuda.empty_cache()

    else:
        print("Instantiating DENSE baseline model.")
        config = AutoConfig.from_pretrained(
            script_args.model_name_or_path,
            num_labels=num_labels,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name_or_path,
            config=config,
        )

        #  NEW BLOCK: Re-initialize FFNs for fair "cold start" test
        if model_base_name == "dense-random-ffn":
            print("🔥 Re-initializing FFN weights for 'dense-random-ffn' model...")
            for layer in model.bert.encoder.layer:
                # A standard BERT FFN is 'intermediate' and 'output.dense'
                layer.intermediate.apply(model._init_weights)
                layer.output.dense.apply(model._init_weights)
            print(" FFN weights have been randomly re-initialized.")
        #  END NEW BLOCK

    # Print model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params/1_000_000:.2f}M")

    # 4. Define Metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        #  MODIFIED: Handle MoE tuple output vs Dense tensor output
        if isinstance(logits, tuple):
            logits_to_use = logits[0]
        else:
            logits_to_use = logits
        predictions = np.argmax(logits_to_use, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Manually remove unused 'text' column
    print("Manually removing unused 'text' column...")
    train_dataset = train_dataset.remove_columns(["text"])
    eval_dataset = eval_dataset.remove_columns(["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # REMOVE default TensorBoard callback
    trainer.remove_callback(TensorBoardCallback)
    # Conditionally add MoE or standard callback
    if script_args.is_moe:
        trainer.add_callback(
            MoETensorBoardCallback(eval_dataset=eval_dataset)
        )
    else:
        trainer.add_callback(TensorBoardCallback())  # Add standard callback

    # 6. Train!
    last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if last_checkpoint:
        print(f"Found checkpoint. Resuming training from: {last_checkpoint}")
    else:
        print("No checkpoint found. Starting fresh training run.")

    print("---  Starting/Resuming Training  ---")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    print("---  Training Complete ---")

    # 7. Final Evaluation
    print("---  Running Final Evaluation ---")
    eval_results = trainer.evaluate()
    print(eval_results)

    # 8. Save Model & Final Results
    model_save_path = os.path.join(training_args.output_dir, "final-model")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Save final eval results for plotting
    eval_results_path = os.path.join(model_save_path, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f" Model and eval results saved to {model_save_path}")
    print(f" View logs with: tensorboard --logdir=./logs")

    # Return eval_results and model_base_name
    return model_save_path, training_args.output_dir, eval_results, model_base_name


# --- 6. Post-Training Analysis Functions (Phase 2+) ---
# =================================================================
# (No changes needed in these sections)

# ANSI Colors for qualitative analysis
COLORS = [
    "\033[94m",
    "\033[92m",
    "\033[93m",
    "\033[91m",
    "\033[95m",
    "\033[96m",
    "\033[34m",
    "\033[32m",
]
RESET = "\033[0m"


def run_qualitative_analysis(model_path):
    """
    Loads the trained model and visualizes token-to-expert routing
    for example sentences. This is already lightweight.
    """
    print("\n" + "=" * 80)
    print("🔬 RUNNING: Qualitative Token-Routing Analysis (Phase 2)")
    print("=" * 80)

    try:
        # Must use our custom class to load
        model = MoEBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model for analysis: {e}")
        print("This may be expected if loading a 'dense' model.")
        print("Skipping qualitative analysis.")
        return

    if not hasattr(model, 'config') or not hasattr(model.config, 'model_type') or model.config.model_type != 'moe-bert':
        print("Model is not a 'moe-bert' model. Skipping qualitative analysis.")
        return

    sentences = [
        "Rockets beat Lakers in playoff thriller.",  # Sports
        "Fed hints at new interest rate hike.",  # Business
        "Global leaders meet to discuss climate change.",  # World
        "New breakthrough in quantum computing announced.",  # Sci/Tech
    ]

    for sentence in sentences:
        print(f"\nAnalyzing: \"{sentence}\"")

        inputs = tokenizer(sentence, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_router_weights=True)

        # Analyze weights from the *last* layer
        router_weights = outputs["all_router_weights"][-1].squeeze(0)  # (SeqLen, NumExperts)
        top1_expert_indices = torch.argmax(router_weights, dim=-1)  # (SeqLen)

        output_str = ""
        for token, expert_id in zip(tokens, top1_expert_indices):
            if token in [tokenizer.cls_token, tokenizer.sep_token]:
                output_str += f"{token} "
                continue

            color = COLORS[expert_id.item() % len(COLORS)]
            output_str += f"{color}{token.replace('##', '')}{RESET} "

        print(output_str)

    print("\n--- Expert Legend ---")
    for i in range(model.config.num_experts):
        print(f"{COLORS[i % len(COLORS)]}Expert {i}{RESET}", end=" | ")
    print("\n" + "=" * 80)


# Modified function signature to accept a 'desc' for tqdm
def get_topic_utilization_for_layer(model, tokenizer, dataset, device, layer_idx, desc="Analyzing Layer"):
    """Helper function to calculate expert utilization for a specific layer."""
    model.to(device)
    model.eval()

    # Use a larger batch size for analysis/inference
    dataloader = DataLoader(dataset, batch_size=128)

    # Store sum of weights and total number of tokens
    total_weights_per_expert = torch.zeros(model.config.num_experts).to(device)
    total_tokens = 0

    with torch.no_grad():
        # Updated tqdm description
        for batch in tqdm(dataloader, desc=f"{desc} (Layer {layer_idx})", leave=False):
            # NOTE: This analysis dataloader is separate from the trainer
            # and still has the 'text' column, so we tokenize on the fly.
            inputs = tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            attention_mask = inputs["attention_mask"]

            outputs = model(**inputs, output_router_weights=True)

            # Get weights for the specified layer: (B, S, E)
            router_weights = outputs["all_router_weights"][layer_idx]

            # Mask out padding tokens
            # (B, S, E) * (B, S, 1)
            router_weights_masked = router_weights * attention_mask.unsqueeze(-1)

            # Sum weights for all active tokens in the batch
            total_weights_per_expert += router_weights_masked.sum(dim=(0, 1))
            total_tokens += attention_mask.sum()

    # Return average utilization (sum of weights / total tokens)
    return (total_weights_per_expert / total_tokens).cpu().numpy()


def run_quantitative_analysis(model_path, plot_dir=".", num_samples=2000):
    """
    Runs topic-based routing analysis for the *last layer*
    and saves a heatmap.
    """
    print("\n" + "=" * 80)
    print(f"RUNNING: Quantitative Topic-Based Routing (on {num_samples} samples)")
    print("=" * 80)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = MoEBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model for analysis: {e}")
        print("This may be expected if loading a 'dense' model.")
        print("Skipping quantitative analysis.")
        return

    if not hasattr(model, 'config') or not hasattr(model.config, 'model_type') or model.config.model_type != 'moe-bert':
        print("Model is not a 'moe-bert' model. Skipping quantitative analysis.")
        return

    #  LIGHTWEIGHT MOD: Shuffle and select N samples
    dataset = (
        load_dataset("ag_news", split="test")
        .shuffle(seed=42)
        .select(range(num_samples))
    )

    labels = dataset.features["label"].names
    category_utilization = {}

    last_layer_idx = model.config.num_hidden_layers - 1

    for i, label_name in enumerate(labels):
        print(f"\n--- Analyzing Category: {label_name} (Label {i}) ---")
        category_dataset = dataset.filter(
            lambda example: example["label"] == i, num_proc=4
        )
        print(f"(Analyzing {len(category_dataset)} samples for this category)")

        #  Pass the 'desc' argument
        utilization = get_topic_utilization_for_layer(
            model, tokenizer, category_dataset, DEVICE, last_layer_idx,
            desc=f"Topic: {label_name}"
        )
        category_utilization[label_name] = utilization
        print(f"Avg Utilization: {np.round(utilization * 100, 2)}")

    # Plot the heatmap
    num_experts = model.config.num_experts
    data = np.array([category_utilization[label] for label in labels]) * 100

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=[f"Expert {i}" for i in range(num_experts)],
        yticklabels=labels,
    )
    plt.title(
        f"Average Expert Utilization (%) by Topic - {model.config.router_type.title()} (Last Layer)",
        fontsize=16,
    )
    plt.xlabel("Expert")
    plt.ylabel("News Category")
    plt.tight_layout()

    filename = f"{plot_dir}/topic_utilization_last_layer.png"
    plt.savefig(filename)
    print(f"\n Heatmap saved to {filename}")
    plt.close() # Close plot to free memory


def run_layer_depth_analysis(model_path, plot_dir=".", num_samples=2000):
    """
    DEEPER ANALYSIS: Runs topic-based analysis for *all layers*
    and saves a grid of heatmaps.
    """
    print("\n" + "=" * 80)
    print(
        f"RUNNING: Deep Analysis - Expert Specialization vs. Layer Depth (on {num_samples} samples)"
    )
    print("=" * 80)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = MoEBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model for analysis: {e}")
        print("This may be expected if loading a 'dense' model.")
        print("Skipping deep analysis.")
        return

    if not hasattr(model, 'config') or not hasattr(model.config, 'model_type') or model.config.model_type != 'moe-bert':
        print("Model is not a 'moe-bert' model. Skipping deep analysis.")
        return

    # ⚡ LIGHTWEIGHT MOD: Shuffle and select N samples
    dataset = (
        load_dataset("ag_news", split="test")
        .shuffle(seed=42)
        .select(range(num_samples))
    )

    labels = dataset.features["label"].names
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_experts

    # Setup plot grid
    ncols = 4
    if num_layers <= 4:
        ncols = num_layers

    nrows = int(np.ceil(num_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for layer_idx in range(num_layers):
        print(f"\n--- Analyzing All Topics for Layer {layer_idx} ---")
        category_utilization = {}

        for i, label_name in enumerate(labels):
            category_dataset = dataset.filter(
                lambda example: example["label"] == i, num_proc=4
            )
            # Pass the 'desc' argument
            utilization = get_topic_utilization_for_layer(
                model, tokenizer, category_dataset, DEVICE, layer_idx,
                desc=f"L{layer_idx} - Topic: {label_name}"
            )
            category_utilization[label_name] = utilization

        # Plot this layer's heatmap
        data = np.array([category_utilization[label] for label in labels]) * 100
        ax = axes[layer_idx]
        sns.heatmap(
            data,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            cbar=False,
            xticklabels=[f"E{i}" for i in range(num_experts)],
            yticklabels=labels if layer_idx % ncols == 0 else False,
        )
        ax.set_title(f"Layer {layer_idx}")

    # Hide any unused subplots
    for i in range(num_layers, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"Expert Utilization (%) by Topic and Layer - {model.config.router_type.title()}",
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"{plot_dir}/topic_utilization_all_layers.png"
    plt.savefig(filename)
    print(f"\n Layer-depth analysis plot saved to {filename}")
    plt.close() # Close plot to free memory


# --- 7. NEW: Specialization Score Calculation Function ---
# =================================================================
def calculate_specialization_score(model_path, num_samples=2000, layer_idx=-1):
    """
    Calculates a specialization score for a given model layer.

    The score is the average standard deviation of expert utilization
    across different topics.
    A higher score means more specialization.
    Returns: Float score or None if model is not MoE.
    """
    print(f"\n--- 🔬 Calculating Specialization Score (Layer {layer_idx}) ---")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = MoEBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        # This is expected for dense models
        print("Model not found or not MoE. Skipping specialization score.")
        return None

    if not hasattr(model, 'config') or not hasattr(model.config, 'model_type') or model.config.model_type != 'moe-bert':
        print("Model is not 'moe-bert'. Skipping specialization score.")
        return None

    # Resolve layer_idx (-1 means last layer)
    if layer_idx == -1:
        layer_idx = model.config.num_hidden_layers - 1

    dataset = (
        load_dataset("ag_news", split="test")
        .shuffle(seed=42)
        .select(range(num_samples))
    )
    labels = dataset.features["label"].names
    category_utilization = {}

    for i, label_name in enumerate(labels):
        category_dataset = dataset.filter(
            lambda example: example["label"] == i, num_proc=4
        )
        # Use the existing helper function, passing a new description
        utilization = get_topic_utilization_for_layer(
            model, tokenizer, category_dataset, DEVICE, layer_idx,
            desc=f"Spec. L{layer_idx} - Topic: {label_name}"
        )
        category_utilization[label_name] = utilization

    # data shape: (num_topics, num_experts)
    data = np.array([category_utilization[label] for label in labels])

    # Transpose to (num_experts, num_topics)
    data_per_expert = data.T

    # Calculate std dev for each expert across topics
    # Shape: (num_experts,)
    expert_specializations = np.std(data_per_expert, axis=1)

    # The final score is the average of these std devs
    mean_specialization_score = np.mean(expert_specializations)

    print(f"--- Specialization Score (Layer {layer_idx}): {mean_specialization_score:.6f} ---")

    # Clean up memory
    del model, tokenizer, dataset, category_utilization, data, data_per_expert
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mean_specialization_score


# --- 8. Ablation Comparison Plotting Functions ---
# =================================================================

def get_significance_stars(p_value):
    """Returns significance stars for a given p-value."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"  # not significant


def plot_ablation_results(all_run_results, plot_save_dir):
    """
    Plots a bar chart comparing all model ablations with error bars
    and statistical significance relative to the 'dense' baseline.
    """
    print("\n" + "=" * 80)
    print("RUNNING: Ablation Comparison Plotting (Accuracy)")
    print("=" * 80)

    if not all_run_results:
        print("No results to plot. Skipping.")
        return

    # --- 1. Save Raw Results ---
    results_path = os.path.join(plot_save_dir, "ablation_all_results.json")
    try:
        with open(results_path, "w") as f:
            json.dump(all_run_results, f, indent=2)
        print(f"Saved all raw results to {results_path}")
    except Exception as e:
        print(f"Error saving raw results: {e}")

    # --- 2. Process Data with Pandas ---
    df = pd.DataFrame(all_run_results)
    if "eval_accuracy" not in df.columns:
        print(f"Error: 'eval_accuracy' not found in results. Columns are: {df.columns}")
        return

    num_seeds = df["seed"].nunique()

    # Use 'model_base_name' for grouping
    grouped = df.groupby("model_base_name")

    # Calculate mean and sem (Standard Error of Mean)
    summary = grouped["eval_accuracy"].agg(["mean", "sem"]).reset_index()
    summary = summary.sort_values(by="mean", ascending=False)

    # Handle cases with 1 seed (sem will be NaN)
    summary["sem"] = summary["sem"].fillna(0) # Replace NaN with 0 if N=1 seed

    plt.figure(figsize=(14, 8)) # Made figure wider

    # --- 3. Create Bar Plot ---
    barplot = sns.barplot(
        x="mean", y="model_base_name", data=summary, orient="h", palette="viridis"
    )

    plt.title(f"Model Accuracy Comparison (N={num_seeds} seed{'s' if num_seeds != 1 else ''})", fontsize=16)
    plt.xlabel("Mean Evaluation Accuracy", fontsize=12)
    plt.ylabel("Model Configuration", fontsize=12)

    # --- 4. Dynamic Axis Limits (FIX for DEBUG) ---
    min_val = summary["mean"].min()
    max_val = summary["mean"].max()
    max_sem = summary["sem"].max()

    # Determine plot range
    # Start at 0 or just below the lowest bar's error range
    plot_min = max(0, min_val - max_sem - 0.02)
    # This is the coordinate for the *end* of the longest error bar
    plot_max_data = max_val + max_sem

    plt.xlim(left=plot_min) # Set left limit now (right limit set later)

    # --- 5. Add Error Bars & Mean Value Text ---
    for i, (idx, row) in enumerate(summary.iterrows()):
        # Add Error Bar
        plt.errorbar(
            x=row["mean"],
            y=i,
            xerr=row["sem"],
            fmt="none",
            c="black",
            capsize=5,
            label="Standard Error" if i == 0 else None
        )
        # Add mean value as text on/near the bar
        plt.text(
            row["mean"] + 0.0005, # Just to the right of the mean
            i,
            f"{row['mean']:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            color="black",
            fontweight="bold"
        )

    # --- 6.Add Statistical Significance (Cleaner) ---
    p_value_texts = [] # Store (y_pos, text_string)
    has_stats = False

    if "dense" in summary["model_base_name"].values:
        baseline_scores = df[df["model_base_name"] == "dense"]["eval_accuracy"].dropna()

        if len(baseline_scores) < 2:
            print("Not enough 'dense' baseline runs (N<2) to perform t-test. Skipping.")
        else:
            has_stats = True
            for i, (idx, row) in enumerate(summary.iterrows()):
                model_name = row["model_base_name"]
                text = "" # Default empty text
                if model_name != "dense":
                    model_scores = df[df["model_base_name"] == model_name]["eval_accuracy"].dropna()

                    if len(model_scores) < 2:
                        text = "(N<2)"
                    else:
                        try:
                            # Perform Welch's T-test (unequal variances)
                            stat, p_value = stats.ttest_ind(
                                baseline_scores, model_scores, equal_var=False
                            )
                            stars = get_significance_stars(p_value)
                            text = f"{stars} (p={p_value:.3f})"
                        except Exception as e:
                            text = "(t-test err)"
                            print(f"t-test failed for {model_name}: {e}")

                p_value_texts.append((i, text)) # Store (y_pos, text)

    #  Find position for and plot p-value text in an aligned column
    # Place text column just past the longest bar+error
    text_x_pos = plot_max_data + 0.005

    max_text_width_buffer = 0.03 # Assume p-value text takes this much space

    for i, text in p_value_texts:
        if text:
            plt.text(text_x_pos, i, text, va="center", ha="left", fontsize=9, color="black")

    # --- 7. Finalize Plot Limits & Layout ---

    # Now set the right limit, adding space for the p-value text
    plot_max = text_x_pos + max_text_width_buffer
    plt.xlim(left=plot_min, right=min(1.0, plot_max)) # Don't go past 1.0

    # Add legend for p-values
    if has_stats: # Only add legend if we actually ran stats
        plt.text(
            0.99,
            0.01,
            "* p<0.05, ** p<0.01, *** p<0.001 (vs 'dense' baseline)",
            transform=plt.gca().transAxes,
            ha="right",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.2"),
        )

    plt.legend(loc="lower right")
    plt.tight_layout()
    plot_path = os.path.join(plot_save_dir, "ablation_accuracy_comparison.png")
    plt.savefig(plot_path)
    print(f"Ablation comparison plot saved to {plot_path}")
    plt.close()


#  Function to plot specialization results
def plot_specialization_results(all_run_results, plot_save_dir):
    """
    Plots a bar chart comparing MoE model specialization with error bars
    and statistical significance relative to the 'softmax-moe' baseline.
    """
    print("\n" + "=" * 80)
    print("📈 RUNNING: Ablation Comparison Plotting (Specialization)")
    print("=" * 80)

    if not all_run_results:
        print("No results to plot. Skipping.")
        return

    df = pd.DataFrame(all_run_results)

    # We need to plot for 'specialization_score_last_layer'
    # and 'specialization_score_first_layer'
    metrics_to_plot = [
        "specialization_score_last_layer",
        "specialization_score_first_layer"
    ]
    baseline_model = "softmax-moe" # Baseline for specialization comparison

    for metric_name in metrics_to_plot:
        # 1. Filter data:
        # Drop rows where this metric is NaN (i.e., dense models)
        metric_df = df.dropna(subset=[metric_name])

        if metric_df.empty:
            print(f"No data found for metric '{metric_name}'. Skipping plot.")
            continue

        num_seeds = metric_df["seed"].nunique()

        # 2. Group and summarize
        grouped = metric_df.groupby("model_base_name")
        summary = grouped[metric_name].agg(["mean", "sem"]).reset_index()
        summary = summary.sort_values(by="mean", ascending=False)
        summary["sem"] = summary["sem"].fillna(0) # Handle N=1 seed

        plt.figure(figsize=(14, 8))

        # 3. Create Bar Plot
        barplot = sns.barplot(
            x="mean", y="model_base_name", data=summary, orient="h", palette="plasma"
        )

        layer_name = "Last Layer" if "last" in metric_name else "First Layer"
        plt.title(f"Expert Specialization Score ({layer_name}) (N={num_seeds} seed{'s' if num_seeds != 1 else ''})", fontsize=16)
        plt.xlabel("Mean Specialization Score (Avg. StdDev across Topics)", fontsize=12)
        plt.ylabel("MoE Model Configuration", fontsize=12)

        # 4. Dynamic Axis Limits
        min_val = summary["mean"].min()
        max_val = summary["mean"].max()
        max_sem = summary["sem"].max()
        # Start near 0 or just below the lowest bar
        plot_min = max(0, min_val - max_sem - (min_val * 0.1))
        plot_max_data = max_val + max_sem
        plt.xlim(left=plot_min)

        # 5. Add Error Bars & Mean Value Text
        for i, (idx, row) in enumerate(summary.iterrows()):
            plt.errorbar(
                x=row["mean"], y=i, xerr=row["sem"],
                fmt="none", c="black", capsize=5,
                label="Standard Error" if i == 0 else None
            )
            plt.text(
                row["mean"] + (max_val * 0.001), # Just to the right
                i,
                f"{row['mean']:.4f}",
                va="center", ha="left", fontsize=9, color="black", fontweight="bold"
            )

        # 6. Add Statistical Significance
        p_value_texts = []
        has_stats = False

        if baseline_model in summary["model_base_name"].values:
            baseline_scores = metric_df[metric_df["model_base_name"] == baseline_model][metric_name].dropna()

            if len(baseline_scores) < 2:
                print(f"Not enough '{baseline_model}' runs (N<2) for t-test. Skipping.")
            else:
                has_stats = True
                for i, (idx, row) in enumerate(summary.iterrows()):
                    model_name = row["model_base_name"]
                    text = ""
                    if model_name != baseline_model:
                        model_scores = metric_df[metric_df["model_base_name"] == model_name][metric_name].dropna()
                        if len(model_scores) < 2:
                            text = "(N<2)"
                        else:
                            try:
                                stat, p_value = stats.ttest_ind(
                                    baseline_scores, model_scores, equal_var=False
                                )
                                stars = get_significance_stars(p_value)
                                text = f"{stars} (p={p_value:.3f})"
                            except Exception:
                                text = "(t-test err)"
                    p_value_texts.append((i, text))

        # 7. Finalize Plot Limits & Layout
        # Position p-value text column
        text_x_pos = plot_max_data + (plot_max_data * 0.02)
        # Add buffer for the text
        max_text_width_buffer = plot_max_data * 0.25

        for i, text in p_value_texts:
            if text:
                plt.text(text_x_pos, i, text, va="center", ha="left", fontsize=9, color="black")

        plot_max = text_x_pos + max_text_width_buffer
        plt.xlim(left=plot_min, right=plot_max)

        if has_stats:
            plt.text(
                0.99, 0.01,
                f"* p<0.05, ** p<0.01, *** p<0.001 (vs '{baseline_model}' baseline)",
                transform=plt.gca().transAxes, ha="right", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.2"),
            )

        plt.legend(loc="lower right")
        plt.tight_layout()
        plot_path = os.path.join(plot_save_dir, f"ablation_specialization_comparison_{layer_name.lower().replace(' ', '_')}.png")
        plt.savefig(plot_path)
        print(f"Specialization comparison plot saved to {plot_path}")
        plt.close()


# --- 9. Main Execution (Ablation Runner) ---
# =================================================================
if __name__ == "__main__":

    # 6. NEW: 3080/Ampere Optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # --- 1. Define Ablation Configurations ---
    NUM_SEEDS = 5  # Set number of random seeds to run (e.g., 3 for stats)
    BASE_SEED = 42

    ablation_configs = {
        # 1. Dense Baseline (Pre-trained)
        "dense": {
            "is_moe": False,
        },
        # NEW FAIR COMPARISON MODEL
        "dense-random-ffn": {
            "is_moe": False,
        },
        # 2. Softmax MoE (no energy loss)
        "softmax-moe": {
            "is_moe": True,
            "router_type": "softmax",
            "energy_loss_alpha": 0.0,
        },
        # 3. Softmax MoE + Energy Loss
        "softmax-moe-energy": {
            "is_moe": True,
            "router_type": "softmax",
            "energy_loss_alpha": 0.01, # Same as astrocyte example
        },
        # 4. Astrocyte (with energy loss) - User's original config
        "astrocyte": {
            "is_moe": True,
            "router_type": "astrocyte",
            "energy_loss_alpha": 0.01,
        },
        # 5. Astrocyte (no energy loss)
        "astrocyte-no-energy": {
            "is_moe": True,
            "router_type": "astrocyte",
            "energy_loss_alpha": 0.0,
        },
        #  Unleash router (no CV loss) and boost purity loss
        "astrocyte-unleashed": {
            "is_moe": True,
            "router_type": "astrocyte",
            "alpha_balance": 0.0,      # Strategy 1: Kill CV loss
            "energy_loss_alpha": 0.05,  # Strategy 1: Boost Purity loss
        },
        # Hierarchical (additive) routing
        "astrocyte-hierarchical": {
            "is_moe": True,
            "router_type": "astrocyte-hierarchical",
            "energy_loss_alpha": 0.01, # Keep same as baseline astrocyte
        },
        # Mean-pooled context routing
        "astrocyte-meanpool": {
            "is_moe": True,
            "router_type": "astrocyte-meanpool",
            "energy_loss_alpha": 0.01, # Keep same as baseline astrocyte
        },
    }

    # Base parameters for all runs
    base_run_config = {
        "model_name_or_path": "google/bert_uncased_L-4_H-512_A-8",
        "dataset_name": "ag_news",
        "num_experts": 8,
        "alpha_balance": 0.01, # Common for all MoE runs (unless overridden)
        "max_seq_length": 128,
    }

    # 2. Define Base Training Arguments
    master_output_dir = "./results"
    log_dir = "./logs"  # Master log directory
    ablation_plot_dir = "./ablation_plots" # Dir for final comparison plot
    os.makedirs(ablation_plot_dir, exist_ok=True)


    base_training_args = TrainingArguments(
        output_dir=master_output_dir,  # Will be sub-divided in run_training
        logging_dir=log_dir,
        report_to="tensorboard",
        num_train_epochs=6, # INCREASED from 4 to 6
        per_device_train_batch_size=192,
        per_device_eval_batch_size=384,
        dataloader_num_workers=4,
        dataloader_pin_memory=False, # Set to False if getting OOM errors
        fp16=True,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=250,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
    )

    # 3. Define Analysis Parameters
    NUM_ANALYSIS_SAMPLES = 2000
    if DEBUG:
        NUM_ANALYSIS_SAMPLES = 100

    # List to store results from all runs
    all_run_results = []

    # --- 4. Start Ablation & Seed Loops ---
    print(f"--- STARTING ABLATION STUDY (N={NUM_SEEDS} SEEDS) ---")

    for seed in range(NUM_SEEDS):
        current_seed = BASE_SEED + seed
        print("\n" + "=" * 80)
        print(f" RUNNING SEED {seed+1}/{NUM_SEEDS} (Seed Value: {current_seed})")
        print("=" * 80)

        # Set seeds for reproducibility
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        # Note: training_args.seed will be set inside the loop

        for config_name, config_overrides in ablation_configs.items():

            # 1. Create deep copies of configs to avoid mutation
            run_config = copy.deepcopy(base_run_config)
            run_config.update(config_overrides)

            training_args = copy.deepcopy(base_training_args)

            # 2. Set seed in TrainingArguments
            training_args.seed = current_seed

            # 3. Parse config objects
            script_args = ScriptArguments(**run_config)

            print(f"\n---  Starting Pipeline for: {config_name} (Seed {current_seed}) ---")

            # 4. Run Training (or skip/resume)
            # This function now handles all directory naming
            # We pass config_name to run_training so it can be used for model_base_name
            trained_model_path, run_output_dir, eval_results, model_base_name = run_training(
                script_args,
                training_args,
                seed=current_seed # Pass the seed for dir naming
            )

            # 5. Calculate Specialization Scores
            specialization_score_last_layer = None
            specialization_score_first_layer = None

            if script_args.is_moe:
                # Use the same NUM_ANALYSIS_SAMPLES as other analyses
                specialization_score_last_layer = calculate_specialization_score(
                    trained_model_path,
                    num_samples=NUM_ANALYSIS_SAMPLES,
                    layer_idx=-1 # Last Layer
                )
                specialization_score_first_layer = calculate_specialization_score(
                    trained_model_path,
                    num_samples=NUM_ANALYSIS_SAMPLES,
                    layer_idx=0 # First Layer
                )

            # 6. Store results
            if eval_results:
                run_data = {
                    "model_base_name": model_base_name,
                    "config_name": config_name,
                    "seed": current_seed,
                    "specialization_score_last_layer": specialization_score_last_layer,
                    "specialization_score_first_layer": specialization_score_first_layer,
                    **eval_results
                }
                all_run_results.append(run_data)
            else:
                print(f"Warning: No eval results for {config_name} (Seed {current_seed}).")

            # 7. Create plot directory for this *specific* run
            plot_dir = os.path.join(run_output_dir, "analysis_plots")
            os.makedirs(plot_dir, exist_ok=True)

            # 8. Run Post-Training Analysis (for this run)
            # Only run MoE-specific analysis if it's an MoE model
            if script_args.is_moe:
                print(f"\n--- Starting MoE Post-Training Analysis on {trained_model_path} ---")
                run_qualitative_analysis(trained_model_path)
                run_quantitative_analysis(
                    trained_model_path, plot_dir=plot_dir, num_samples=NUM_ANALYSIS_SAMPLES
                )
                run_layer_depth_analysis(
                    trained_model_path, plot_dir=plot_dir, num_samples=NUM_ANALYSIS_SAMPLES
                )
            else:
                print(f"\n--- Skipping MoE-specific analysis for dense model. ---")

            # 9.  Clear Memory 
            print(f"\n--- Clearing memory after {config_name} (Seed {current_seed}) ---")
            del script_args
            del training_args
            del run_config
            # Models/Trainers were local to functions, but we force GC
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"--- Memory Cleared ---")

    # --- 5. All runs complete. Plot final comparison. ---
    print("\n" + "=" * 80)
    print(" ALL ABLATION RUNS COMPLETE ")
    print("=" * 80)

    # Plot accuracy results
    plot_ablation_results(all_run_results, ablation_plot_dir)

    #  Plot specialization results
    plot_specialization_results(all_run_results, ablation_plot_dir)

    print("\n--- Full Research Pipeline Complete ---")
