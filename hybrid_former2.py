#!/usr/bin/env python
"""
Hybrid Symbolic Transformer Experiment (Improved Version)
-------------------------------------------------------------

This experiment demonstrates an improved hybrid approach that integrates a symbolic regression
module into a transformer architecture. Improvements include:
  - Dynamic pruning of candidates using a "prune score" (mimicking human synaptic pruning),
    so that a candidate is only permanently deactivated after sustained low activity.
  - Learning rate warmup followed by cosine annealing.
  - Aggressive freeze scheduling of the non-symbolic branch.
  - Better initialization of candidate scales.

Supported benchmarks: "rastrigin", "sphere", and "tough" (10 challenging 1D functions).
"""

import os, time, random, logging, gc, argparse, sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import triton
import triton.language as tl

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#####################################
# Helper function for grid calculation
#####################################

def cdiv(a, b):
    """Compute the ceiling division of a by b."""
    return (a + b - 1) // b

#####################################
# Experiment Folder Setup & Logging
#####################################

def setup_experiment_folder():
    now_str = datetime.now().strftime("%Y%m%d_%H")
    base_folder = "experiments"
    os.makedirs(base_folder, exist_ok=True)
    exp_folder = os.path.join(base_folder, f"ImprovedBenchmark_{now_str}")
    os.makedirs(exp_folder, exist_ok=True)
    log_file = os.path.join(exp_folder, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)]
    )
    logging.info("Experiment folder created: %s", exp_folder)
    return exp_folder

def save_hyperparameters(exp_folder, hyperparams):
    hyper_file = os.path.join(exp_folder, "hyperparameters.txt")
    with open(hyper_file, "w") as f:
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")
    logging.info("Hyperparameters saved to %s", hyper_file)

#####################################
# 1. TRITON KERNEL FOR CANDIDATE FUNCTIONS
#####################################

CLAMP_BOUND = tl.constexpr(1e6)
CLAMP_BOUND_FLOAT = 1e6  # Use as a float constant for torch.nan_to_num

@triton.jit
def candidate_kernel_parallel(X_ptr, Y_ptr, in_scale_ptr, out_scale_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Computes candidate function outputs in parallel.
    Candidates:
      0: x
      1: x^2
      2: cos(2*pi*x)
      3: sin(x)
      4: exp(x)
      5: 1
      6: x^3
      7: log(|x|+1)
      8: sqrt(|x|)
      9: tanh(x)  (manually implemented)
    """
    cid = tl.program_id(0)
    pid = tl.program_id(1)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask)
    scale_in = tl.load(in_scale_ptr + cid)
    x_scaled = x * scale_in
    x_clamped = tl.minimum(tl.maximum(x_scaled, -CLAMP_BOUND), CLAMP_BOUND)
    res = tl.zeros_like(x)

    if cid == 0:
        res = x_clamped
    elif cid == 1:
        res = x_clamped * x_clamped
    elif cid == 2:
        TWO_PI = tl.constexpr(6.283185307179586)
        res = tl.cos(TWO_PI * x_clamped)
    elif cid == 3:
        res = tl.sin(x_clamped)
    elif cid == 4:
        res = tl.exp(x_clamped)
    elif cid == 5:
        res = tl.full((BLOCK_SIZE,), tl.constexpr(1.0), dtype=tl.float32)
    elif cid == 6:
        res = x_clamped * x_clamped * x_clamped
    elif cid == 7:
        res = tl.log(tl.abs(x_clamped) + tl.constexpr(1.0))
    elif cid == 8:
        res = tl.sqrt(tl.abs(x_clamped))
    elif cid == 9:
        exp_val = tl.exp(x_clamped)
        exp_neg = tl.exp(-x_clamped)
        res = (exp_val - exp_neg) / (exp_val + exp_neg)
    else:
        res = x_clamped
    scale_out = tl.load(out_scale_ptr + cid)
    res = res * scale_out
    res = tl.minimum(tl.maximum(res, -CLAMP_BOUND), CLAMP_BOUND)
    tl.store(Y_ptr + offsets, res, mask=mask)

#####################################
# 2. SYMBOLIC MATH EXPERT MODULE
#####################################

class SymbolicMathExpert(nn.Module):
    """
    Computes a weighted sum of candidate functions for a 1D function.
    Provides methods to extract a symbolic equation and refine it via LS.
    Implements a dynamic "prune score" for each candidate (mimicking synaptic pruning).
    """
    def __init__(self, input_dim, num_candidates=10, debug=False, decimals=3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.num_candidates = num_candidates
        self.debug = debug
        self.decimals = decimals
        self.log_in_scale = nn.Parameter(torch.randn(num_candidates) * 0.05)
        self.log_out_scale = nn.Parameter(torch.randn(num_candidates) * 0.05)
        self.gate_fc = nn.Linear(input_dim, num_candidates)
        self.candidate_activations = None
        # Initialize a prune score for each candidate (similar to synaptic strength)
        self.register_buffer("prune_score", torch.zeros(num_candidates))

    def forward(self, x):
        x_norm = self.input_norm(x)
        B, S, D = x_norm.shape
        N = B * S * D
        x_flat = x_norm.view(-1)
        y_candidates = torch.empty((self.num_candidates, N), device=x.device, dtype=x.dtype)
        BLOCK_SIZE = 512
        grid = (self.num_candidates, cdiv(N, BLOCK_SIZE))
        candidate_kernel_parallel[grid](x_flat, y_candidates,
                                        self.log_in_scale.exp(), self.log_out_scale.exp(),
                                        N, BLOCK_SIZE=BLOCK_SIZE)
        candidates = y_candidates.view(self.num_candidates, B, S, D).permute(1, 2, 0, 3)
        candidates = F.layer_norm(candidates, candidates.shape[2:])
        candidates = torch.clamp(candidates, -1e6, 1e6)
        candidates = torch.nan_to_num(candidates, nan=0.0, posinf=CLAMP_BOUND_FLOAT, neginf=-CLAMP_BOUND_FLOAT)
        gate_logits = self.gate_fc(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        weighted = (gate_weights.unsqueeze(-1) * candidates).sum(dim=2)
        self.candidate_activations = candidates.detach().cpu()
        return weighted, gate_weights

    def extract_symbolic_equation(self, var_name="x", weight_threshold=0.05):
        dummy_val = torch.ones(1, self.gate_fc.in_features, device=self.gate_fc.weight.device)
        with torch.no_grad():
            gate_logits = self.gate_fc(dummy_val)
            gate_weights = F.softmax(gate_logits, dim=-1).squeeze(0).cpu().numpy()
        x_sym = sp.symbols(var_name)
        candidate_dict = {
            0: x_sym,
            1: x_sym**2,
            2: sp.cos(2*sp.pi*x_sym),
            3: sp.sin(x_sym),
            4: sp.exp(x_sym),
            5: 1,
            6: x_sym**3,
            7: sp.log(sp.Abs(x_sym)+1),
            8: sp.sqrt(sp.Abs(x_sym)),
            9: sp.tanh(x_sym)
        }
        in_scales = self.log_in_scale.exp().detach().cpu().numpy()
        out_scales = self.log_out_scale.exp().detach().cpu().numpy()
        equation = 0
        selected = False
        for i in range(self.num_candidates):
            weight = gate_weights[i]
            if weight > weight_threshold:
                selected = True
                coeff = np.round(weight * in_scales[i] * out_scales[i], self.decimals)
                equation += coeff * candidate_dict[i]
        if not selected:
            i = int(np.argmax(gate_weights))
            coeff = np.round(gate_weights[i] * in_scales[i] * out_scales[i], self.decimals)
            equation = coeff * candidate_dict[i]
        return sp.simplify(equation)

    def refine_symbolic_equation_with_ls(self, x_data, y_data, var_name="x", weight_threshold=0.05, round_threshold=1e-3):
        dummy_val = torch.ones(1, self.gate_fc.in_features, device=self.gate_fc.weight.device)
        with torch.no_grad():
            gate_logits = self.gate_fc(dummy_val)
            gate_weights = F.softmax(gate_logits, dim=-1).squeeze(0).cpu().numpy()
        selected_indices = [i for i in range(self.num_candidates) if gate_weights[i] > weight_threshold]
        if len(selected_indices) == 0:
            selected_indices = [int(np.argmax(gate_weights))]
        candidate_funcs_numeric = {
            0: lambda x: x,
            1: lambda x: x**2,
            2: lambda x: torch.cos(2 * torch.pi * x),
            3: lambda x: torch.sin(x),
            4: lambda x: torch.exp(x),
            5: lambda x: torch.ones_like(x),
            6: lambda x: x**3,
            7: lambda x: torch.log(torch.abs(x) + 1.0),
            8: lambda x: torch.sqrt(torch.abs(x)),
            9: lambda x: torch.tanh(x)
        }
        x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
        candidate_columns = [candidate_funcs_numeric[idx](x_tensor) for idx in selected_indices]
        X_candidates = torch.cat(candidate_columns, dim=1)
        y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
        solution = torch.linalg.lstsq(X_candidates, y_tensor).solution.squeeze(1).cpu().numpy()
        coefficients = np.where(np.abs(solution) < round_threshold, 0, solution)
        coefficients = np.round(coefficients, self.decimals)
        x_sym = sp.symbols(var_name)
        candidate_funcs_symbolic = {
            0: lambda x_sym: x_sym,
            1: lambda x_sym: x_sym**2,
            2: lambda x_sym: sp.cos(2*sp.pi*x_sym),
            3: lambda x_sym: sp.sin(x_sym),
            4: lambda x_sym: sp.exp(x_sym),
            5: lambda x_sym: 1,
            6: lambda x_sym: x_sym**3,
            7: lambda x_sym: sp.log(sp.Abs(x_sym)+1),
            8: lambda x_sym: sp.sqrt(sp.Abs(x_sym)),
            9: lambda x_sym: sp.tanh(x_sym)
        }
        equation = 0
        for coeff, idx in zip(coefficients, selected_indices):
            equation += coeff * candidate_funcs_symbolic[idx](x_sym)
        return sp.simplify(equation)

    def update_prune_scores(self, threshold, gamma=0.01, delta=0.005, maturity_threshold=1.0):
        """
        Updates a dynamic "prune score" for each candidate. If a candidate’s effective coefficient
        is below the threshold, its score increases; otherwise, it decreases. Once the score exceeds
        maturity_threshold, the candidate is permanently pruned.
        """
        with torch.no_grad():
            effective_coeff = self.log_in_scale.data.exp() * self.log_out_scale.data.exp()
            below = (effective_coeff < threshold).float()
            above = (effective_coeff >= threshold).float()
            self.prune_score.add_(gamma * below - delta * above)
            self.prune_score.clamp_(min=0)
            mask = self.prune_score > maturity_threshold
            if mask.any():
                self.log_in_scale.data[mask] = -100.0
                self.log_out_scale.data[mask] = -100.0
                logging.info("Permanently pruned %d candidate(s) (prune_score > %.2f).", int(mask.sum().item()), maturity_threshold)

#####################################
# 3. BASELINE MLP REGRESSOR (inside hybrid branch)
#####################################

class MLPRegressor(nn.Module):
    """
    A simple MLP for regression.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    def forward(self, x):
        B, S, D = x.shape
        x = x.view(B * S, D)
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.fc3(out)
        out = out.view(B, S, -1)
        return out.mean(dim=1)

#####################################
# 4. HYBRID SYMBOLIC REGRESSOR
#####################################

class HybridSymbolicRegressor(nn.Module):
    """
    Fuses a symbolic branch with a standard MLP branch.
    Also supports gradually freezing the MLP branch.
    """
    def __init__(self, d_model, hidden_dim, output_dim=1, symbolic_weight=0.5, debug=False):
        super().__init__()
        self.symbolic_expert = SymbolicMathExpert(input_dim=hidden_dim, debug=debug)
        self.mlp_branch = MLPRegressor(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.fc_in = nn.Linear(d_model, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.raw_symbolic_weight = nn.Parameter(torch.tensor(symbolic_weight))
        self.debug = debug
        self.last_gate_weights = None
        self.use_symbolic_only = False
    def forward(self, x):
        x_proj = F.relu(self.fc_in(x))
        symbolic_out, gate_weights = self.symbolic_expert(x_proj)
        self.last_gate_weights = gate_weights
        if self.use_symbolic_only:
            effective_symbolic_weight = 1.0
            mlp_out = torch.zeros_like(symbolic_out.mean(dim=1))
        else:
            mlp_out = self.mlp_branch(x_proj)
            effective_symbolic_weight = torch.sigmoid(self.raw_symbolic_weight)
        combined = effective_symbolic_weight * symbolic_out.mean(dim=1) + (1 - effective_symbolic_weight) * mlp_out
        out = self.fc_out(combined)
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

#####################################
# 5. BASELINE TRANSFORMER MODEL
#####################################

class BaselineTransformer(nn.Module):
    """
    A transformer-based regressor.
    """
    def __init__(self, input_dim, d_model, num_transformer_layers, num_heads, hidden_dim, output_dim=1):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                    dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.fc_out = nn.Linear(d_model, output_dim)
    def forward(self, x):
        x_proj = self.fc_in(x)
        transformer_out = self.transformer_encoder(x_proj)
        pooled = transformer_out.mean(dim=1)
        return self.fc_out(pooled)

#####################################
# 6. HYBRID SYMBOLIC TRANSFORMER
#####################################

class HybridSymbolicTransformer(nn.Module):
    """
    Combines a transformer encoder with a hybrid symbolic regression branch.
    """
    def __init__(self, input_dim, d_model, num_transformer_layers, num_heads,
                 hidden_dim, output_dim=1, symbolic_weight=0.5, debug=False):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                    dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.hybrid_regressor = HybridSymbolicRegressor(d_model=d_model, hidden_dim=hidden_dim,
                                                        output_dim=output_dim,
                                                        symbolic_weight=symbolic_weight, debug=debug)
    def forward(self, x):
        x_proj = self.fc_in(x)
        transformer_out = self.transformer_encoder(x_proj)
        return self.hybrid_regressor(transformer_out)

#####################################
# 7. DATA GENERATION (Benchmark Functions)
#####################################

def generate_data(benchmark, num_samples, input_dim, low, high):
    """
    Generates samples uniformly in [low, high]^input_dim.
    For 'rastrigin': f(x) = A*input_dim + sum(x_i^2 - A*cos(2*pi*x_i))
    For 'sphere': f(x) = sum(x_i^2)
    For 'tough': returns a dict of 10 tough functions.
    """
    if benchmark.lower() == "rastrigin":
        X = torch.empty(num_samples, input_dim).uniform_(low, high)
        A = 10.0
        f = A * input_dim + torch.sum(X**2 - A * torch.cos(2 * np.pi * X), dim=1, keepdim=True)
    elif benchmark.lower() == "sphere":
        X = torch.empty(num_samples, input_dim).uniform_(low, high)
        f = torch.sum(X**2, dim=1, keepdim=True)
    elif benchmark.lower() == "tough":
        X = torch.empty(num_samples, input_dim).uniform_(-10, 10)
        tough_functions = {
            "tough1_rastrigin": lambda x: 10*1 + x**2 - 10*torch.cos(2*np.pi*x),
            "tough2_sphere": lambda x: torch.sum(x**2, dim=1, keepdim=True),
            "tough3_ackley": lambda x: -20*torch.exp(-0.2*torch.abs(x)) - torch.exp(torch.cos(2*np.pi*x)) + 20 + np.e,
            "tough4_griewank": lambda x: 1 + (x**2)/4000 - torch.cos(x),
            "tough5_schwefel": lambda x: 418.9829 - x * torch.sin(torch.sqrt(torch.abs(x))),
            "tough6_michalewicz": lambda x: - torch.sin(x) * (torch.sin(x**2/np.pi))**20,
            "tough7_levy": lambda x: (torch.sin(3*np.pi*(1+(x-1)/4))**2 +
                                      ((1+(x-1)/4)-1)**2*(1+torch.sin(3*np.pi*(1+(x-1)/4)+1)**2) +
                                      ((1+(x-1)/4)-1)**2*(1+torch.sin(2*np.pi*(1+(x-1)/4))**2)),
            "tough8_alpine": lambda x: torch.abs(x*torch.sin(x) + 0.1*x),
            "tough9_step": lambda x: (torch.floor(x) - x).unsqueeze(1),
            "tough10_poly": lambda x: (x**5 - 2*x**4 + x**3 - x**2 + x)
        }
        data_dict = {}
        for name, func in tough_functions.items():
            f_val = func(X)
            f_val = torch.nan_to_num(f_val, nan=0.0, posinf=CLAMP_BOUND_FLOAT, neginf=-CLAMP_BOUND_FLOAT)
            if f_val.dim() == 1:
                f_val = f_val.unsqueeze(1)
            data_dict[name] = (X, f_val)
        return data_dict
    else:
        raise ValueError("Unknown benchmark function.")
    f = torch.nan_to_num(f, nan=0.0, posinf=CLAMP_BOUND_FLOAT, neginf=-CLAMP_BOUND_FLOAT)
    return X, f

#####################################
# 8. PRUNING & FREEZING UTILITIES (Improved)
#####################################

def prune_small_candidates(model, threshold, gamma=0.01, delta=0.005, maturity_threshold=1.0):
    """
    Updates a dynamic "prune score" for each candidate.
    Increases the score if effective coefficient is below threshold,
    decreases if above, and permanently prunes when the score exceeds maturity_threshold.
    """
    symbolic_expert = None
    if hasattr(model, 'hybrid_regressor'):
        symbolic_expert = model.hybrid_regressor.symbolic_expert
    elif hasattr(model, 'symbolic_expert'):
        symbolic_expert = model.symbolic_expert
    if symbolic_expert is not None:
        with torch.no_grad():
            effective_coeff = symbolic_expert.log_in_scale.data.exp() * symbolic_expert.log_out_scale.data.exp()
            below = (effective_coeff < threshold).float()
            above = (effective_coeff >= threshold).float()
            symbolic_expert.prune_score.add_(gamma * below - delta * above)
            symbolic_expert.prune_score.clamp_(min=0)
            mask = symbolic_expert.prune_score > maturity_threshold
            if mask.any():
                symbolic_expert.log_in_scale.data[mask] = -100.0
                symbolic_expert.log_out_scale.data[mask] = -100.0
                logging.info("Permanently pruned %d candidate(s) (prune_score > %.2f).", int(mask.sum().item()), maturity_threshold)

def freeze_mlp_branch(model):
    """
    Freezes the MLP branch so that only the symbolic branch is used.
    """
    if hasattr(model, 'hybrid_regressor'):
        for param in model.hybrid_regressor.mlp_branch.parameters():
            param.data.zero_()
            param.requires_grad = False
        model.hybrid_regressor.raw_symbolic_weight.data.fill_(10.0)
        model.hybrid_regressor.use_symbolic_only = True
        logging.info("MLP branch fully frozen; now using symbolic branch only.")

#####################################
# 9. TRAINING FUNCTION (Improved)
#####################################

def train_model(model, train_loader, val_loader, num_epochs, lr, device,
                complexity_lambda, entropy_lambda, accuracy_threshold,
                initial_prune_threshold, freeze_mlp_start, freeze_mlp_end, warmup_epochs=50):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    # Learning rate warmup + cosine annealing
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    scheduler = LambdaLR(optimizer, lr_lambda)

    optimizer.zero_grad()
    dummy = torch.tensor(0.0, device=device, requires_grad=True)
    dummy.backward()
    optimizer.step()
    scheduler.step()

    criterion = nn.MSELoss()
    scaler = GradScaler()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Dynamic pruning threshold decays over time
        current_prune_threshold = initial_prune_threshold * (0.9 ** (epoch / 100.0))

        model.train()
        epoch_loss = 0.0
        train_preds_list = []
        train_targets_list = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=(device=='cuda')):
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                if hasattr(model, 'hybrid_regressor'):
                    symbolic_expert = model.hybrid_regressor.symbolic_expert
                    coeffs = torch.exp(symbolic_expert.log_in_scale) * torch.exp(symbolic_expert.log_out_scale)
                    complexity_loss = torch.sum(torch.abs(coeffs))
                    gate_weights = model.hybrid_regressor.last_gate_weights
                    if gate_weights is not None:
                        eps = 1e-8
                        entropy = -torch.sum(gate_weights * torch.log(gate_weights + eps), dim=-1)
                        entropy_loss = torch.mean(entropy)
                    else:
                        entropy_loss = 0.0
                    loss = loss + complexity_lambda * complexity_loss + entropy_lambda * entropy_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * batch_x.size(0)
            train_preds_list.append(preds.detach().cpu())
            train_targets_list.append(batch_y.detach().cpu())

        train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_preds_cat = torch.cat(train_preds_list, dim=0)
        train_targets_cat = torch.cat(train_targets_list, dim=0)
        train_acc = (torch.abs(train_preds_cat - train_targets_cat) < accuracy_threshold).float().mean().item()
        train_accuracies.append(train_acc)

        model.eval()
        val_loss_epoch = 0.0
        val_preds_list = []
        val_targets_list = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                with autocast(device_type='cuda', enabled=(device=='cuda')):
                    preds = model(batch_x)
                    loss_val = criterion(preds, batch_y)
                val_loss_epoch += loss_val.item() * batch_x.size(0)
                val_preds_list.append(preds.detach().cpu())
                val_targets_list.append(batch_y.detach().cpu())
        val_loss = val_loss_epoch / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_preds_cat = torch.cat(val_preds_list, dim=0)
        val_targets_cat = torch.cat(val_targets_list, dim=0)
        val_acc = (torch.abs(val_preds_cat - val_targets_cat) < accuracy_threshold).float().mean().item()
        val_accuracies.append(val_acc)

        scheduler.step()
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

        # Apply the dynamic, brain-like candidate pruning
        prune_small_candidates(model, threshold=current_prune_threshold)

        # Aggressive freeze scheduling for MLP branch
        if hasattr(model, 'hybrid_regressor'):
            if epoch + 1 >= freeze_mlp_start and epoch + 1 < freeze_mlp_end:
                alpha = (epoch + 1 - freeze_mlp_start + 1) / (freeze_mlp_end - freeze_mlp_start + 1)
                new_val = alpha * 10.0
                model.hybrid_regressor.raw_symbolic_weight.data = torch.tensor(new_val, device=device)
                logging.info("Updating raw_symbolic_weight to %.2f at epoch %d", new_val, epoch+1)
            elif epoch + 1 >= freeze_mlp_end:
                freeze_mlp_branch(model)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses, train_accuracies, val_accuracies

#####################################
# 10. PLOTTING UTILS
#####################################

def plot_loss_and_accuracy(all_train_losses, all_val_losses, all_train_acc, all_val_acc, title_prefix="", save_path=None):
    if len(all_train_losses) == 0:
        print("No training losses available to plot.")
        return
    epochs = len(all_train_losses[0])
    train_losses = np.array(all_train_losses)
    val_losses = np.array(all_val_losses)
    train_acc = np.array(all_train_acc)
    val_acc = np.array(all_val_acc)

    mean_train_loss = np.mean(train_losses, axis=0)
    std_train_loss = np.std(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    std_val_loss = np.std(val_losses, axis=0)

    mean_train_acc = np.mean(train_acc, axis=0)
    std_train_acc = np.std(train_acc, axis=0)
    mean_val_acc = np.mean(val_acc, axis=0)
    std_val_acc = np.std(val_acc, axis=0)

    x_axis = np.arange(1, epochs+1)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(x_axis, mean_train_loss, label="Train Loss")
    plt.fill_between(x_axis, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.3)
    plt.plot(x_axis, mean_val_loss, label="Val Loss")
    plt.fill_between(x_axis, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title_prefix + " Loss Curves")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(x_axis, mean_train_acc, label="Train Accuracy")
    plt.fill_between(x_axis, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha=0.3)
    plt.plot(x_axis, mean_val_acc, label="Val Accuracy")
    plt.fill_between(x_axis, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (Fraction within threshold)")
    plt.title(title_prefix + " Accuracy Curves")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

#####################################
# 11. MAIN FUNCTION: RUN EXPERIMENTS
#####################################

def main():
    parser = argparse.ArgumentParser(description="Improved Hybrid Symbolic Transformer Experiment")
    parser.add_argument('--model', type=str, choices=['baseline', 'hybrid', 'both'], default='both',
                        help="Choose model type: 'baseline', 'hybrid', or 'both'.")
    parser.add_argument('--benchmark', type=str, choices=['rastrigin', 'sphere', 'tough'], default='rastrigin',
                        help="Benchmark function: 'rastrigin', 'sphere', or 'tough' (10 challenging functions).")
    parser.add_argument('--num_runs', type=int, default=3,
                        help="Number of runs (different seeds) for confidence intervals.")
    args = parser.parse_args()

    exp_folder = setup_experiment_folder()
    hyperparams = {
        "input_dim": 1,
        "d_model": 64,
        "num_transformer_layers": 2,
        "num_heads": 4,
        "hidden_dim": 128,
        "output_dim": 1,
        "num_samples": 15000,
        "seq_len": 1,
        "num_epochs": 1000,  # For demonstration; increase for real experiments
        "lr": 1e-4,
        "batch_size": 16,
        "complexity_lambda": 1e-5,
        "entropy_lambda": 1e-5,
        "num_candidates": 10,
        "BLOCK_SIZE": 512,
        "accuracy_threshold": 2.0,
        "prune_threshold": 1e-3,  # Initial pruning threshold
        "freeze_mlp_start": 1500,  # Freeze MLP branch starting at epoch 500
        "freeze_mlp_end": 1700     # Fully freeze by epoch 700
    }
    hyperparams["benchmark"] = args.benchmark
    hyperparams["num_runs"] = args.num_runs
    save_hyperparameters(exp_folder, hyperparams)
    logging.info("Hyperparameters: %s", hyperparams)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    overall_rmse = {}
    symbolic_metrics = {}  # For refined equations
    compressed_rmse = {}   # For compressed (symbolic) predictions
    all_train_losses = []
    all_val_losses = []
    all_train_acc = []
    all_val_acc = []

    if args.benchmark.lower() == "tough":
        data_dict = generate_data("tough", hyperparams["num_samples"], hyperparams["input_dim"], -10, 10)
        overlay_data = {}
        for name, (X_data, y_data) in data_dict.items():
            logging.info("Running tough benchmark: %s", name)
            X_data = X_data.view(hyperparams["num_samples"], hyperparams["seq_len"], hyperparams["input_dim"])
            y_data = y_data.view(hyperparams["num_samples"], 1)
            indices = torch.randperm(hyperparams["num_samples"])
            train_idx = indices[:int(0.8 * hyperparams["num_samples"])]
            val_idx = indices[int(0.8 * hyperparams["num_samples"]):]
            X_train, y_train = X_data[train_idx], y_data[train_idx]
            X_val, y_val = X_data[val_idx], y_data[val_idx]
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True,
                                      num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"], shuffle=False,
                                    num_workers=4, pin_memory=True)
            if args.model in ['baseline', 'both']:
                model = BaselineTransformer(input_dim=hyperparams["input_dim"],
                                            d_model=hyperparams["d_model"],
                                            num_transformer_layers=hyperparams["num_transformer_layers"],
                                            num_heads=hyperparams["num_heads"],
                                            hidden_dim=hyperparams["hidden_dim"],
                                            output_dim=hyperparams["output_dim"])
            elif args.model in ['hybrid']:
                model = HybridSymbolicTransformer(input_dim=hyperparams["input_dim"],
                                                  d_model=hyperparams["d_model"],
                                                  num_transformer_layers=hyperparams["num_transformer_layers"],
                                                  num_heads=hyperparams["num_heads"],
                                                  hidden_dim=hyperparams["hidden_dim"],
                                                  output_dim=hyperparams["output_dim"],
                                                  symbolic_weight=0.5,
                                                  debug=False)
            train_losses, val_losses, train_acc, val_acc = train_model(
                model, train_loader, val_loader,
                num_epochs=hyperparams["num_epochs"],
                lr=hyperparams["lr"],
                device='cuda' if torch.cuda.is_available() else 'cpu',
                complexity_lambda=hyperparams["complexity_lambda"],
                entropy_lambda=hyperparams["entropy_lambda"],
                accuracy_threshold=hyperparams["accuracy_threshold"],
                initial_prune_threshold=hyperparams["prune_threshold"],
                freeze_mlp_start=hyperparams["freeze_mlp_start"],
                freeze_mlp_end=hyperparams["freeze_mlp_end"]
            )
            model.eval()
            with torch.no_grad():
                preds = model(X_val.to('cuda' if torch.cuda.is_available() else 'cpu'))
            preds_np = preds.cpu().numpy().flatten()
            y_val_np = y_val.numpy().flatten()
            rmse = np.sqrt(np.mean((preds_np - y_val_np)**2))
            overall_rmse[name] = rmse
            logging.info("Benchmark %s RMSE: %.8f", name, rmse)
            X_val_np = X_val.squeeze().cpu().numpy()
            sort_idx = np.argsort(X_val_np)
            overlay_data[name] = {"x": X_val_np[sort_idx], "true": y_val_np[sort_idx], "pred": preds_np[sort_idx]}
            if hasattr(model, 'hybrid_regressor'):
                symbolic_unrefined = model.hybrid_regressor.symbolic_expert.extract_symbolic_equation(var_name="x", weight_threshold=0.05)
                symbolic_refined = model.hybrid_regressor.symbolic_expert.refine_symbolic_equation_with_ls(
                    x_data=X_train[:, 0, :].squeeze().cpu().numpy(),
                    y_data=y_train.squeeze().cpu().numpy(),
                    var_name="x", weight_threshold=0.05, round_threshold=1e-3)
                func_symbolic = sp.lambdify(sp.symbols("x"), symbolic_refined, modules=["numpy"])
                X_val_np_flat = X_val.squeeze().cpu().numpy()
                symbolic_preds = np.array(func_symbolic(X_val_np_flat)).flatten()
                sym_rmse = np.sqrt(np.mean((symbolic_preds - y_val_np)**2))
                compressed_rmse[name] = sym_rmse

                complexity = sp.count_ops(symbolic_refined)
                mem_usage = sys.getsizeof(str(symbolic_refined))
                symbolic_metrics[name] = {"complexity": complexity, "mem_usage": mem_usage, "rmse": sym_rmse}
                logging.info("Benchmark %s: Refined Equation: %s", name, sp.pretty(symbolic_refined))
                print(f"Benchmark {name}:")
                print("  Refined Symbolic Equation:")
                print(sp.pretty(symbolic_refined))
                print("  Complexity (count_ops):", complexity)
                print("  Memory usage (bytes):", mem_usage)
                print("  Compressed Model RMSE: {:.8f}".format(sym_rmse))
                print("  Transformer Model RMSE: {:.8f}".format(rmse))
                print("------------------------------------------------------")
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_train_acc.append(train_acc)
            all_val_acc.append(val_acc)
        plt.figure(figsize=(10, 6))
        for name, data in overlay_data.items():
            plt.plot(data["x"], data["true"], label=f"{name} true", linestyle="--")
            plt.plot(data["x"], data["pred"], label=f"{name} pred")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Tough Benchmarks: Ground Truth vs. Transformer Prediction")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, "tough_overlay.png"))
        plt.close()
        if symbolic_metrics:
            complexities = [symbolic_metrics[k]["complexity"] for k in symbolic_metrics]
            mem_usages = [symbolic_metrics[k]["mem_usage"] for k in symbolic_metrics]
            rmses_sym = [symbolic_metrics[k]["rmse"] for k in symbolic_metrics]
            plt.figure(figsize=(8,6))
            plt.scatter(complexities, rmses_sym, s=np.array(mem_usages)/10, alpha=0.7)
            for k in symbolic_metrics:
                plt.annotate(k, (symbolic_metrics[k]["complexity"], symbolic_metrics[k]["rmse"]))
            plt.xlabel("Symbolic Equation Complexity (count_ops)")
            plt.ylabel("Compressed Model RMSE")
            plt.title("Refined Symbolic Equations: Complexity vs. RMSE (Bubble size ~ mem usage)")
            plt.tight_layout()
            plt.savefig(os.path.join(exp_folder, "tough_symbolic_metrics.png"))
            plt.close()
    else:
        X_data, y_data = generate_data(args.benchmark, hyperparams["num_samples"],
                                       hyperparams["input_dim"], low=-5.12, high=5.12)
        X_data = X_data.view(hyperparams["num_samples"], hyperparams["seq_len"], hyperparams["input_dim"])
        y_data = y_data.view(hyperparams["num_samples"], 1)
        indices = torch.randperm(hyperparams["num_samples"])
        train_idx = indices[:int(0.8 * hyperparams["num_samples"])]
        val_idx = indices[int(0.8 * hyperparams["num_samples"]):]
        X_train, y_train = X_data[train_idx], y_data[train_idx]
        X_val, y_val = X_data[val_idx], y_data[val_idx]
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"], shuffle=False,
                                num_workers=4, pin_memory=True)

        if args.model in ['baseline', 'both']:
            logging.info("Training BaselineTransformer...")
            model = BaselineTransformer(input_dim=hyperparams["input_dim"],
                                        d_model=hyperparams["d_model"],
                                        num_transformer_layers=hyperparams["num_transformer_layers"],
                                        num_heads=hyperparams["num_heads"],
                                        hidden_dim=hyperparams["hidden_dim"],
                                        output_dim=hyperparams["output_dim"])
        elif args.model in ['hybrid']:
            logging.info("Training HybridSymbolicTransformer...")
            model = HybridSymbolicTransformer(input_dim=hyperparams["input_dim"],
                                              d_model=hyperparams["d_model"],
                                              num_transformer_layers=hyperparams["num_transformer_layers"],
                                              num_heads=hyperparams["num_heads"],
                                              hidden_dim=hyperparams["hidden_dim"],
                                              output_dim=hyperparams["output_dim"],
                                              symbolic_weight=0.5,
                                              debug=False)
        train_losses, val_losses, train_acc, val_acc = train_model(
            model, train_loader, val_loader,
            num_epochs=hyperparams["num_epochs"],
            lr=hyperparams["lr"],
            device='cuda' if torch.cuda.is_available() else 'cpu',
            complexity_lambda=hyperparams["complexity_lambda"],
            entropy_lambda=hyperparams["entropy_lambda"],
            accuracy_threshold=hyperparams["accuracy_threshold"],
            initial_prune_threshold=hyperparams["prune_threshold"],
            freeze_mlp_start=hyperparams["freeze_mlp_start"],
            freeze_mlp_end=hyperparams["freeze_mlp_end"]
        )

        model.eval()
        with torch.no_grad():
            preds = model(X_val.to('cuda' if torch.cuda.is_available() else 'cpu'))
        preds_np = preds.cpu().numpy().flatten()
        y_val_np = y_val.numpy().flatten()
        rmse = np.sqrt(np.mean((preds_np - y_val_np)**2))
        overall_rmse[args.benchmark] = rmse
        logging.info("RMSE: %.8f", rmse)

        if hasattr(model, 'hybrid_regressor'):
            symbolic_unrefined = model.hybrid_regressor.symbolic_expert.extract_symbolic_equation(var_name="x", weight_threshold=0.05)
            symbolic_refined = model.hybrid_regressor.symbolic_expert.refine_symbolic_equation_with_ls(
                x_data=X_train[:, 0, :].squeeze().cpu().numpy(),
                y_data=y_train.squeeze().cpu().numpy(),
                var_name="x", weight_threshold=0.05, round_threshold=1e-3)
            x_sym = sp.symbols("x")
            if args.benchmark.lower() == "rastrigin":
                true_eq = 10*1 + x_sym**2 - 10*sp.cos(2*sp.pi*x_sym)
            elif args.benchmark.lower() == "sphere":
                true_eq = x_sym**2
            else:
                true_eq = sp.sympify("0")
            logging.info("Unrefined Symbolic Equation:\n%s", sp.pretty(symbolic_unrefined))
            logging.info("Refined Symbolic Equation:\n%s", sp.pretty(symbolic_refined))
            logging.info("True Equation:\n%s", sp.pretty(true_eq))
            print("Unrefined Symbolic Equation:")
            print(sp.pretty(symbolic_unrefined))
            print("Refined Symbolic Equation:")
            print(sp.pretty(symbolic_refined))
            print("True Equation:")
            print(sp.pretty(true_eq))
            print("------------------------------------------------------")

        plot_loss_and_accuracy(all_train_losses, all_val_losses, all_train_acc, all_val_acc,
                               title_prefix=f"{args.benchmark.capitalize()} Benchmark",
                               save_path=os.path.join(exp_folder, f"{args.benchmark}_loss_accuracy.png"))

    mean_rmse = np.mean(list(overall_rmse.values()))
    logging.info("Mean RMSE over runs: %.8f", mean_rmse)
    results_summary = f"\n----- Final Experiment Summary -----\nMean RMSE: {mean_rmse:.8f}\n"
    print(results_summary)

if __name__ == '__main__':
    main()
