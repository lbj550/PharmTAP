#!/usr/bin/env python3
"""
Fusion ablation for multitask graph + RDKit descriptor + ECFP ADMET models.

This script compares fusion mechanisms under matched molecular evidence:

0. mt_graph_desc_ecfp_concat
   DMPNN graph embedding + RDKit descriptor embedding + ECFP embedding are
   concatenated before the shared prediction layer.

1. mt_graph_desc_ecfp_attention
   The same three modality embeddings are treated as modality tokens and fused
   with cross-modal self-attention.

2. mt_graph_desc_ecfp_gated
   The same three modality embeddings are weighted by learned sample-specific
   modality gates.

3. mt_graph_desc_ecfp_gated_attention
   The same three modality embeddings are treated as modality tokens and fused
   with cross-modal self-attention followed by learned modality gates.

The data loading, endpoint pooling, weighted multitask objective, validation
thresholding, and metrics follow admet_multimodal_core.py.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import admet_graph_desc_baseline as base
import admet_multimodal_core as mt


DATA_ROOT = os.getenv("DATA_ROOT", base.DATA_ROOT)
DATA_ROOT_LABEL = Path(DATA_ROOT).name.replace("data_", "") or "original"
ENV_SPLIT_LABEL = os.getenv("SPLIT_MODE", "")
SPLIT_LABEL = os.getenv(
    "SPLIT_LABEL",
    DATA_ROOT_LABEL if ENV_SPLIT_LABEL in ("", "original") and DATA_ROOT_LABEL != "data" else ENV_SPLIT_LABEL or DATA_ROOT_LABEL,
)

BASE_MODEL_DIR = os.getenv("MODEL_DIR", "data/admet/ADMET_multimodal_fusion_ablation_model_pth")
BASE_RESULT_DIR = os.getenv("RESULT_DIR", "data/admet/ADMET_multimodal_fusion_ablation_results")
MODEL_KIND = "ADMET_Multimodal_Fusion_Ablation"
DEVICE = base.DEVICE

BRANCH_DIM = int(os.getenv("BRANCH_DIM", str(mt.BRANCH_DIM)))
GATED_ATTENTION_HEADS = int(os.getenv("GATED_ATTENTION_HEADS", "4"))
GATED_ATTENTION_DROPOUT = float(os.getenv("GATED_ATTENTION_DROPOUT", "0.1"))

MT_BATCH_SIZE = int(os.getenv("MT_BATCH_SIZE", os.getenv("BATCH_SIZE", "96")))
MT_MAX_EPOCHS = int(os.getenv("MT_MAX_EPOCHS", os.getenv("MAX_EPOCHS", "120")))
MT_PATIENCE = int(os.getenv("MT_PATIENCE", "18"))
MT_LR = float(os.getenv("MT_LR", "2e-4"))
MT_WEIGHT_DECAY = float(os.getenv("MT_WEIGHT_DECAY", "6e-4"))
USE_WEIGHTED_SAMPLER = os.getenv("USE_WEIGHTED_SAMPLER", "1") == "1"
STOP_ON_ERROR = os.getenv("FUSION_STOP_ON_ERROR", "1").lower() not in {"0", "false", "no"}
MAX_TASKS = int(os.getenv("FUSION_MAX_TASKS", "0"))

MT_ENSEMBLE_SEEDS = [
    int(item.strip())
    for item in os.getenv("MT_ENSEMBLE_SEEDS", "42").split(",")
    if item.strip()
]

FUSION_MODES = [
    item.strip().lower()
    for item in os.getenv(
        "FUSION_MODES",
        "mt_graph_desc_ecfp_attention,mt_graph_desc_ecfp_gated,mt_graph_desc_ecfp_gated_attention",
    ).split(",")
    if item.strip()
]
VALID_FUSION_MODES = {
    "mt_graph_desc_ecfp_concat",
    "mt_graph_desc_ecfp_attention",
    "mt_graph_desc_ecfp_gated",
    "mt_graph_desc_ecfp_gated_attention",
}
invalid_modes = [mode for mode in FUSION_MODES if mode not in VALID_FUSION_MODES]
if invalid_modes:
    raise ValueError(f"Invalid FUSION_MODES: {invalid_modes}; valid={sorted(VALID_FUSION_MODES)}")

CURRENT_MODE = FUSION_MODES[0]
MODEL_DIR = BASE_MODEL_DIR
RESULT_DIR = BASE_RESULT_DIR

base.DATA_ROOT = DATA_ROOT
mt.DATA_ROOT = DATA_ROOT
mt.BRANCH_DIM = BRANCH_DIM
mt.MT_BATCH_SIZE = MT_BATCH_SIZE
mt.MT_MAX_EPOCHS = MT_MAX_EPOCHS
mt.MT_PATIENCE = MT_PATIENCE
mt.MT_LR = MT_LR
mt.MT_WEIGHT_DECAY = MT_WEIGHT_DECAY
mt.USE_WEIGHTED_SAMPLER = USE_WEIGHTED_SAMPLER
mt.MT_ENSEMBLE_SEEDS = MT_ENSEMBLE_SEEDS


def flush_print(message: str) -> None:
    print(message, flush=True)


def current_model_kind() -> str:
    return f"{MODEL_KIND}_{CURRENT_MODE}"


def set_fusion_mode(mode: str) -> None:
    global CURRENT_MODE, MODEL_DIR, RESULT_DIR
    if mode not in VALID_FUSION_MODES:
        raise ValueError(f"Invalid fusion mode: {mode}")
    CURRENT_MODE = mode
    MODEL_DIR = os.path.join(BASE_MODEL_DIR, mode)
    RESULT_DIR = os.path.join(BASE_RESULT_DIR, mode)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


class CrossModalGatedAttentionMultiTaskModel(nn.Module):
    """Graph + descriptor + ECFP model with cross-modal gated attention fusion."""

    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.modalities = ("graph", "desc", "ecfp")
        self.graph_encoder = mt.DMPNNGraphEncoder()
        self.desc_encoder = nn.Sequential(
            nn.Linear(len(base.DESCRIPTOR_NAMES), 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(384, BRANCH_DIM),
            nn.ReLU(),
        )
        self.ecfp_encoder = nn.Sequential(
            nn.Linear(base.FP_SIZE, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, BRANCH_DIM),
            nn.ReLU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=BRANCH_DIM,
            num_heads=GATED_ATTENTION_HEADS,
            dropout=GATED_ATTENTION_DROPOUT,
            batch_first=True,
        )
        self.token_norm = nn.LayerNorm(BRANCH_DIM)
        self.gate = nn.Sequential(
            nn.Linear(BRANCH_DIM * 3, BRANCH_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(BRANCH_DIM, 3),
        )
        self.shared = nn.Sequential(
            nn.Linear(BRANCH_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.task_heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(num_tasks)])

    def forward(self, vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, task_idx, task_type_id):
        graph_feat = self.graph_encoder(
            atom_features,
            directed_bond_features,
            a2b,
            b2src,
            b2revb,
            bond_mask,
            atom_mask,
        )
        desc_feat = self.desc_encoder(vector[:, base.FP_SIZE : base.FP_INPUT_DIM])
        ecfp_feat = self.ecfp_encoder(vector[:, : base.FP_SIZE])
        tokens = torch.stack([graph_feat, desc_feat, ecfp_feat], dim=1)
        attended, _ = self.cross_attention(tokens, tokens, tokens, need_weights=False)
        attended = self.token_norm(tokens + attended)
        gate_weights = torch.softmax(self.gate(torch.cat([graph_feat, desc_feat, ecfp_feat], dim=1)), dim=1)
        fused = torch.sum(attended * gate_weights.unsqueeze(-1), dim=1)
        hidden = self.shared(fused)

        out = torch.zeros((hidden.size(0), 1), dtype=hidden.dtype, device=hidden.device)
        for task_id in task_idx.unique():
            mask = task_idx == task_id
            out[mask] = self.task_heads[int(task_id.item())](hidden[mask])
        return out


class CrossModalAttentionMultiTaskModel(nn.Module):
    """Graph + descriptor + ECFP model with cross-modal attention fusion."""

    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.modalities = ("graph", "desc", "ecfp")
        self.graph_encoder = mt.DMPNNGraphEncoder()
        self.desc_encoder = nn.Sequential(
            nn.Linear(len(base.DESCRIPTOR_NAMES), 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(384, BRANCH_DIM),
            nn.ReLU(),
        )
        self.ecfp_encoder = nn.Sequential(
            nn.Linear(base.FP_SIZE, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, BRANCH_DIM),
            nn.ReLU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=BRANCH_DIM,
            num_heads=GATED_ATTENTION_HEADS,
            dropout=GATED_ATTENTION_DROPOUT,
            batch_first=True,
        )
        self.token_norm = nn.LayerNorm(BRANCH_DIM)
        self.shared = nn.Sequential(
            nn.Linear(BRANCH_DIM * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.task_heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(num_tasks)])

    def forward(self, vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, task_idx, task_type_id):
        graph_feat = self.graph_encoder(
            atom_features,
            directed_bond_features,
            a2b,
            b2src,
            b2revb,
            bond_mask,
            atom_mask,
        )
        desc_feat = self.desc_encoder(vector[:, base.FP_SIZE : base.FP_INPUT_DIM])
        ecfp_feat = self.ecfp_encoder(vector[:, : base.FP_SIZE])
        tokens = torch.stack([graph_feat, desc_feat, ecfp_feat], dim=1)
        attended, _ = self.cross_attention(tokens, tokens, tokens, need_weights=False)
        attended = self.token_norm(tokens + attended)
        hidden = self.shared(attended.reshape(attended.size(0), -1))

        out = torch.zeros((hidden.size(0), 1), dtype=hidden.dtype, device=hidden.device)
        for task_id in task_idx.unique():
            mask = task_idx == task_id
            out[mask] = self.task_heads[int(task_id.item())](hidden[mask])
        return out


class CrossModalGatedMultiTaskModel(nn.Module):
    """Graph + descriptor + ECFP model with sample-specific gated fusion."""

    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.modalities = ("graph", "desc", "ecfp")
        self.graph_encoder = mt.DMPNNGraphEncoder()
        self.desc_encoder = nn.Sequential(
            nn.Linear(len(base.DESCRIPTOR_NAMES), 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(384, BRANCH_DIM),
            nn.ReLU(),
        )
        self.ecfp_encoder = nn.Sequential(
            nn.Linear(base.FP_SIZE, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, BRANCH_DIM),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(BRANCH_DIM * 3, BRANCH_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(BRANCH_DIM, 3),
        )
        self.shared = nn.Sequential(
            nn.Linear(BRANCH_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.task_heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(num_tasks)])

    def forward(self, vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, task_idx, task_type_id):
        graph_feat = self.graph_encoder(
            atom_features,
            directed_bond_features,
            a2b,
            b2src,
            b2revb,
            bond_mask,
            atom_mask,
        )
        desc_feat = self.desc_encoder(vector[:, base.FP_SIZE : base.FP_INPUT_DIM])
        ecfp_feat = self.ecfp_encoder(vector[:, : base.FP_SIZE])
        gate_weights = torch.softmax(self.gate(torch.cat([graph_feat, desc_feat, ecfp_feat], dim=1)), dim=1)
        tokens = torch.stack([graph_feat, desc_feat, ecfp_feat], dim=1)
        fused = torch.sum(tokens * gate_weights.unsqueeze(-1), dim=1)
        hidden = self.shared(fused)

        out = torch.zeros((hidden.size(0), 1), dtype=hidden.dtype, device=hidden.device)
        for task_id in task_idx.unique():
            mask = task_idx == task_id
            out[mask] = self.task_heads[int(task_id.item())](hidden[mask])
        return out


def make_model(num_tasks: int) -> nn.Module:
    if CURRENT_MODE == "mt_graph_desc_ecfp_concat":
        return mt.MultiModalMultiTaskModel(num_tasks, ("graph", "desc", "ecfp"))
    if CURRENT_MODE == "mt_graph_desc_ecfp_attention":
        return CrossModalAttentionMultiTaskModel(num_tasks)
    if CURRENT_MODE == "mt_graph_desc_ecfp_gated":
        return CrossModalGatedMultiTaskModel(num_tasks)
    if CURRENT_MODE == "mt_graph_desc_ecfp_gated_attention":
        return CrossModalGatedAttentionMultiTaskModel(num_tasks)
    raise ValueError(f"Unsupported CURRENT_MODE: {CURRENT_MODE}")


def preflight_checks() -> None:
    required_base_attrs = [
        "DESCRIPTOR_NAMES",
        "FP_SIZE",
        "FP_INPUT_DIM",
        "ATOM_FEATURE_DIM",
        "DMPNN_BOND_INPUT_DIM",
        "MAX_ATOMS",
        "MAX_DIRECTED_BONDS",
        "MAX_BONDS_PER_ATOM",
        "SingleTaskDataset",
        "get_task_preprocess",
        "logits_to_probs",
        "find_best_threshold",
        "calculate_classification_metrics",
        "calculate_task_metrics",
        "RegressionLoss",
    ]
    required_mt_attrs = [
        "DMPNNGraphEncoder",
        "MultiEndpointDataset",
        "MultiTaskLoss",
        "move_multitask_batch",
        "evaluate_predictions",
        "validation_score",
        "discover_filtered_tasks",
        "validate_task_splits",
    ]
    missing = [f"base.{name}" for name in required_base_attrs if not hasattr(base, name)]
    missing += [f"mt.{name}" for name in required_mt_attrs if not hasattr(mt, name)]
    if missing:
        raise RuntimeError("Missing required imported attributes: " + ", ".join(missing))

    if BRANCH_DIM % GATED_ATTENTION_HEADS != 0:
        raise RuntimeError(f"BRANCH_DIM={BRANCH_DIM} must be divisible by GATED_ATTENTION_HEADS={GATED_ATTENTION_HEADS}")

    for directory, label in [(MODEL_DIR, "MODEL_DIR"), (RESULT_DIR, "RESULT_DIR")]:
        os.makedirs(directory, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="preflight_", suffix=".tmp", dir=directory)
        os.close(fd)
        os.remove(path)
        if not os.access(directory, os.W_OK):
            raise RuntimeError(f"{label} is not writable: {directory}")

    for mode in FUSION_MODES:
        set_fusion_mode(mode)
        model = make_model(num_tasks=2).to(DEVICE)
        batch_size = 2
        vector = torch.zeros(batch_size, base.FP_INPUT_DIM, dtype=torch.float32, device=DEVICE)
        atom_features = torch.zeros(batch_size, base.MAX_ATOMS, base.ATOM_FEATURE_DIM, dtype=torch.float32, device=DEVICE)
        directed_bond_features = torch.zeros(
            batch_size,
            base.MAX_DIRECTED_BONDS + 1,
            base.DMPNN_BOND_INPUT_DIM,
            dtype=torch.float32,
            device=DEVICE,
        )
        a2b = torch.zeros(batch_size, base.MAX_ATOMS, base.MAX_BONDS_PER_ATOM, dtype=torch.long, device=DEVICE)
        b2src = torch.zeros(batch_size, base.MAX_DIRECTED_BONDS + 1, dtype=torch.long, device=DEVICE)
        b2revb = torch.zeros(batch_size, base.MAX_DIRECTED_BONDS + 1, dtype=torch.long, device=DEVICE)
        bond_mask = torch.zeros(batch_size, base.MAX_DIRECTED_BONDS + 1, dtype=torch.float32, device=DEVICE)
        atom_mask = torch.ones(batch_size, base.MAX_ATOMS, dtype=torch.float32, device=DEVICE)
        task_idx = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)
        task_type_id = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out = model(vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, task_idx, task_type_id)
        if tuple(out.shape) != (batch_size, 1):
            raise RuntimeError(f"{mode} returned unexpected output shape: {tuple(out.shape)}")
        test_model_path = os.path.join(MODEL_DIR, f"{current_model_kind()}_preflight_{SPLIT_LABEL}.pth")
        torch.save({"model_kind": current_model_kind(), "preflight": True}, test_model_path)
        os.remove(test_model_path)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


def train_one_seed(tasks, task_to_idx, seed: int):
    mt.seed_everything(seed)
    train_ds = mt.MultiEndpointDataset(tasks, task_to_idx, "train")
    val_ds = mt.MultiEndpointDataset(tasks, task_to_idx, "val")
    test_ds = mt.MultiEndpointDataset(tasks, task_to_idx, "test")

    sampler = None
    if USE_WEIGHTED_SAMPLER:
        sampler = WeightedRandomSampler(train_ds.sample_weights(), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, **mt.loader_kwargs(shuffle=sampler is None, sampler=sampler))
    val_loader = DataLoader(val_ds, **mt.loader_kwargs(shuffle=False))
    test_loader = DataLoader(test_ds, **mt.loader_kwargs(shuffle=False))

    model = make_model(len(task_to_idx)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=MT_LR, weight_decay=MT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    loss_fn = mt.MultiTaskLoss(train_ds)

    best_state = None
    best_score = -np.inf
    wait = 0
    history = []
    for epoch in range(MT_MAX_EPOCHS):
        train_loss = mt.run_epoch(model, train_loader, loss_fn, optimizer)
        val_preds, val_labels, val_task_indices, _ = mt.predict_dataset(model, val_loader)
        val_metrics = mt.evaluate_predictions(tasks, task_to_idx, val_preds, val_labels, val_task_indices, "val")
        score = mt.validation_score(val_metrics)
        scheduler.step(score)
        history.append(
            {
                "Seed": seed,
                "Mode": CURRENT_MODE,
                "Epoch": epoch,
                "TrainLoss": train_loss,
                "ValMeanScore": score,
            }
        )
        flush_print(f"mode={CURRENT_MODE} seed={seed} epoch={epoch:03d} train_loss={train_loss:.5f} val_mean_score={score:.5f}")
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
        if wait >= MT_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"No valid best model state for mode={CURRENT_MODE} seed={seed}")
    model.load_state_dict(best_state)
    val_preds, val_labels, val_task_indices, _ = mt.predict_dataset(model, val_loader)
    test_preds, test_labels, test_task_indices, _ = mt.predict_dataset(model, test_loader)
    val_metrics, thresholds = mt.evaluate_predictions(
        tasks,
        task_to_idx,
        val_preds,
        val_labels,
        val_task_indices,
        "val",
        return_thresholds=True,
    )
    test_metrics = mt.evaluate_predictions(
        tasks,
        task_to_idx,
        test_preds,
        test_labels,
        test_task_indices,
        "test",
        thresholds=thresholds,
    )

    model_path = os.path.join(MODEL_DIR, f"{current_model_kind()}_{SPLIT_LABEL}_seed{seed}.pth")
    torch.save(
        {
            "model_kind": current_model_kind(),
            "model_state_dict": model.state_dict(),
            "task_to_idx": task_to_idx,
            "tasks": tasks,
            "modalities": ("graph", "desc", "ecfp"),
            "fusion": CURRENT_MODE,
            "split_label": SPLIT_LABEL,
            "descriptor_names": base.DESCRIPTOR_NAMES,
            "fp_size": base.FP_SIZE,
            "branch_dim": BRANCH_DIM,
            "attention_heads": GATED_ATTENTION_HEADS if "attention" in CURRENT_MODE else None,
        },
        model_path,
    )
    return model, val_metrics, test_metrics, pd.DataFrame(history), model_path


def combine_seed_rows(seed_metrics, model_paths):
    rows = []
    for seed, val_metrics, test_metrics in seed_metrics:
        merged = val_metrics.merge(test_metrics, on=["Category", "Task", "TaskType"], suffixes=("_Val", "_Test"))
        merged.insert(0, "Seed", seed)
        merged.insert(0, "ModelKind", current_model_kind())
        merged.insert(0, "Mode", CURRENT_MODE)
        merged.insert(0, "SplitMode", SPLIT_LABEL)
        merged["SavedModels"] = model_paths.get(seed, "")
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_by_mode(summary):
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for (mode, model_kind), df in summary.groupby(["Mode", "ModelKind"]):
        clf = df[df["TaskType"] == "clf"]
        reg = df[df["TaskType"] == "reg"]
        rows.append(
            {
                "Mode": mode,
                "ModelKind": model_kind,
                "ClassificationTasks": clf[["Category", "Task"]].drop_duplicates().shape[0],
                "MCC_Val_Mean": clf["MCC_Val"].mean() if "MCC_Val" in clf else np.nan,
                "MCC_Test_Mean": clf["MCC_Test"].mean() if "MCC_Test" in clf else np.nan,
                "AUC_Test_Mean": clf["AUC_Test"].mean() if "AUC_Test" in clf else np.nan,
                "ACC_Test_Mean": clf["ACC_Test"].mean() if "ACC_Test" in clf else np.nan,
                "RegressionTasks": reg[["Category", "Task"]].drop_duplicates().shape[0],
                "R2_Val_Mean": reg["R2_Val"].mean() if "R2_Val" in reg else np.nan,
                "R2_Test_Mean": reg["R2_Test"].mean() if "R2_Test" in reg else np.nan,
                "Pearson_Test_Mean": reg["Pearson_Test"].mean() if "Pearson_Test" in reg else np.nan,
                "Spearman_Test_Mean": reg["Spearman_Test"].mean() if "Spearman_Test" in reg else np.nan,
                "MAE_Test_Mean": reg["MAE_Test"].mean() if "MAE_Test" in reg else np.nan,
                "RMSE_Test_Mean": reg["RMSE_Test"].mean() if "RMSE_Test" in reg else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_mode():
    flush_print(f"Mode: {CURRENT_MODE}; split={SPLIT_LABEL}; data={DATA_ROOT}; device={DEVICE}")
    tasks = mt.discover_filtered_tasks()
    if not tasks:
        raise RuntimeError(f"No endpoints discovered under DATA_ROOT={DATA_ROOT}")
    tasks, skipped = mt.validate_task_splits(tasks)
    if MAX_TASKS > 0:
        tasks = tasks[:MAX_TASKS]
    skipped.to_csv(os.path.join(RESULT_DIR, "skipped_endpoints.csv"), index=False)
    if not tasks:
        raise RuntimeError(f"No valid endpoints remained after split validation under DATA_ROOT={DATA_ROOT}")
    flush_print(f"Using {len(tasks)} valid endpoints; skipped {len(skipped)} endpoints.")
    pd.DataFrame(tasks).drop(columns=["SplitFiles"], errors="ignore").to_csv(
        os.path.join(RESULT_DIR, "discovered_endpoints.csv"),
        index=False,
    )
    task_to_idx = {mt.task_key(task_cfg): idx for idx, task_cfg in enumerate(tasks)}

    seed_metrics = []
    model_paths = {}
    histories = []
    for seed in MT_ENSEMBLE_SEEDS:
        model, val_metrics, test_metrics, history, model_path = train_one_seed(tasks, task_to_idx, seed)
        seed_metrics.append((seed, val_metrics, test_metrics))
        model_paths[seed] = model_path
        histories.append(history)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    summary = combine_seed_rows(seed_metrics, model_paths)
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    summary.to_csv(os.path.join(RESULT_DIR, "all_task_metrics_summary.csv"), index=False)
    history_df.to_csv(os.path.join(RESULT_DIR, "training_history.csv"), index=False)
    summarize_by_mode(summary).to_csv(os.path.join(RESULT_DIR, "mode_summary.csv"), index=False)
    flush_print(f"Wrote results to {RESULT_DIR}")
    return summary



def parse_runtime_args() -> None:
    global DATA_ROOT, BASE_MODEL_DIR, BASE_RESULT_DIR, FUSION_MODES, MT_ENSEMBLE_SEEDS
    global MT_MAX_EPOCHS, MT_PATIENCE, MT_BATCH_SIZE, MAX_TASKS, MODEL_DIR, RESULT_DIR
    parser = argparse.ArgumentParser(description="Train PharmTAP multimodal multitask ADMET fusion ablation models.")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Directory containing scaffold-split ADMET endpoint folders.")
    parser.add_argument("--model-dir", default=BASE_MODEL_DIR, help="Directory for trained model checkpoints.")
    parser.add_argument("--result-dir", default=BASE_RESULT_DIR, help="Directory for metric tables and prediction outputs.")
    parser.add_argument("--fusion-modes", default=",".join(FUSION_MODES), help="Comma-separated fusion modes to train.")
    parser.add_argument("--seeds", default=",".join(str(s) for s in MT_ENSEMBLE_SEEDS), help="Comma-separated random seeds.")
    parser.add_argument("--max-epochs", type=int, default=MT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=MT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=MT_BATCH_SIZE)
    parser.add_argument("--max-tasks", type=int, default=MAX_TASKS, help="Optional debug limit; 0 uses all endpoints.")
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    BASE_MODEL_DIR = args.model_dir
    BASE_RESULT_DIR = args.result_dir
    FUSION_MODES = [item.strip() for item in args.fusion_modes.split(",") if item.strip()]
    invalid = [mode for mode in FUSION_MODES if mode not in VALID_FUSION_MODES]
    if invalid:
        raise ValueError(f"Invalid --fusion-modes: {invalid}; valid={sorted(VALID_FUSION_MODES)}")
    MT_ENSEMBLE_SEEDS = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    MT_MAX_EPOCHS = args.max_epochs
    MT_PATIENCE = args.patience
    MT_BATCH_SIZE = args.batch_size
    MAX_TASKS = args.max_tasks
    MODEL_DIR = BASE_MODEL_DIR
    RESULT_DIR = BASE_RESULT_DIR
    base.DATA_ROOT = DATA_ROOT
    mt.DATA_ROOT = DATA_ROOT
    mt.MT_MAX_EPOCHS = MT_MAX_EPOCHS
    mt.MT_PATIENCE = MT_PATIENCE
    mt.MT_BATCH_SIZE = MT_BATCH_SIZE
    mt.MT_ENSEMBLE_SEEDS = MT_ENSEMBLE_SEEDS

def run_all_modes():
    parse_runtime_args()
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)
    preflight_checks()
    flush_print("Preflight checks passed.")
    if "--preflight" in sys.argv:
        return

    all_summaries = []
    for mode in FUSION_MODES:
        try:
            set_fusion_mode(mode)
            summary = run_mode()
            all_summaries.append(summary)
        except Exception:
            if STOP_ON_ERROR:
                raise
            flush_print(f"ERROR while running mode={mode}",)
    combined = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    combined.to_csv(os.path.join(BASE_RESULT_DIR, "all_fusion_modes_task_metrics_summary.csv"), index=False)
    summarize_by_mode(combined).to_csv(os.path.join(BASE_RESULT_DIR, "all_fusion_modes_summary.csv"), index=False)
    flush_print(f"Wrote combined fusion results to {BASE_RESULT_DIR}")


if __name__ == "__main__":
    run_all_modes()

