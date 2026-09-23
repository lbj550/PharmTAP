#!/usr/bin/env python3
"""
Independent single-task graph + RDKit descriptor + ECFP concat baselines for ADMET endpoints.

Matched single-task counterpart of admet_multimodal_core.py::mt_graph_desc_ecfp_concat.
It uses the same split files, molecular featurization, preprocessing, metrics, and validation-threshold
logic, but trains one independent model per endpoint instead of sharing a multitask encoder.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Importing the modality-ablation module reuses its DMPNN branch and its parallel
# featurization patch for admet_graph_desc_baseline.
import admet_multimodal_core as mt
import admet_graph_desc_baseline as base

DATA_ROOT = os.getenv("DATA_ROOT", base.DATA_ROOT)
DATA_ROOT_LABEL = Path(DATA_ROOT).name.replace("data_", "") or "original"
ENV_SPLIT_LABEL = os.getenv("SPLIT_MODE", "")
SPLIT_LABEL = DATA_ROOT_LABEL if ENV_SPLIT_LABEL in ("", "original") and DATA_ROOT_LABEL != "data" else ENV_SPLIT_LABEL or DATA_ROOT_LABEL
MODEL_DIR = os.getenv("MODEL_DIR", "data/admet/ADMET_multimodal_single_task_gde_model_pth")
RESULT_DIR = os.getenv("RESULT_DIR", "data/admet/ADMET_multimodal_single_task_gde_results")
MODEL_KIND = "ADMET_Multimodal_SingleTask_Graph_Desc_ECFP_Concat"
DEVICE = mt.DEVICE
BRANCH_DIM = mt.BRANCH_DIM

ST_ENSEMBLE_SEEDS = [
    int(item.strip())
    for item in os.getenv("ST_ENSEMBLE_SEEDS", os.getenv("MT_ENSEMBLE_SEEDS", "42")).split(",")
    if item.strip()
]
ST_MAX_EPOCHS = int(os.getenv("ST_MAX_EPOCHS", os.getenv("MT_MAX_EPOCHS", "120")))
ST_PATIENCE = int(os.getenv("ST_PATIENCE", os.getenv("MT_PATIENCE", "18")))
ST_BATCH_SIZE = int(os.getenv("ST_BATCH_SIZE", os.getenv("BATCH_SIZE", "128")))
ST_NUM_WORKERS = int(os.getenv("ST_NUM_WORKERS", os.getenv("NUM_WORKERS", "0")))
CLASS_LR = float(os.getenv("ST_CLASS_LR", "3e-4"))
REG_LR = float(os.getenv("ST_REG_LR", "2e-4"))
CLASS_WEIGHT_DECAY = float(os.getenv("ST_CLASS_WEIGHT_DECAY", "4e-4"))
REG_WEIGHT_DECAY = float(os.getenv("ST_REG_WEIGHT_DECAY", "8e-4"))
STOP_ON_ERROR = os.getenv("ST_STOP_ON_ERROR", "1").lower() not in {"0", "false", "no"}
MAX_TASKS = int(os.getenv("ST_MAX_TASKS", "0"))

base.DATA_ROOT = DATA_ROOT
mt.DATA_ROOT = DATA_ROOT
base.BATCH_SIZE = ST_BATCH_SIZE
base.DATALOADER_WORKERS = ST_NUM_WORKERS


def flush_print(message: str) -> None:
    print(message, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_kwargs(shuffle: bool) -> dict:
    return base.loader_kwargs(shuffle=shuffle)


def safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return text or "endpoint"


def preflight_checks() -> None:
    required_base_attrs = [
        "DATA_ROOT",
        "DEVICE",
        "DESCRIPTOR_NAMES",
        "FP_SIZE",
        "FP_INPUT_DIM",
        "ATOM_FEATURE_DIM",
        "DMPNN_BOND_INPUT_DIM",
        "MAX_ATOMS",
        "MAX_DIRECTED_BONDS",
        "MAX_BONDS_PER_ATOM",
        "SingleTaskDataset",
        "loader_kwargs",
        "get_loss_fn",
        "logits_to_probs",
        "find_best_threshold",
        "calculate_classification_metrics",
        "calculate_task_metrics",
    ]
    required_mt_attrs = ["DEVICE", "BRANCH_DIM", "DMPNNGraphEncoder", "discover_filtered_tasks", "validate_task_splits"]
    missing = [f"base.{name}" for name in required_base_attrs if not hasattr(base, name)]
    missing += [f"mt.{name}" for name in required_mt_attrs if not hasattr(mt, name)]
    if missing:
        raise RuntimeError("Missing required imported attributes: " + ", ".join(missing))

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    for directory, label in [(MODEL_DIR, "MODEL_DIR"), (RESULT_DIR, "RESULT_DIR")]:
        fd, path = tempfile.mkstemp(prefix="preflight_", suffix=".tmp", dir=directory)
        os.close(fd)
        os.remove(path)
        if not os.access(directory, os.W_OK):
            raise RuntimeError(f"{label} is not writable: {directory}")

    model = GraphDescECFPSingleTaskModel().to(DEVICE)
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
    with torch.no_grad():
        out = model(vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask)
    if tuple(out.shape) != (batch_size, 1):
        raise RuntimeError(f"Unexpected model output shape from preflight forward pass: {tuple(out.shape)}")

    probs = base.logits_to_probs(np.array([0.0, 1.0], dtype=np.float32))
    threshold = base.find_best_threshold(np.array([0, 1], dtype=np.float32), probs)
    base.calculate_classification_metrics(np.array([0, 1], dtype=np.float32), probs, threshold)
    base.calculate_task_metrics(
        "reg",
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        0.0,
        1.0,
        {"name": "standard", "shift": 0.0},
    )

    test_model_path = os.path.join(MODEL_DIR, f"{MODEL_KIND}_preflight_{safe_filename(SPLIT_LABEL)}.pth")
    torch.save({"model_kind": MODEL_KIND, "preflight": True}, test_model_path)
    os.remove(test_model_path)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


class GraphDescECFPSingleTaskModel(nn.Module):
    """Single-task counterpart of mt_graph_desc_ecfp_concat."""

    def __init__(self):
        super().__init__()
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
        self.head = nn.Sequential(
            nn.Linear(BRANCH_DIM * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask):
        graph_feat = self.graph_encoder(atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask)
        desc_feat = self.desc_encoder(vector[:, base.FP_SIZE : base.FP_INPUT_DIM])
        ecfp_feat = self.ecfp_encoder(vector[:, : base.FP_SIZE])
        return self.head(torch.cat([graph_feat, desc_feat, ecfp_feat], dim=1))


def move_batch(batch):
    vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, y = batch
    return (
        vector.to(DEVICE, non_blocking=True),
        atom_features.to(DEVICE, non_blocking=True),
        directed_bond_features.to(DEVICE, non_blocking=True),
        a2b.to(DEVICE, non_blocking=True),
        b2src.to(DEVICE, non_blocking=True),
        b2revb.to(DEVICE, non_blocking=True),
        bond_mask.to(DEVICE, non_blocking=True),
        atom_mask.to(DEVICE, non_blocking=True),
        y.to(DEVICE, non_blocking=True),
    )


def run_epoch(model, loader, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses = []
    for batch in loader:
        vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, y = move_batch(batch)
        pred = model(vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask)
        loss = loss_fn(pred, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan


@torch.no_grad()
def predict_dataset(model, loader):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, y = move_batch(batch)
        pred = model(vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask)
        preds.append(pred.detach().cpu().numpy().reshape(-1))
        labels.append(y.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(labels)


def make_datasets(task_cfg):
    train_ds = base.SingleTaskDataset(task_cfg, "train")
    val_ds = base.SingleTaskDataset(task_cfg, "val", train_ds.y_mean, train_ds.y_std, train_ds.active_descriptors, train_ds.target_transform, train_ds.descriptor_scaler)
    test_ds = base.SingleTaskDataset(task_cfg, "test", train_ds.y_mean, train_ds.y_std, train_ds.active_descriptors, train_ds.target_transform, train_ds.descriptor_scaler)
    return train_ds, val_ds, test_ds


def evaluate_task(task_cfg, preds, labels, dataset, split_name, threshold=None):
    if task_cfg["TaskType"] == "clf":
        probs = base.logits_to_probs(preds)
        if threshold is None:
            threshold = base.find_best_threshold(labels, probs) if split_name == "val" else 0.5
        return base.calculate_classification_metrics(labels, probs, threshold), threshold
    return base.calculate_task_metrics("reg", preds, labels, dataset.y_mean, dataset.y_std, dataset.target_transform), None


def model_selection_score(task_type, metrics):
    score = metrics.get("AUC", np.nan) if task_type == "clf" else metrics.get("R2", np.nan)
    return -np.inf if pd.isna(score) else float(score)


def train_one_task_seed(task_cfg, seed):
    seed_everything(seed)
    train_ds, val_ds, test_ds = make_datasets(task_cfg)
    train_loader = DataLoader(train_ds, **loader_kwargs(shuffle=True))
    val_loader = DataLoader(val_ds, **loader_kwargs(shuffle=False))
    test_loader = DataLoader(test_ds, **loader_kwargs(shuffle=False))

    model = GraphDescECFPSingleTaskModel().to(DEVICE)
    if task_cfg["TaskType"] == "clf":
        optimizer = optim.AdamW(model.parameters(), lr=CLASS_LR, weight_decay=CLASS_WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=REG_LR, weight_decay=REG_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    loss_fn = base.get_loss_fn(task_cfg["TaskType"], train_ds)

    best_state = None
    best_score = -np.inf
    wait = 0
    history = []
    for epoch in range(ST_MAX_EPOCHS):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_preds, val_labels = predict_dataset(model, val_loader)
        val_metrics, _ = evaluate_task(task_cfg, val_preds, val_labels, val_ds, "val")
        score = model_selection_score(task_cfg["TaskType"], val_metrics)
        scheduler.step(score)
        history.append({"Seed": seed, "Category": task_cfg["Category"], "Task": task_cfg["Task"], "TaskType": task_cfg["TaskType"], "Epoch": epoch, "TrainLoss": train_loss, "ValScore": score})
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
        if wait >= ST_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No valid validation score was produced; best_state is empty.")
    model.load_state_dict(best_state)
    val_preds, val_labels = predict_dataset(model, val_loader)
    test_preds, test_labels = predict_dataset(model, test_loader)
    val_metrics, threshold = evaluate_task(task_cfg, val_preds, val_labels, val_ds, "val")
    test_metrics, _ = evaluate_task(task_cfg, test_preds, test_labels, test_ds, "test", threshold=threshold)

    safe_category = safe_filename(task_cfg["Category"])
    safe_task = safe_filename(task_cfg["Task"])
    model_path = os.path.join(MODEL_DIR, f"{MODEL_KIND}_{safe_category}_{safe_task}_{SPLIT_LABEL}_seed{seed}.pth")
    torch.save({
        "model_kind": MODEL_KIND,
        "model_state_dict": model.state_dict(),
        "task_cfg": {k: v for k, v in task_cfg.items() if k != "SplitFiles"},
        "split_label": SPLIT_LABEL,
        "seed": seed,
        "active_descriptors": train_ds.active_descriptors,
        "descriptor_scaler": train_ds.descriptor_scaler,
        "target_transform": train_ds.target_transform,
        "y_mean": train_ds.y_mean,
        "y_std": train_ds.y_std,
        "descriptor_names": base.DESCRIPTOR_NAMES,
        "fp_size": base.FP_SIZE,
        "branch_dim": BRANCH_DIM,
        "threshold": threshold,
    }, model_path)

    row = {"SplitMode": SPLIT_LABEL, "Mode": "single_graph_desc_ecfp_concat", "ModelKind": MODEL_KIND, "Seed": seed, "Category": task_cfg["Category"], "Task": task_cfg["Task"], "TaskType": task_cfg["TaskType"], "Split_Val": "val"}
    row.update({f"{k}_Val": v for k, v in val_metrics.items()})
    row["Split_Test"] = "test"
    row.update({f"{k}_Test": v for k, v in test_metrics.items()})
    row["SavedModels"] = model_path
    return row, pd.DataFrame(history)


def summarize(summary):
    clf = summary[summary["TaskType"] == "clf"]
    reg = summary[summary["TaskType"] == "reg"]
    return pd.DataFrame([{
        "Mode": "single_graph_desc_ecfp_concat",
        "ClassificationTasks": len(clf),
        "MCC_Val_Mean": clf["MCC_Val"].mean() if "MCC_Val" in clf else np.nan,
        "MCC_Test_Mean": clf["MCC_Test"].mean() if "MCC_Test" in clf else np.nan,
        "AUC_Test_Mean": clf["AUC_Test"].mean() if "AUC_Test" in clf else np.nan,
        "RegressionTasks": len(reg),
        "R2_Val_Mean": reg["R2_Val"].mean() if "R2_Val" in reg else np.nan,
        "R2_Test_Mean": reg["R2_Test"].mean() if "R2_Test" in reg else np.nan,
        "Pearson_Test_Mean": reg["Pearson_Test"].mean() if "Pearson_Test" in reg else np.nan,
        "MAE_Test_Mean": reg["MAE_Test"].mean() if "MAE_Test" in reg else np.nan,
    }])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    flush_print(f"Mode: single_graph_desc_ecfp_concat; split={SPLIT_LABEL}; data={DATA_ROOT}; device={DEVICE}")
    featurize_workers = getattr(mt, "FEATURIZE_WORKERS", os.getenv("FEATURIZE_WORKERS", "module-default"))
    featurize_progress = getattr(mt, "FEATURIZE_PROGRESS_EVERY", os.getenv("FEATURIZE_PROGRESS_EVERY", "module-default"))
    flush_print(f"Featurization workers: {featurize_workers}; progress every {featurize_progress} molecules")
    preflight_checks()
    flush_print("Preflight checks passed.")
    if "--preflight" in sys.argv:
        return
    tasks = mt.discover_filtered_tasks()
    tasks, skipped = mt.validate_task_splits(tasks)
    if MAX_TASKS > 0:
        tasks = tasks[:MAX_TASKS]
    pd.DataFrame(tasks).drop(columns=["SplitFiles"], errors="ignore").to_csv(os.path.join(RESULT_DIR, "discovered_endpoints.csv"), index=False)
    skipped.to_csv(os.path.join(RESULT_DIR, "skipped_endpoints.csv"), index=False)
    if not tasks:
        raise RuntimeError(f"No valid endpoints found under DATA_ROOT={DATA_ROOT}")
    flush_print(f"Using {len(tasks)} valid endpoints; skipped {len(skipped)} endpoints.")

    rows, histories = [], []
    total = len(tasks) * len(ST_ENSEMBLE_SEEDS)
    done = 0
    for seed in ST_ENSEMBLE_SEEDS:
        for task_cfg in tasks:
            done += 1
            flush_print(f"[{done}/{total}] seed={seed} {task_cfg['Category']} / {task_cfg['Task']} ({task_cfg['TaskType']})")
            try:
                row, history = train_one_task_seed(task_cfg, seed)
                rows.append(row)
                histories.append(history)
            except Exception as exc:
                flush_print(f"ERROR seed={seed} {task_cfg['Category']} / {task_cfg['Task']}: {exc}")
                rows.append({"SplitMode": SPLIT_LABEL, "Mode": "single_graph_desc_ecfp_concat", "ModelKind": MODEL_KIND, "Seed": seed, "Category": task_cfg["Category"], "Task": task_cfg["Task"], "TaskType": task_cfg["TaskType"], "Error": repr(exc)})
                if STOP_ON_ERROR:
                    raise
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    summary.to_csv(os.path.join(RESULT_DIR, "all_task_metrics_summary.csv"), index=False)
    history_df.to_csv(os.path.join(RESULT_DIR, "training_history.csv"), index=False)
    summarize(summary).to_csv(os.path.join(RESULT_DIR, "mode_summary.csv"), index=False)
    flush_print(f"Wrote results to {RESULT_DIR}")


if __name__ == "__main__":
    main()

