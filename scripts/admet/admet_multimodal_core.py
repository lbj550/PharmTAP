import copy
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


os.environ.setdefault("DATA_ROOT", "data/admet/data_scaffold_8_1_1")
os.environ.setdefault("MODEL_DIR", "data/admet/ADMET_multimodal_modality_ablation_model_pth")
os.environ.setdefault("RESULT_DIR", "data/admet/ADMET_multimodal_modality_ablation_results")
os.environ.setdefault("SPLIT_MODE", "original")

import admet_graph_desc_baseline as base


warnings.filterwarnings("ignore")

DEVICE = base.DEVICE
DATA_ROOT = base.DATA_ROOT
BASE_MODEL_DIR = os.getenv("MODEL_DIR", "data/admet/ADMET_multimodal_modality_ablation_model_pth")
BASE_RESULT_DIR = os.getenv("RESULT_DIR", "data/admet/ADMET_multimodal_modality_ablation_results")
MODEL_KIND = "ADMET_Multimodal_Modality_Ablation"
SPLIT_LABEL = os.getenv("SPLIT_LABEL", "scaffold_8_1_1")

ENDPOINT_EMBED_DIM = int(os.getenv("ENDPOINT_EMBED_DIM", "64"))
BRANCH_DIM = int(os.getenv("BRANCH_DIM", "256"))
FEATURIZE_WORKERS = int(os.getenv("FEATURIZE_WORKERS", str(os.cpu_count() or 1)))
FEATURIZE_PROGRESS_EVERY = int(os.getenv("FEATURIZE_PROGRESS_EVERY", "500"))
MT_BATCH_SIZE = int(os.getenv("MT_BATCH_SIZE", os.getenv("BATCH_SIZE", "96")))
MT_MAX_EPOCHS = int(os.getenv("MT_MAX_EPOCHS", os.getenv("MAX_EPOCHS", "120")))
MT_PATIENCE = int(os.getenv("MT_PATIENCE", "18"))
MT_LR = float(os.getenv("MT_LR", "2e-4"))
MT_WEIGHT_DECAY = float(os.getenv("MT_WEIGHT_DECAY", "6e-4"))
USE_WEIGHTED_SAMPLER = os.getenv("USE_WEIGHTED_SAMPLER", "1") == "1"
MT_ENSEMBLE_SEEDS = [
    int(item.strip())
    for item in os.getenv("MT_ENSEMBLE_SEEDS", "42").split(",")
    if item.strip()
]

MODALITY_MODES = {
    "mt_graph_only": ("graph",),
    "mt_desc_only": ("desc",),
    "mt_ecfp_only": ("ecfp",),
    "mt_graph_desc": ("graph", "desc"),
    "mt_graph_ecfp": ("graph", "ecfp"),
    "mt_desc_ecfp": ("desc", "ecfp"),
    "mt_graph_desc_ecfp_concat": ("graph", "desc", "ecfp"),
}
ABLATION_MODES = [
    item.strip().lower()
    for item in os.getenv(
        "ABLATION_MODES",
        "mt_graph_only,mt_desc_only,mt_ecfp_only,mt_graph_desc,mt_graph_ecfp,mt_desc_ecfp,mt_graph_desc_ecfp_concat",
    ).split(",")
    if item.strip()
]
invalid_modes = [mode for mode in ABLATION_MODES if mode not in MODALITY_MODES]
if invalid_modes:
    raise ValueError(f"Invalid ABLATION_MODES: {invalid_modes}; valid={sorted(MODALITY_MODES)}")

CURRENT_MODE = ABLATION_MODES[0]
MODEL_DIR = BASE_MODEL_DIR
RESULT_DIR = BASE_RESULT_DIR


def _flush_print(message):
    print(message, flush=True)


def _featurize_one_smiles(args):
    idx, smiles, label = args
    features = base.MoleculeFeaturizer.smiles_to_raw_features(smiles)
    if features is None:
        return idx, None, None
    return idx, features, float(label)


def parallel_build_split_feature_cache(df, smiles_col):
    total = len(df)
    worker_count = max(1, min(FEATURIZE_WORKERS, total if total else 1))
    _flush_print(f"    featurizing {total} molecules with {worker_count} CPU worker(s)")

    tasks = [
        (idx, row[smiles_col], row["label"])
        for idx, row in df.reset_index(drop=True).iterrows()
    ]
    fps, descriptors, atom_features, directed_bond_features = [], [], [], []
    a2b, b2src, b2revb, bond_mask, atom_mask, labels = [], [], [], [], [], []
    invalid_count = 0
    processed = 0
    started = time.time()

    def consume_result(result):
        nonlocal invalid_count, processed
        _, features, label = result
        processed += 1
        if features is None:
            invalid_count += 1
        else:
            fp_arr, descriptor_arr, atom_arr, directed_bond_arr, a2b_arr, b2src_arr, b2revb_arr, bond_mask_arr, mask_arr = features
            fps.append(fp_arr)
            descriptors.append(descriptor_arr)
            atom_features.append(atom_arr)
            directed_bond_features.append(directed_bond_arr)
            a2b.append(a2b_arr)
            b2src.append(b2src_arr)
            b2revb.append(b2revb_arr)
            bond_mask.append(bond_mask_arr)
            atom_mask.append(mask_arr)
            labels.append(label)
        if processed == total or processed % max(1, FEATURIZE_PROGRESS_EVERY) == 0:
            elapsed = max(time.time() - started, 1e-6)
            rate = processed / elapsed
            _flush_print(f"      processed {processed}/{total}; valid={len(labels)} invalid={invalid_count}; {rate:.1f} mol/s")

    if worker_count <= 1:
        for item in tasks:
            consume_result(_featurize_one_smiles(item))
    else:
        chunksize = max(1, min(128, total // (worker_count * 4) if total else 1))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for result in executor.map(_featurize_one_smiles, tasks, chunksize=chunksize):
                consume_result(result)

    if not labels:
        raise ValueError("No valid molecules found after SMILES featurization.")
    return {
        "fps": np.stack(fps).astype(np.float32, copy=False),
        "descriptors": np.stack(descriptors).astype(np.float32, copy=False),
        "atom_features": np.stack(atom_features).astype(np.float32, copy=False),
        "directed_bond_features": np.stack(directed_bond_features).astype(np.float32, copy=False),
        "a2b": np.stack(a2b).astype(np.int64, copy=False),
        "b2src": np.stack(b2src).astype(np.int64, copy=False),
        "b2revb": np.stack(b2revb).astype(np.int64, copy=False),
        "bond_mask": np.stack(bond_mask).astype(np.float32, copy=False),
        "atom_mask": np.stack(atom_mask).astype(np.float32, copy=False),
        "labels": np.asarray(labels, dtype=np.float32),
        "invalid_count": invalid_count,
    }


base.build_split_feature_cache = parallel_build_split_feature_cache


def set_ablation_mode(mode):
    global CURRENT_MODE, MODEL_DIR, RESULT_DIR
    if mode not in MODALITY_MODES:
        raise ValueError(f"Invalid mode: {mode}")
    CURRENT_MODE = mode
    MODEL_DIR = os.path.join(BASE_MODEL_DIR, mode)
    RESULT_DIR = os.path.join(BASE_RESULT_DIR, mode)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def current_model_kind():
    return f"{MODEL_KIND}_{CURRENT_MODE}"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = DEVICE.type == "cuda"


def task_key(task_cfg):
    return f"{task_cfg['Category']}::{task_cfg['Task']}"


def discover_filtered_tasks():
    tasks = base.discover_tasks(DATA_ROOT)
    if base.TASK_CATEGORIES:
        allowed = {category.lower() for category in base.TASK_CATEGORIES}
        tasks = [task for task in tasks if task["Category"].lower() in allowed]
    if base.TASK_FILTER:
        tasks = [
            task
            for task in tasks
            if base.TASK_FILTER in task["Category"].lower()
            or base.TASK_FILTER in task["Task"].lower()
        ]
    return tasks


def validate_task_splits(tasks):
    valid_tasks = []
    skipped_rows = []
    for task_no, task_cfg in enumerate(tasks, 1):
        sizes = {}
        try:
            _flush_print(f"[validate {task_no}/{len(tasks)}] {task_cfg.get('Category')} / {task_cfg.get('Task')}")
            for split in ("train", "val", "test"):
                _flush_print(f"  loading {split} split")
                ds = base.SingleTaskDataset(task_cfg, split)
                sizes[f"{split}_n"] = len(ds)
                if len(ds) == 0:
                    raise ValueError(f"{split} split has zero valid molecules")
            valid_tasks.append(task_cfg)
        except Exception as exc:
            skipped_rows.append(
                {
                    "Category": task_cfg.get("Category"),
                    "Task": task_cfg.get("Task"),
                    "TaskType": task_cfg.get("TaskType"),
                    "Reason": str(exc),
                    **sizes,
                }
            )
            try:
                base.clear_task_caches(task_cfg)
            except Exception:
                pass
            _flush_print(f"Skipping endpoint with invalid split: {task_cfg.get('Category')} / {task_cfg.get('Task')} :: {exc}")
    return valid_tasks, pd.DataFrame(skipped_rows)


class MultiEndpointDataset(Dataset):
    def __init__(self, tasks, task_to_idx, split):
        self.tasks = tasks
        self.task_to_idx = task_to_idx
        self.split = split
        self.datasets = []
        self.samples = []
        for task_cfg in tasks:
            ds = base.SingleTaskDataset(task_cfg, split)
            ds_idx = len(self.datasets)
            task_idx = task_to_idx[task_key(task_cfg)]
            task_type_id = 0 if task_cfg["TaskType"] == "clf" else 1
            self.datasets.append(ds)
            for local_idx in range(len(ds)):
                self.samples.append((ds_idx, local_idx, task_idx, task_type_id))

    def __len__(self):
        return len(self.samples)

    def sample_weights(self):
        counts = {}
        for _, _, task_idx, _ in self.samples:
            counts[task_idx] = counts.get(task_idx, 0) + 1
        return torch.DoubleTensor([1.0 / counts[task_idx] for _, _, task_idx, _ in self.samples])

    def __getitem__(self, idx):
        ds_idx, local_idx, task_idx, task_type_id = self.samples[idx]
        item = self.datasets[ds_idx][local_idx]
        return item + (
            torch.tensor(task_idx, dtype=torch.long),
            torch.tensor(task_type_id, dtype=torch.long),
        )


def move_multitask_batch(batch):
    (
        vector,
        atom_features,
        directed_bond_features,
        a2b,
        b2src,
        b2revb,
        bond_mask,
        atom_mask,
        y,
        task_idx,
        task_type_id,
    ) = batch
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
        task_idx.to(DEVICE, non_blocking=True),
        task_type_id.to(DEVICE, non_blocking=True),
    )


class DMPNNGraphEncoder(nn.Module):
    def __init__(self, hidden_dim=256, depth=4, dropout=0.15):
        super().__init__()
        self.depth = depth
        self.input_layer = nn.Linear(base.DMPNN_BOND_INPUT_DIM, hidden_dim)
        self.message_layer = nn.Linear(hidden_dim, hidden_dim)
        self.atom_output = nn.Sequential(
            nn.Linear(base.ATOM_FEATURE_DIM + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask):
        h0 = torch.relu(self.input_layer(directed_bond_features))
        h = h0 * bond_mask.unsqueeze(-1)
        for _ in range(max(0, self.depth - 1)):
            incoming = h.gather(
                1,
                a2b.unsqueeze(-1).expand(-1, -1, -1, h.size(-1)).reshape(h.size(0), -1, h.size(-1)),
            ).reshape(h.size(0), base.MAX_ATOMS, base.MAX_BONDS_PER_ATOM, h.size(-1)).sum(dim=2)
            source_messages = incoming.gather(1, b2src.unsqueeze(-1).expand(-1, -1, h.size(-1)))
            reverse_messages = h.gather(1, b2revb.unsqueeze(-1).expand(-1, -1, h.size(-1)))
            message = source_messages - reverse_messages
            h = torch.relu(h0 + self.message_layer(message))
            h = self.dropout(h) * bond_mask.unsqueeze(-1)

        incoming = h.gather(
            1,
            a2b.unsqueeze(-1).expand(-1, -1, -1, h.size(-1)).reshape(h.size(0), -1, h.size(-1)),
        ).reshape(h.size(0), base.MAX_ATOMS, base.MAX_BONDS_PER_ATOM, h.size(-1)).sum(dim=2)
        atom_hidden = self.atom_output(torch.cat([atom_features, incoming], dim=-1))
        atom_hidden = atom_hidden * atom_mask.unsqueeze(-1)
        denom = atom_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return atom_hidden.sum(dim=1) / denom


class MultiModalMultiTaskModel(nn.Module):
    def __init__(self, num_tasks, modalities):
        super().__init__()
        self.num_tasks = num_tasks
        self.modalities = tuple(modalities)
        self.graph_encoder = DMPNNGraphEncoder() if "graph" in self.modalities else None
        self.desc_encoder = (
            nn.Sequential(
                nn.Linear(len(base.DESCRIPTOR_NAMES), 384),
                nn.LayerNorm(384),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(384, BRANCH_DIM),
                nn.ReLU(),
            )
            if "desc" in self.modalities
            else None
        )
        self.ecfp_encoder = (
            nn.Sequential(
                nn.Linear(base.FP_SIZE, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(512, BRANCH_DIM),
                nn.ReLU(),
            )
            if "ecfp" in self.modalities
            else None
        )
        fusion_dim = BRANCH_DIM * len(self.modalities)
        self.shared = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.task_heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(num_tasks)])

    def forward(self, vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, task_idx, task_type_id):
        features = []
        if "graph" in self.modalities:
            features.append(
                self.graph_encoder(
                    atom_features,
                    directed_bond_features,
                    a2b,
                    b2src,
                    b2revb,
                    bond_mask,
                    atom_mask,
                )
            )
        if "desc" in self.modalities:
            features.append(self.desc_encoder(vector[:, base.FP_SIZE : base.FP_INPUT_DIM]))
        if "ecfp" in self.modalities:
            features.append(self.ecfp_encoder(vector[:, : base.FP_SIZE]))
        hidden = self.shared(torch.cat(features, dim=1))
        out = torch.zeros((hidden.size(0), 1), dtype=hidden.dtype, device=hidden.device)
        for task_id in task_idx.unique():
            mask = task_idx == task_id
            out[mask] = self.task_heads[int(task_id.item())](hidden[mask])
        return out


class MultiTaskLoss(nn.Module):
    def __init__(self, train_dataset):
        super().__init__()
        pos_weight_by_task = {}
        for ds_idx, local_idx, task_idx, task_type_id in train_dataset.samples:
            if task_type_id != 0:
                continue
            label = int(train_dataset.datasets[ds_idx].y[local_idx].item())
            counts = pos_weight_by_task.setdefault(task_idx, [0, 0])
            counts[label] += 1
        self.pos_weight_by_task = {
            task_idx: float(np.clip(neg / max(pos, 1), 0.2, 8.0))
            for task_idx, (neg, pos) in pos_weight_by_task.items()
        }
        self.regression_loss = base.RegressionLoss()

    def forward(self, pred, target, task_idx, task_type_id):
        pred = pred.view(-1)
        target = target.view(-1)
        losses = []
        clf_mask = task_type_id == 0
        if clf_mask.any():
            clf_pred = pred[clf_mask]
            clf_target = target[clf_mask]
            clf_tasks = task_idx[clf_mask]
            weights = torch.ones_like(clf_target)
            for task_id, pos_weight in self.pos_weight_by_task.items():
                weights[clf_tasks == task_id] = pos_weight
            losses.append(
                nn.functional.binary_cross_entropy_with_logits(
                    clf_pred,
                    clf_target,
                    pos_weight=weights,
                )
            )
        reg_mask = task_type_id == 1
        if reg_mask.any():
            losses.append(self.regression_loss(pred[reg_mask], target[reg_mask]))
        if not losses:
            return pred.sum() * 0.0
        return sum(losses) / len(losses)


def loader_kwargs(shuffle=False, sampler=None):
    kwargs = {
        "batch_size": MT_BATCH_SIZE,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "pin_memory": DEVICE.type == "cuda",
        "num_workers": base.DATALOADER_WORKERS,
    }
    if base.DATALOADER_WORKERS > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def run_epoch(model, loader, loss_fn, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    for batch in loader:
        batch = move_multitask_batch(batch)
        vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, y, task_idx, task_type_id = batch
        if is_train:
            optimizer.zero_grad()
        pred = model(vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, task_idx, task_type_id)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=30.0, neginf=-30.0)
        loss = loss_fn(pred, y, task_idx, task_type_id)
        if not torch.isfinite(loss):
            continue
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else np.nan


def predict_dataset(model, loader):
    model.eval()
    preds, labels, task_indices, task_type_ids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_multitask_batch(batch)
            vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, y, task_idx, task_type_id = batch
            pred = model(vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, task_idx, task_type_id)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=30.0, neginf=-30.0)
            preds.append(pred.cpu().numpy().reshape(-1))
            labels.append(y.cpu().numpy().reshape(-1))
            task_indices.append(task_idx.cpu().numpy().reshape(-1))
            task_type_ids.append(task_type_id.cpu().numpy().reshape(-1))
    return (
        np.concatenate(preds),
        np.concatenate(labels),
        np.concatenate(task_indices),
        np.concatenate(task_type_ids),
    )


def evaluate_predictions(tasks, task_to_idx, preds, labels, task_indices, split_name, thresholds=None, return_thresholds=False):
    thresholds = thresholds or {}
    learned_thresholds = {}
    idx_to_task = {idx: task for task, idx in [(task_cfg, task_to_idx[task_key(task_cfg)]) for task_cfg in tasks]}
    rows = []
    for task_idx in sorted(idx_to_task):
        task_cfg = idx_to_task[task_idx]
        mask = task_indices == task_idx
        if not np.any(mask):
            continue
        preprocess = base.get_task_preprocess(task_cfg)
        task_preds = preds[mask]
        task_labels = labels[mask]
        if task_cfg["TaskType"] == "clf":
            probs = base.logits_to_probs(task_preds)
            key = task_key(task_cfg)
            threshold = thresholds.get(key)
            if threshold is None:
                threshold = base.find_best_threshold(task_labels, probs) if split_name == "val" else 0.5
            learned_thresholds[key] = threshold
            metrics = base.calculate_classification_metrics(task_labels, probs, threshold)
        else:
            metrics = base.calculate_task_metrics(
                "reg",
                task_preds,
                task_labels,
                preprocess["y_mean"],
                preprocess["y_std"],
                preprocess["target_transform"],
            )
        row = {
            "Category": task_cfg["Category"],
            "Task": task_cfg["Task"],
            "TaskType": task_cfg["TaskType"],
            "Split": split_name,
        }
        row.update(metrics)
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    if return_thresholds:
        return metrics_df, learned_thresholds
    return metrics_df


def validation_score(metrics_df):
    scores = []
    for _, row in metrics_df.iterrows():
        score = row["AUC"] if row["TaskType"] == "clf" else row["R2"]
        if pd.notna(score):
            scores.append(float(score))
    return float(np.mean(scores)) if scores else -np.inf


def train_one_seed(tasks, task_to_idx, seed):
    seed_everything(seed)
    train_ds = MultiEndpointDataset(tasks, task_to_idx, "train")
    val_ds = MultiEndpointDataset(tasks, task_to_idx, "val")
    test_ds = MultiEndpointDataset(tasks, task_to_idx, "test")
    sampler = None
    if USE_WEIGHTED_SAMPLER:
        sampler = WeightedRandomSampler(train_ds.sample_weights(), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, **loader_kwargs(shuffle=sampler is None, sampler=sampler))
    val_loader = DataLoader(val_ds, **loader_kwargs(shuffle=False))
    test_loader = DataLoader(test_ds, **loader_kwargs(shuffle=False))

    model = MultiModalMultiTaskModel(len(task_to_idx), MODALITY_MODES[CURRENT_MODE]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=MT_LR, weight_decay=MT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    loss_fn = MultiTaskLoss(train_ds)

    best_state = None
    best_score = -np.inf
    wait = 0
    history = []
    for epoch in range(MT_MAX_EPOCHS):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_preds, val_labels, val_task_indices, _ = predict_dataset(model, val_loader)
        val_metrics = evaluate_predictions(tasks, task_to_idx, val_preds, val_labels, val_task_indices, "val")
        score = validation_score(val_metrics)
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
        print(f"mode={CURRENT_MODE} seed={seed} epoch={epoch:03d} train_loss={train_loss:.5f} val_mean_score={score:.5f}")
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
        if wait >= MT_PATIENCE:
            break

    model.load_state_dict(best_state)
    val_preds, val_labels, val_task_indices, _ = predict_dataset(model, val_loader)
    test_preds, test_labels, test_task_indices, _ = predict_dataset(model, test_loader)
    val_metrics, thresholds = evaluate_predictions(
        tasks,
        task_to_idx,
        val_preds,
        val_labels,
        val_task_indices,
        "val",
        return_thresholds=True,
    )
    test_metrics = evaluate_predictions(
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
            "modalities": MODALITY_MODES[CURRENT_MODE],
            "split_label": SPLIT_LABEL,
            "descriptor_names": base.DESCRIPTOR_NAMES,
            "fp_size": base.FP_SIZE,
            "branch_dim": BRANCH_DIM,
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
                "RegressionTasks": reg[["Category", "Task"]].drop_duplicates().shape[0],
                "R2_Val_Mean": reg["R2_Val"].mean() if "R2_Val" in reg else np.nan,
                "R2_Test_Mean": reg["R2_Test"].mean() if "R2_Test" in reg else np.nan,
                "Pearson_Test_Mean": reg["Pearson_Test"].mean() if "Pearson_Test" in reg else np.nan,
                "MAE_Test_Mean": reg["MAE_Test"].mean() if "MAE_Test" in reg else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_mode():
    _flush_print(f"Mode: {CURRENT_MODE}; modalities={MODALITY_MODES[CURRENT_MODE]}; split={SPLIT_LABEL}; data={DATA_ROOT}")
    _flush_print(f"Featurization workers: {FEATURIZE_WORKERS}; progress every {FEATURIZE_PROGRESS_EVERY} molecules")
    tasks = discover_filtered_tasks()
    if not tasks:
        raise RuntimeError(f"No endpoints discovered under DATA_ROOT={DATA_ROOT}")
    tasks, skipped = validate_task_splits(tasks)
    skipped.to_csv(os.path.join(RESULT_DIR, "skipped_endpoints.csv"), index=False)
    if not tasks:
        raise RuntimeError(f"No valid endpoints remained after split validation under DATA_ROOT={DATA_ROOT}")
    _flush_print(f"Using {len(tasks)} valid endpoints; skipped {len(skipped)} endpoints.")
    task_to_idx = {task_key(task_cfg): idx for idx, task_cfg in enumerate(tasks)}
    pd.DataFrame(tasks).drop(columns=["SplitFiles"], errors="ignore").to_csv(
        os.path.join(RESULT_DIR, "discovered_endpoints.csv"),
        index=False,
    )

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
    print(f"Wrote results to {RESULT_DIR}")
    return summary


def run_all_modes():
    all_summaries = []
    for mode in ABLATION_MODES:
        set_ablation_mode(mode)
        summary = run_mode()
        all_summaries.append(summary)
    combined = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)
    combined.to_csv(os.path.join(BASE_RESULT_DIR, "all_modes_task_metrics_summary.csv"), index=False)
    summarize_by_mode(combined).to_csv(os.path.join(BASE_RESULT_DIR, "all_modes_summary.csv"), index=False)
    print(f"Wrote combined results to {BASE_RESULT_DIR}")


if __name__ == "__main__":
    run_all_modes()

