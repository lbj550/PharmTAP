import os
import pickle
import random
import warnings
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")

DATA_ROOT = os.getenv("DATA_ROOT", "data/admet/data")
MODEL_DIR = os.getenv("MODEL_DIR", "data/admet/DMPNN_Desc_single_task_model_pth")
RESULT_DIR = os.getenv("RESULT_DIR", "data/admet/DMPNN_Desc_single_task_results")

FP_SIZE = int(os.getenv("FP_SIZE", "2048"))
MAX_ATOMS = int(os.getenv("MAX_ATOMS", "128"))
MAX_DIRECTED_BONDS = int(os.getenv("MAX_DIRECTED_BONDS", "320"))
MAX_BONDS_PER_ATOM = int(os.getenv("MAX_BONDS_PER_ATOM", "8"))
ATOM_FEATURE_DIM = 48
BOND_FEATURE_DIM = 16
DMPNN_BOND_INPUT_DIM = ATOM_FEATURE_DIM + BOND_FEATURE_DIM
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
DATALOADER_WORKERS = int(os.getenv("DATALOADER_WORKERS", "4"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "160"))
PATIENCE = int(os.getenv("PATIENCE", "18"))
REGRESSION_PATIENCE = int(os.getenv("REGRESSION_PATIENCE", "28"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "3"))
ACTIVE_FOLD_INDEX = 0
ACTIVE_FOLDS_TEXT = os.getenv("ACTIVE_FOLDS", "0").strip()
VALIDATION_FRACTION = float(os.getenv("VALIDATION_FRACTION", "0.15"))
ENSEMBLE_TOP_K = int(os.getenv("ENSEMBLE_TOP_K", "3"))
MAX_ACTIVE_DESCRIPTOR_COUNT = int(os.getenv("MAX_ACTIVE_DESCRIPTOR_COUNT", "96"))
DESCRIPTOR_EPS = 1e-6
DESCRIPTOR_CLIP_VALUE = float(os.getenv("DESCRIPTOR_CLIP_VALUE", "10.0"))
TASK_FILTER = os.getenv("TASK_FILTER", "").strip().lower()
TASK_CATEGORIES = [
    item.strip()
    for item in os.getenv("TASK_CATEGORIES", "").split(",")
    if item.strip()
]
MODEL_FILTER = [
    item.strip()
    for item in os.getenv("MODEL_FILTER", "DMPNN_Desc").split(",")
    if item.strip()
]
SPLIT_MODE = os.getenv("SPLIT_MODE", "cv").strip().lower()
SPLIT_LABEL = f"{CV_FOLDS}fold_cv" if SPLIT_MODE == "cv" else "original_split"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


UNSTABLE_DESCRIPTOR_NAMES = {
    "MaxAbsPartialCharge",
    "MaxPartialCharge",
    "MinAbsPartialCharge",
    "MinPartialCharge",
}
DESCRIPTOR_FUNCTIONS = {
    name: fn
    for name, fn in Descriptors._descList
    if name not in UNSTABLE_DESCRIPTOR_NAMES
}
DESCRIPTOR_FUNCTIONS.update(
    {
        "FormalCharge": lambda mol: Chem.GetFormalCharge(mol),
        "AromaticRings": lambda mol: Descriptors.NumAromaticRings(mol),
    }
)
DESCRIPTOR_NAMES = sorted(DESCRIPTOR_FUNCTIONS.keys())
FP_INPUT_DIM = FP_SIZE + len(DESCRIPTOR_NAMES)


MODEL_VARIANTS = {
    "DMPNN_Desc": {
        "group": "Baseline",
        "architecture": "dpnn_desc",
        "use_fp": False,
        "use_descriptors": True,
    },
}

ACTIVE_MODEL_KINDS = [name for name in MODEL_FILTER if name in MODEL_VARIANTS]
if not ACTIVE_MODEL_KINDS:
    raise ValueError(f"MODEL_FILTER did not match any model variants: {MODEL_FILTER}")


def parse_active_folds(text):
    if not text or text.lower() == "all":
        return list(range(CV_FOLDS))
    folds = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        fold_idx = int(part)
        if fold_idx < 0 or fold_idx >= CV_FOLDS:
            raise ValueError(f"ACTIVE_FOLDS contains invalid fold {fold_idx}; expected 0 <= fold < {CV_FOLDS}")
        folds.append(fold_idx)
    return sorted(set(folds))


ACTIVE_FOLDS = parse_active_folds(ACTIVE_FOLDS_TEXT)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = DEVICE.type == "cuda"


def print_runtime_info():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"CUDA device count visible to PyTorch: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"BATCH_SIZE: {BATCH_SIZE}; DATALOADER_WORKERS: {DATALOADER_WORKERS}")


def safe_name(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("%", "pct")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def discover_tasks(data_root=DATA_ROOT):
    tasks = []
    for category in sorted(os.listdir(data_root)):
        category_path = os.path.join(data_root, category)
        if not os.path.isdir(category_path):
            continue
        for dirpath, _, files in os.walk(category_path):
            csvs = [f for f in files if f.lower().endswith(".csv")]
            if not csvs:
                continue
            frames = []
            split_files = {}
            for csv_name in csvs:
                path = os.path.join(dirpath, csv_name)
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                if "label" not in df.columns:
                    continue
                frames.append(df)
                lower = csv_name.lower()
                if "_train" in lower:
                    split_files["train"] = path
                elif "_val" in lower:
                    split_files["val"] = path
                elif "_test" in lower:
                    split_files["test"] = path
            if not frames:
                continue
            labels = pd.to_numeric(pd.concat(frames, ignore_index=True)["label"], errors="coerce").dropna()
            if labels.empty:
                continue
            unique = set(labels.unique().tolist())
            task_type = "clf" if unique.issubset({0, 1, 0.0, 1.0}) and len(unique) <= 2 else "reg"
            task_name = os.path.relpath(dirpath, category_path)
            tasks.append(
                {
                    "Task": task_name,
                    "Category": category,
                    "TaskType": task_type,
                    "Folder": dirpath,
                    "SplitFiles": split_files,
                }
            )
    return tasks


def result_key(row):
    return (
        str(row["Category"]),
        str(row["Task"]),
        int(row["Fold"]),
    )


def result_file_candidates():
    patterns = [
        os.path.join(RESULT_DIR, "all_task_metrics_summary.csv"),
        os.path.join(RESULT_DIR, "all_task_metrics_summary_partial*.csv"),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob(pattern))
    return sorted(set(paths))


def load_existing_results():
    rows = []
    seen = set()
    expected_models = ",".join(ACTIVE_MODEL_KINDS)
    for path in result_file_candidates():
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        required = {"Category", "Task", "Fold"}
        if not required.issubset(df.columns):
            continue
        for _, row in df.iterrows():
            if pd.isna(row.get("Category")) or pd.isna(row.get("Task")) or pd.isna(row.get("Fold")):
                continue
            if str(row.get("SplitMode", "")) != SPLIT_LABEL:
                continue
            if str(row.get("TrainedModels", "")) != expected_models:
                continue
            key = result_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row.to_dict())
    return rows, seen


def calculate_descriptor_values(mol):
    values = {}
    for name, fn in DESCRIPTOR_FUNCTIONS.items():
        try:
            value = float(fn(mol))
        except Exception:
            value = 0.0
        values[name] = value if np.isfinite(value) else 0.0
    return values


def choose_target_transform(labels):
    labels = np.asarray(labels, dtype=np.float32)
    identity = {"name": "standard", "shift": 0.0}
    if len(labels) < 5 or np.min(labels) <= -0.999:
        return identity
    raw_skew = abs(pd.Series(labels).skew())
    shifted = labels - labels.min()
    log_skew = abs(pd.Series(np.log1p(shifted)).skew())
    if np.isfinite(log_skew) and log_skew + 0.05 < raw_skew:
        return {"name": "log1p_shift", "shift": float(labels.min())}
    return identity


def apply_target_transform(values, transform):
    values = np.asarray(values, dtype=np.float32)
    if transform["name"] == "log1p_shift":
        return np.log1p(values - transform["shift"])
    return values


def inverse_target_transform(values, transform):
    values = np.asarray(values, dtype=np.float32)
    if transform["name"] == "log1p_shift":
        return np.expm1(values) + transform["shift"]
    return values


def denormalize_labels(values, y_mean, y_std, target_transform=None):
    transformed = np.asarray(values, dtype=np.float32) * y_std + y_mean
    return inverse_target_transform(transformed, target_transform or {"name": "standard", "shift": 0.0})


def make_stratify_labels(labels, task_type):
    labels = np.asarray(labels, dtype=np.float32)
    if len(labels) < 4:
        return None
    if task_type == "clf":
        labels_int = labels.astype(int)
        _, counts = np.unique(labels_int, return_counts=True)
        return labels_int if len(counts) > 1 and counts.min() >= 2 else None
    bin_count = min(8, max(2, len(labels) // 20))
    try:
        bins = pd.qcut(labels, q=bin_count, labels=False, duplicates="drop")
    except ValueError:
        return None
    if pd.isna(bins).any() or len(np.unique(bins)) < 2:
        return None
    _, counts = np.unique(np.asarray(bins), return_counts=True)
    return np.asarray(bins) if counts.min() >= 2 else None


def split_train_val(indices, labels, task_type, random_state):
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 3:
        return indices, indices[:0]
    rng = np.random.default_rng(random_state)
    val_size = max(1, int(round(len(indices) * VALIDATION_FRACTION)))
    val_size = min(val_size, len(indices) - 1)
    stratify = make_stratify_labels(labels[indices], task_type)
    if stratify is None:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        return shuffled[val_size:], shuffled[:val_size]

    val_indices = []
    train_indices = []
    for group in np.unique(stratify):
        group_indices = indices[stratify == group].copy()
        rng.shuffle(group_indices)
        group_val_size = max(1, int(round(len(group_indices) * val_size / len(indices))))
        group_val_size = min(group_val_size, len(group_indices) - 1) if len(group_indices) > 1 else 0
        val_indices.extend(group_indices[:group_val_size].tolist())
        train_indices.extend(group_indices[group_val_size:].tolist())
    if not val_indices:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        return shuffled[val_size:], shuffled[:val_size]
    return np.asarray(train_indices, dtype=int), np.asarray(val_indices, dtype=int)


def load_task_dataframes(task_cfg):
    split_files = task_cfg["SplitFiles"]
    required_splits = ["train", "val", "test"]
    missing = [split for split in required_splits if split not in split_files]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} split files for {task_cfg['Category']} / {task_cfg['Task']}"
        )

    split_frames = {}
    smiles_col = None
    for split in required_splits:
        df = pd.read_csv(split_files[split])
        if "label" not in df.columns:
            raise ValueError(f"Missing label column in {split_files[split]}")
        current_smiles_col = "smiles_standarized" if "smiles_standarized" in df.columns else df.columns[0]
        if smiles_col is None:
            smiles_col = current_smiles_col
        elif current_smiles_col != smiles_col:
            df = df.rename(columns={current_smiles_col: smiles_col})
        split_frames[split] = (
            df[[smiles_col, "label"]]
            .dropna()
            .drop_duplicates(subset=[smiles_col, "label"])
            .reset_index(drop=True)
        )
    return split_frames, smiles_col


def load_task_dataframes_for_cv(task_cfg):
    original_splits, smiles_col = load_task_dataframes(task_cfg)
    df = pd.concat(original_splits.values(), ignore_index=True)
    df = df.drop_duplicates(subset=[smiles_col, "label"]).reset_index(drop=True)
    labels = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=np.float32)
    indices = np.arange(len(df))
    stratify = make_stratify_labels(labels, task_cfg["TaskType"])
    folds = [[] for _ in range(CV_FOLDS)]
    rng = np.random.default_rng(42)

    if stratify is None:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        folds = [fold.tolist() for fold in np.array_split(shuffled, CV_FOLDS)]
    else:
        for group in np.unique(stratify):
            group_indices = indices[stratify == group].copy()
            rng.shuffle(group_indices)
            for fold_idx, fold_values in enumerate(np.array_split(group_indices, CV_FOLDS)):
                folds[fold_idx].extend(fold_values.tolist())

    test_fold = ACTIVE_FOLD_INDEX % CV_FOLDS
    test_indices = np.asarray(folds[test_fold], dtype=int)
    pool_indices = np.asarray(
        [idx for idx, fold in enumerate(folds) if idx != test_fold for idx in fold],
        dtype=int,
    )
    train_indices, val_indices = split_train_val(
        pool_indices,
        labels,
        task_cfg["TaskType"],
        random_state=142 + ACTIVE_FOLD_INDEX,
    )
    split_frames = {
        "train": df.iloc[train_indices].reset_index(drop=True),
        "val": df.iloc[val_indices].reset_index(drop=True),
        "test": df.iloc[test_indices].reset_index(drop=True),
    }
    return split_frames, smiles_col


SPLIT_CACHE = {}


def split_task_dataframe(task_cfg):
    cache_key = (task_cfg["Category"], task_cfg["Task"], SPLIT_MODE, ACTIVE_FOLD_INDEX)
    if cache_key in SPLIT_CACHE:
        return SPLIT_CACHE[cache_key]

    if SPLIT_MODE == "cv":
        SPLIT_CACHE[cache_key] = load_task_dataframes_for_cv(task_cfg)
    elif SPLIT_MODE == "original":
        SPLIT_CACHE[cache_key] = load_task_dataframes(task_cfg)
    else:
        raise ValueError("SPLIT_MODE must be 'cv' or 'original'")
    return SPLIT_CACHE[cache_key]


def select_data_driven_descriptors_from_array(task_cfg, labels, descriptor_values):
    labels = np.asarray(labels, dtype=np.float32)
    scores = {}
    for idx, name in enumerate(DESCRIPTOR_NAMES):
        values = descriptor_values[:, idx].astype(np.float32, copy=False)
        if np.std(values) < 1e-8:
            scores[name] = 0.0
            continue
        if task_cfg["TaskType"] == "clf":
            pos = values[labels == 1]
            neg = values[labels == 0]
            if len(pos) == 0 or len(neg) == 0:
                scores[name] = 0.0
            else:
                scores[name] = abs(np.mean(pos) - np.mean(neg)) / (np.std(values) + DESCRIPTOR_EPS)
        else:
            corr = np.corrcoef(values, labels)[0, 1]
            scores[name] = abs(corr) if np.isfinite(corr) else 0.0
    ranked = [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return ranked[:MAX_ACTIVE_DESCRIPTOR_COUNT]


def fit_descriptor_scaler_from_array(descriptor_values, active_descriptors):
    mean, std = {}, {}
    name_to_idx = {name: idx for idx, name in enumerate(DESCRIPTOR_NAMES)}
    for name in active_descriptors:
        values = descriptor_values[:, name_to_idx[name]].astype(np.float32, copy=False)
        mean[name] = float(np.mean(values))
        scale = float(np.std(values))
        std[name] = scale if scale > DESCRIPTOR_EPS else 1.0
    return {"mean": mean, "std": std}


TASK_FEATURE_CACHE = {}
TASK_PREPROCESS_CACHE = {}


def task_cache_key(task_cfg):
    return (str(task_cfg["Category"]), str(task_cfg["Task"]))


def build_split_feature_cache(df, smiles_col):
    fps, descriptors, atom_features, directed_bond_features = [], [], [], []
    a2b, b2src, b2revb, bond_mask, atom_mask, labels = [], [], [], [], [], []
    invalid_count = 0
    for _, row in df.iterrows():
        features = MoleculeFeaturizer.smiles_to_raw_features(row[smiles_col])
        if features is None:
            invalid_count += 1
            continue
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
        labels.append(float(row["label"]))
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


def get_task_feature_cache(task_cfg):
    key = task_cache_key(task_cfg)
    if key in TASK_FEATURE_CACHE:
        return TASK_FEATURE_CACHE[key]
    split_frames, smiles_col = split_task_dataframe(task_cfg)
    split_features = {
        split: build_split_feature_cache(df, smiles_col)
        for split, df in split_frames.items()
    }
    TASK_FEATURE_CACHE[key] = {
        "splits": split_features,
        "smiles_col": smiles_col,
        "invalid_count": sum(item["invalid_count"] for item in split_features.values()),
    }
    return TASK_FEATURE_CACHE[key]


def get_task_preprocess(task_cfg):
    key = task_cache_key(task_cfg)
    if key in TASK_PREPROCESS_CACHE:
        return TASK_PREPROCESS_CACHE[key]
    train_features = get_task_feature_cache(task_cfg)["splits"]["train"]
    labels = train_features["labels"]
    active_descriptors = select_data_driven_descriptors_from_array(
        task_cfg,
        labels,
        train_features["descriptors"],
    )
    descriptor_scaler = fit_descriptor_scaler_from_array(train_features["descriptors"], active_descriptors)
    target_transform = {"name": "standard", "shift": 0.0}
    y_mean = None
    y_std = None
    if task_cfg["TaskType"] == "reg":
        target_transform = choose_target_transform(labels)
        transformed = apply_target_transform(labels, target_transform)
        y_mean = float(np.mean(transformed))
        y_std = float(np.std(transformed))
        if y_std < DESCRIPTOR_EPS:
            y_std = 1.0
    TASK_PREPROCESS_CACHE[key] = {
        "active_descriptors": active_descriptors,
        "descriptor_scaler": descriptor_scaler,
        "target_transform": target_transform,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return TASK_PREPROCESS_CACHE[key]


def clear_task_caches(task_cfg):
    key = task_cache_key(task_cfg)
    TASK_FEATURE_CACHE.pop(key, None)
    TASK_PREPROCESS_CACHE.pop(key, None)
    for split_key in list(SPLIT_CACHE):
        if split_key[0] == task_cfg["Category"] and split_key[1] == task_cfg["Task"]:
            SPLIT_CACHE.pop(split_key, None)


class MoleculeFeaturizer:
    @staticmethod
    def atom_to_features(atom):
        symbols = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H", "B", "Si", "Se"]
        hybrids = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]
        features = []
        features.extend([float(atom.GetSymbol() == symbol) for symbol in symbols])
        features.append(float(atom.GetSymbol() not in symbols))
        features.extend([float(atom.GetHybridization() == hybrid) for hybrid in hybrids])
        features.extend(
            [
                atom.GetAtomicNum() / 100,
                atom.GetTotalDegree() / 6,
                atom.GetFormalCharge() / 5,
                atom.GetTotalNumHs() / 4,
                atom.GetImplicitValence() / 6,
                atom.GetTotalValence() / 8,
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                atom.GetMass() / 200,
            ]
        )
        features.extend([0.0] * max(0, ATOM_FEATURE_DIM - len(features)))
        return np.asarray(features[:ATOM_FEATURE_DIM], dtype=np.float32)

    @staticmethod
    def bond_to_features(bond):
        if bond is None:
            return np.zeros(BOND_FEATURE_DIM, dtype=np.float32)
        bond_type = bond.GetBondType()
        features = [
            float(bond_type == Chem.rdchem.BondType.SINGLE),
            float(bond_type == Chem.rdchem.BondType.DOUBLE),
            float(bond_type == Chem.rdchem.BondType.TRIPLE),
            float(bond_type == Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            float(bond.GetStereo() == Chem.rdchem.BondStereo.STEREONONE),
            float(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOANY),
            float(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ),
            float(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE),
        ]
        features.extend([0.0] * max(0, BOND_FEATURE_DIM - len(features)))
        return np.asarray(features[:BOND_FEATURE_DIM], dtype=np.float32)

    @staticmethod
    def smiles_to_raw_features(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_SIZE)
        fp_arr = np.zeros((FP_SIZE,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, fp_arr)

        descriptor_values = calculate_descriptor_values(mol)
        descriptors = np.asarray(
            [
                descriptor_values[name] if np.isfinite(descriptor_values[name]) else 0.0
                for name in DESCRIPTOR_NAMES
            ],
            dtype=np.float32,
        )

        atom_features = np.zeros((MAX_ATOMS, ATOM_FEATURE_DIM), dtype=np.float32)
        atom_mask = np.zeros(MAX_ATOMS, dtype=np.float32)
        directed_bond_features = np.zeros((MAX_DIRECTED_BONDS + 1, DMPNN_BOND_INPUT_DIM), dtype=np.float32)
        a2b = np.zeros((MAX_ATOMS, MAX_BONDS_PER_ATOM), dtype=np.int64)
        b2src = np.zeros((MAX_DIRECTED_BONDS + 1,), dtype=np.int64)
        b2revb = np.zeros((MAX_DIRECTED_BONDS + 1,), dtype=np.int64)
        bond_mask = np.zeros((MAX_DIRECTED_BONDS + 1,), dtype=np.float32)
        atom_bond_counts = np.zeros((MAX_ATOMS,), dtype=np.int64)

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx >= MAX_ATOMS:
                continue
            atom_features[idx] = MoleculeFeaturizer.atom_to_features(atom)
            atom_mask[idx] = 1.0

        directed_index = 1
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin >= MAX_ATOMS or end >= MAX_ATOMS:
                continue
            if directed_index + 1 > MAX_DIRECTED_BONDS:
                continue
            feat = MoleculeFeaturizer.bond_to_features(bond)
            forward_idx = directed_index
            reverse_idx = directed_index + 1
            directed_index += 2

            directed_bond_features[forward_idx] = np.concatenate([atom_features[begin], feat])
            directed_bond_features[reverse_idx] = np.concatenate([atom_features[end], feat])
            b2src[forward_idx] = begin
            b2src[reverse_idx] = end
            b2revb[forward_idx] = reverse_idx
            b2revb[reverse_idx] = forward_idx
            bond_mask[forward_idx] = 1.0
            bond_mask[reverse_idx] = 1.0

            if atom_bond_counts[end] < MAX_BONDS_PER_ATOM:
                a2b[end, atom_bond_counts[end]] = forward_idx
                atom_bond_counts[end] += 1
            if atom_bond_counts[begin] < MAX_BONDS_PER_ATOM:
                a2b[begin, atom_bond_counts[begin]] = reverse_idx
                atom_bond_counts[begin] += 1

        return fp_arr, descriptors, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask

    @staticmethod
    def build_vector(fp_arr, descriptor_arr, active_descriptors=None, descriptor_scaler=None):
        active_descriptors = set(active_descriptors or DESCRIPTOR_NAMES)
        descriptors = np.zeros((len(DESCRIPTOR_NAMES),), dtype=np.float32)
        for idx, name in enumerate(DESCRIPTOR_NAMES):
            if name not in active_descriptors:
                continue
            value = descriptor_arr[idx]
            if descriptor_scaler is not None:
                mean = descriptor_scaler.get("mean", {}).get(name, 0.0)
                std = descriptor_scaler.get("std", {}).get(name, 1.0)
                value = (value - mean) / std
            descriptors[idx] = value if np.isfinite(value) else 0.0
        return np.concatenate([fp_arr, descriptors]).astype(np.float32, copy=False)

    @staticmethod
    def smiles_to_features(smiles, active_descriptors=None, descriptor_scaler=None):
        raw_features = MoleculeFeaturizer.smiles_to_raw_features(smiles)
        if raw_features is None:
            return None
        fp_arr, descriptors, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask = raw_features
        vector = MoleculeFeaturizer.build_vector(fp_arr, descriptors, active_descriptors, descriptor_scaler)
        return vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask


class SingleTaskDataset(Dataset):
    def __init__(
        self,
        task_cfg,
        split,
        y_mean=None,
        y_std=None,
        active_descriptors=None,
        target_transform=None,
        descriptor_scaler=None,
    ):
        self.task_cfg = task_cfg
        self.task_type = task_cfg["TaskType"]
        self.split = split
        self.y_mean = y_mean
        self.y_std = y_std
        self.active_descriptors = active_descriptors
        self.target_transform = target_transform or {"name": "standard", "shift": 0.0}
        self.descriptor_scaler = descriptor_scaler

        preprocess = get_task_preprocess(task_cfg)
        if self.active_descriptors is None:
            self.active_descriptors = preprocess["active_descriptors"]
        if self.descriptor_scaler is None:
            self.descriptor_scaler = preprocess["descriptor_scaler"]
        if target_transform is None:
            self.target_transform = preprocess["target_transform"]
        if self.task_type == "reg":
            if self.y_mean is None:
                self.y_mean = preprocess["y_mean"]
            if self.y_std is None:
                self.y_std = preprocess["y_std"]

        cached_split = get_task_feature_cache(task_cfg)["splits"][split]
        descriptor_matrix = np.zeros_like(cached_split["descriptors"], dtype=np.float32)
        name_to_idx = {name: idx for idx, name in enumerate(DESCRIPTOR_NAMES)}
        for name in self.active_descriptors:
            idx = name_to_idx[name]
            values = cached_split["descriptors"][:, idx]
            mean = self.descriptor_scaler.get("mean", {}).get(name, 0.0)
            std = self.descriptor_scaler.get("std", {}).get(name, 1.0)
            descriptor_matrix[:, idx] = (values - mean) / std
        descriptor_matrix = np.nan_to_num(
            descriptor_matrix,
            nan=0.0,
            posinf=DESCRIPTOR_CLIP_VALUE,
            neginf=-DESCRIPTOR_CLIP_VALUE,
        )
        descriptor_matrix = np.clip(descriptor_matrix, -DESCRIPTOR_CLIP_VALUE, DESCRIPTOR_CLIP_VALUE)
        vectors = np.concatenate([cached_split["fps"], descriptor_matrix], axis=1).astype(np.float32, copy=False)
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=DESCRIPTOR_CLIP_VALUE, neginf=-DESCRIPTOR_CLIP_VALUE)

        labels = cached_split["labels"].astype(np.float32, copy=True)
        if self.task_type == "reg":
            labels = apply_target_transform(labels, self.target_transform)
            labels = (labels - self.y_mean) / self.y_std

        self.vector = torch.from_numpy(vectors)
        self.atom_features = torch.from_numpy(cached_split["atom_features"])
        self.directed_bond_features = torch.from_numpy(cached_split["directed_bond_features"])
        self.a2b = torch.from_numpy(cached_split["a2b"])
        self.b2src = torch.from_numpy(cached_split["b2src"])
        self.b2revb = torch.from_numpy(cached_split["b2revb"])
        self.bond_mask = torch.from_numpy(cached_split["bond_mask"])
        self.atom_mask = torch.from_numpy(cached_split["atom_mask"])
        self.y = torch.from_numpy(labels.reshape(-1, 1).astype(np.float32, copy=False))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (
            self.vector[idx],
            self.atom_features[idx],
            self.directed_bond_features[idx],
            self.a2b[idx],
            self.b2src[idx],
            self.b2revb[idx],
            self.bond_mask[idx],
            self.atom_mask[idx],
            self.y[idx],
        )


class VectorModel(nn.Module):
    def __init__(self, use_fp=True, use_descriptors=True):
        super().__init__()
        self.use_fp = use_fp
        self.use_descriptors = use_descriptors
        input_dim = (FP_SIZE if use_fp else 0) + (len(DESCRIPTOR_NAMES) if use_descriptors else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, vector, atom_features=None, adjacency=None, bond_features=None, atom_mask=None):
        pieces = []
        if self.use_fp:
            pieces.append(vector[:, :FP_SIZE])
        if self.use_descriptors:
            pieces.append(vector[:, FP_SIZE:FP_INPUT_DIM])
        return self.net(torch.cat(pieces, dim=1))


class AttentiveGINELayer(nn.Module):
    def __init__(self, hidden_dim=192, dropout=0.15):
        super().__init__()
        self.edge_encoder = nn.Linear(BOND_FEATURE_DIM, hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adjacency, bond_features):
        edge_h = self.edge_encoder(bond_features)
        messages = self.message_mlp(torch.relu(h.unsqueeze(1) + edge_h))
        messages = messages * adjacency.unsqueeze(-1)
        aggregated = messages.sum(dim=2)
        return self.update_norm(h + self.dropout(aggregated))


class AttentiveGINEEncoder(nn.Module):
    def __init__(self, hidden_dim=192, depth=4, dropout=0.15):
        super().__init__()
        self.atom_encoder = nn.Sequential(
            nn.Linear(ATOM_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList([AttentiveGINELayer(hidden_dim, dropout) for _ in range(depth)])
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, atom_features, adjacency, bond_features, atom_mask):
        h = self.atom_encoder(atom_features)
        h = h * atom_mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, adjacency, bond_features)
            h = h * atom_mask.unsqueeze(-1)
        logits = self.attention(h).squeeze(-1).masked_fill(atom_mask == 0, -1e9)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (h * weights).sum(dim=1)


class AttentiveGINEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AttentiveGINEEncoder()
        self.head = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, vector, atom_features, adjacency, bond_features, atom_mask):
        graph = self.encoder(atom_features, adjacency, bond_features, atom_mask)
        return self.head(graph)


class DGAFNet(nn.Module):
    def __init__(self, use_fp=False, use_descriptors=True):
        super().__init__()
        self.use_fp = use_fp
        self.use_descriptors = use_descriptors
        self.graph_encoder = AttentiveGINEEncoder()
        vector_dim = (FP_SIZE if use_fp else 0) + (len(DESCRIPTOR_NAMES) if use_descriptors else 0)
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(384, 192),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(384, 192),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, vector, atom_features, adjacency, bond_features, atom_mask):
        graph_feat = self.graph_encoder(atom_features, adjacency, bond_features, atom_mask)
        pieces = []
        if self.use_fp:
            pieces.append(vector[:, :FP_SIZE])
        if self.use_descriptors:
            pieces.append(vector[:, FP_SIZE:FP_INPUT_DIM])
        vector_feat = self.vector_encoder(torch.cat(pieces, dim=1))
        gate = self.gate(torch.cat([graph_feat, vector_feat], dim=1))
        fused = gate * graph_feat + (1.0 - gate) * vector_feat
        return self.head(fused)


class DMPNNEncoder(nn.Module):
    def __init__(self, hidden_dim=256, depth=4, dropout=0.15):
        super().__init__()
        self.depth = depth
        self.input_layer = nn.Linear(DMPNN_BOND_INPUT_DIM, hidden_dim)
        self.message_layer = nn.Linear(hidden_dim, hidden_dim)
        self.atom_output = nn.Sequential(
            nn.Linear(ATOM_FEATURE_DIM + hidden_dim, hidden_dim),
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
            ).reshape(h.size(0), MAX_ATOMS, MAX_BONDS_PER_ATOM, h.size(-1)).sum(dim=2)
            source_messages = incoming.gather(
                1,
                b2src.unsqueeze(-1).expand(-1, -1, h.size(-1)),
            )
            reverse_messages = h.gather(
                1,
                b2revb.unsqueeze(-1).expand(-1, -1, h.size(-1)),
            )
            message = source_messages - reverse_messages
            h = torch.relu(h0 + self.message_layer(message))
            h = self.dropout(h) * bond_mask.unsqueeze(-1)

        incoming = h.gather(
            1,
            a2b.unsqueeze(-1).expand(-1, -1, -1, h.size(-1)).reshape(h.size(0), -1, h.size(-1)),
        ).reshape(h.size(0), MAX_ATOMS, MAX_BONDS_PER_ATOM, h.size(-1)).sum(dim=2)
        atom_hidden = self.atom_output(torch.cat([atom_features, incoming], dim=-1))
        atom_hidden = atom_hidden * atom_mask.unsqueeze(-1)
        denom = atom_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return atom_hidden.sum(dim=1) / denom


class DMPNNDescModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_encoder = DMPNNEncoder()
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(len(DESCRIPTOR_NAMES), 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(384, 256),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask):
        graph_feat = self.graph_encoder(
            atom_features,
            directed_bond_features,
            a2b,
            b2src,
            b2revb,
            bond_mask,
            atom_mask,
        )
        descriptor_feat = self.descriptor_encoder(vector[:, FP_SIZE:FP_INPUT_DIM])
        return self.head(torch.cat([graph_feat, descriptor_feat], dim=1))


def make_model(model_kind):
    cfg = MODEL_VARIANTS[model_kind]
    if cfg["architecture"] == "dpnn_desc":
        return DMPNNDescModel()
    if cfg["architecture"] == "vector":
        return VectorModel(cfg["use_fp"], cfg["use_descriptors"])
    if cfg["architecture"] == "graph":
        return AttentiveGINEModel()
    if cfg["architecture"] == "dgaf":
        return DGAFNet(cfg["use_fp"], cfg["use_descriptors"])
    raise ValueError(f"{model_kind} is not a torch model.")


class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=0.5)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        return 0.8 * self.huber(pred, target) + 0.2 * self.mse(pred, target)


def get_loss_fn(task_type, train_ds):
    if task_type == "reg":
        return RegressionLoss()
    labels = train_ds.y.view(-1)
    positives = (labels == 1).sum().float()
    negatives = (labels == 0).sum().float()
    pos_weight = torch.tensor(1.0 if positives <= 0 else negatives / positives).clamp(0.2, 8.0)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))


def loader_kwargs(shuffle=False):
    kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "pin_memory": DEVICE.type == "cuda",
        "num_workers": DATALOADER_WORKERS,
    }
    if DATALOADER_WORKERS > 0:
        kwargs["persistent_workers"] = True
    return kwargs


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
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    for batch in loader:
        vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, y = move_batch(batch)
        if is_train:
            optimizer.zero_grad()
        pred = model(vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=30.0, neginf=-30.0)
        loss = loss_fn(pred.view(-1), y.view(-1))
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
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask, y = move_batch(batch)
            pred = model(vector, atom_features, directed_bond_features, a2b, b2src, b2revb, bond_mask, atom_mask)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=30.0, neginf=-30.0)
            preds.append(pred.cpu().numpy().reshape(-1))
            labels.append(y.cpu().numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(labels)


def dataset_to_tree_arrays(dataset):
    return (
        dataset.vector.numpy().astype(np.float32, copy=False),
        dataset.y.numpy().reshape(-1).astype(np.float32, copy=False),
    )


def make_tree_model(task_type):
    if task_type == "reg":
        return ExtraTreesRegressor(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            bootstrap=True,
            random_state=42 + ACTIVE_FOLD_INDEX,
            n_jobs=-1,
        ), "ExtraTrees"
    return ExtraTreesClassifier(
        n_estimators=700,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        bootstrap=True,
        random_state=42 + ACTIVE_FOLD_INDEX,
        n_jobs=-1,
    ), "ExtraTrees"


def find_best_threshold(labels, probs):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        pred_cls = (probs >= threshold).astype(int)
        score = matthews_corrcoef(labels, pred_cls) if len(np.unique(pred_cls)) > 1 else -1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def calculate_regression_metrics_raw(labels_raw, preds_raw):
    labels_raw = np.nan_to_num(np.asarray(labels_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    preds_raw = np.nan_to_num(np.asarray(preds_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pearson = (
        np.corrcoef(labels_raw, preds_raw)[0, 1]
        if len(labels_raw) > 1 and np.std(labels_raw) > 0 and np.std(preds_raw) > 0
        else np.nan
    )
    spearman = pd.Series(labels_raw).corr(pd.Series(preds_raw), method="spearman")
    return {
        "R2": r2_score(labels_raw, preds_raw),
        "MAE": mean_absolute_error(labels_raw, preds_raw),
        "RMSE": np.sqrt(mean_squared_error(labels_raw, preds_raw)),
        "MedAE": median_absolute_error(labels_raw, preds_raw),
        "Pearson": pearson,
        "Spearman": spearman,
        "ExplainedVariance": explained_variance_score(labels_raw, preds_raw),
    }


def calculate_classification_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probs = np.nan_to_num(np.asarray(probs, dtype=np.float32), nan=0.5, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 0.0, 1.0)
    pred_cls = (probs >= threshold).astype(int)
    has_two_classes = len(np.unique(labels)) > 1
    if has_two_classes:
        tn, fp, fn, tp = confusion_matrix(labels, pred_cls, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    else:
        specificity = np.nan
        sensitivity = np.nan
    return {
        "AUC": roc_auc_score(labels, probs) if has_two_classes else np.nan,
        "ACC": accuracy_score(labels, pred_cls),
        "MCC": matthews_corrcoef(labels, pred_cls) if has_two_classes else np.nan,
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "Threshold": threshold,
    }


def logits_to_probs(logits):
    logits = np.nan_to_num(np.asarray(logits, dtype=np.float32), nan=0.0, posinf=30.0, neginf=-30.0)
    return 1 / (1 + np.exp(-logits))


def calculate_task_metrics(task_type, preds, labels, y_mean=None, y_std=None, target_transform=None, threshold=0.5):
    if task_type == "reg":
        preds_raw = denormalize_labels(preds, y_mean, y_std, target_transform)
        labels_raw = denormalize_labels(labels, y_mean, y_std, target_transform)
        return calculate_regression_metrics_raw(labels_raw, preds_raw)
    probs = logits_to_probs(preds)
    return calculate_classification_metrics(labels, probs, threshold)


def model_selection_score(task_type, metrics):
    if task_type == "clf":
        score = metrics.get("AUC", np.nan)
        return -np.inf if np.isnan(score) else score
    score = metrics.get("R2", np.nan)
    return -np.inf if np.isnan(score) else score


def train_torch_model(task_cfg, model_kind):
    train_ds = SingleTaskDataset(task_cfg, "train")
    val_ds = SingleTaskDataset(
        task_cfg,
        "val",
        train_ds.y_mean,
        train_ds.y_std,
        train_ds.active_descriptors,
        train_ds.target_transform,
        train_ds.descriptor_scaler,
    )
    test_ds = SingleTaskDataset(
        task_cfg,
        "test",
        train_ds.y_mean,
        train_ds.y_std,
        train_ds.active_descriptors,
        train_ds.target_transform,
        train_ds.descriptor_scaler,
    )
    train_loader = DataLoader(train_ds, **loader_kwargs(shuffle=True))
    val_loader = DataLoader(val_ds, **loader_kwargs(shuffle=False))
    test_loader = DataLoader(test_ds, **loader_kwargs(shuffle=False))

    model = make_model(model_kind).to(DEVICE)
    if task_cfg["TaskType"] == "reg":
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=8e-4)
        patience = REGRESSION_PATIENCE
    else:
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=4e-4)
        patience = PATIENCE
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    loss_fn = get_loss_fn(task_cfg["TaskType"], train_ds)

    best_state = None
    best_val_score = -np.inf
    wait = 0
    for epoch in range(MAX_EPOCHS):
        run_epoch(model, train_loader, loss_fn, optimizer)
        val_preds, val_labels = predict_dataset(model, val_loader)
        if task_cfg["TaskType"] == "clf":
            val_probs = logits_to_probs(val_preds)
            threshold = find_best_threshold(val_labels, val_probs)
            val_metrics = calculate_classification_metrics(val_labels, val_probs, threshold)
        else:
            val_metrics = calculate_task_metrics(
                task_cfg["TaskType"],
                val_preds,
                val_labels,
                train_ds.y_mean,
                train_ds.y_std,
                train_ds.target_transform,
            )
        val_score = model_selection_score(task_cfg["TaskType"], val_metrics)
        scheduler.step(val_score)
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    val_preds, val_labels = predict_dataset(model, val_loader)
    test_preds, test_labels = predict_dataset(model, test_loader)
    if task_cfg["TaskType"] == "clf":
        val_probs = logits_to_probs(val_preds)
        threshold = find_best_threshold(val_labels, val_probs)
        val_metrics = calculate_classification_metrics(val_labels, val_probs, threshold)
        test_metrics = calculate_classification_metrics(test_labels, logits_to_probs(test_preds), threshold)
    else:
        threshold = None
        val_metrics = calculate_task_metrics(
            task_cfg["TaskType"],
            val_preds,
            val_labels,
            train_ds.y_mean,
            train_ds.y_std,
            train_ds.target_transform,
        )
        test_metrics = calculate_task_metrics(
            task_cfg["TaskType"],
            test_preds,
            test_labels,
            train_ds.y_mean,
            train_ds.y_std,
            train_ds.target_transform,
        )
    return model, val_metrics, test_metrics, {
        "y_mean": train_ds.y_mean,
        "y_std": train_ds.y_std,
        "active_descriptors": train_ds.active_descriptors,
        "target_transform": train_ds.target_transform,
        "descriptor_scaler": train_ds.descriptor_scaler,
        "threshold": threshold,
    }


def train_tree_model(task_cfg, model_kind):
    train_ds = SingleTaskDataset(task_cfg, "train")
    val_ds = SingleTaskDataset(
        task_cfg,
        "val",
        train_ds.y_mean,
        train_ds.y_std,
        train_ds.active_descriptors,
        train_ds.target_transform,
        train_ds.descriptor_scaler,
    )
    test_ds = SingleTaskDataset(
        task_cfg,
        "test",
        train_ds.y_mean,
        train_ds.y_std,
        train_ds.active_descriptors,
        train_ds.target_transform,
        train_ds.descriptor_scaler,
    )
    x_train, y_train = dataset_to_tree_arrays(train_ds)
    x_val, y_val = dataset_to_tree_arrays(val_ds)
    x_test, y_test = dataset_to_tree_arrays(test_ds)
    model, backend = make_tree_model(task_cfg["TaskType"])
    model.fit(x_train, y_train)
    if task_cfg["TaskType"] == "reg":
        val_preds = model.predict(x_val)
        test_preds = model.predict(x_test)
        val_metrics = calculate_task_metrics("reg", val_preds, y_val, train_ds.y_mean, train_ds.y_std, train_ds.target_transform)
        test_metrics = calculate_task_metrics("reg", test_preds, y_test, train_ds.y_mean, train_ds.y_std, train_ds.target_transform)
        threshold = None
    else:
        val_probs = model.predict_proba(x_val)[:, 1]
        test_probs = model.predict_proba(x_test)[:, 1]
        threshold = find_best_threshold(y_val, val_probs)
        val_metrics = calculate_classification_metrics(y_val, val_probs, threshold)
        test_metrics = calculate_classification_metrics(y_test, test_probs, threshold)
    val_metrics["Backend"] = backend
    test_metrics["Backend"] = backend
    return model, val_metrics, test_metrics, {
        "y_mean": train_ds.y_mean,
        "y_std": train_ds.y_std,
        "active_descriptors": train_ds.active_descriptors,
        "target_transform": train_ds.target_transform,
        "descriptor_scaler": train_ds.descriptor_scaler,
        "threshold": threshold,
    }


def save_model(task_cfg, model, model_kind, meta, rank):
    fold_suffix = f"_{SPLIT_LABEL}_fold{ACTIVE_FOLD_INDEX}_rank{rank}_{model_kind}"
    base = f"{safe_name(task_cfg['Category'])}_{safe_name(task_cfg['Task'])}{fold_suffix}"
    if MODEL_VARIANTS[model_kind]["architecture"] == "tree":
        path = os.path.join(MODEL_DIR, f"{base}.pkl")
        with open(path, "wb") as handle:
            pickle.dump({"model": model, "model_kind": model_kind, "task_cfg": task_cfg, "meta": meta}, handle)
        return path
    path = os.path.join(MODEL_DIR, f"{base}.pth")
    torch.save(
        {
            "model_kind": model_kind,
            "model_state_dict": model.state_dict(),
            "task_cfg": task_cfg,
            "meta": meta,
            "fp_size": FP_SIZE,
            "descriptor_names": DESCRIPTOR_NAMES,
            "max_atoms": MAX_ATOMS,
            "max_directed_bonds": MAX_DIRECTED_BONDS,
            "max_bonds_per_atom": MAX_BONDS_PER_ATOM,
        },
        path,
    )
    return path


def predict_model_on_dataset(model, model_kind, dataset):
    if MODEL_VARIANTS[model_kind]["architecture"] == "tree":
        x, y = dataset_to_tree_arrays(dataset)
        if dataset.task_type == "clf":
            return model.predict_proba(x)[:, 1], y
        return model.predict(x), y
    loader = DataLoader(dataset, **loader_kwargs(shuffle=False))
    preds, labels = predict_dataset(model, loader)
    if dataset.task_type == "clf":
        return logits_to_probs(preds), labels
    return preds, labels


def calculate_ensemble_metrics(task_cfg, top_items, meta):
    val_ds = SingleTaskDataset(
        task_cfg,
        "val",
        meta["y_mean"],
        meta["y_std"],
        meta["active_descriptors"],
        meta["target_transform"],
        meta["descriptor_scaler"],
    )
    test_ds = SingleTaskDataset(
        task_cfg,
        "test",
        meta["y_mean"],
        meta["y_std"],
        meta["active_descriptors"],
        meta["target_transform"],
        meta["descriptor_scaler"],
    )
    val_predictions, test_predictions = [], []
    val_labels = test_labels = None
    for model_kind, model in top_items:
        val_preds, current_val_labels = predict_model_on_dataset(model, model_kind, val_ds)
        test_preds, current_test_labels = predict_model_on_dataset(model, model_kind, test_ds)
        val_labels = current_val_labels if val_labels is None else val_labels
        test_labels = current_test_labels if test_labels is None else test_labels
        val_predictions.append(val_preds)
        test_predictions.append(test_preds)
    val_mean = np.mean(np.vstack(val_predictions), axis=0)
    test_mean = np.mean(np.vstack(test_predictions), axis=0)
    if task_cfg["TaskType"] == "clf":
        threshold = find_best_threshold(val_labels, val_mean)
        val_metrics = calculate_classification_metrics(val_labels, val_mean, threshold)
        test_metrics = calculate_classification_metrics(test_labels, test_mean, threshold)
    else:
        threshold = None
        val_metrics = calculate_task_metrics("reg", val_mean, val_labels, meta["y_mean"], meta["y_std"], meta["target_transform"])
        test_metrics = calculate_task_metrics("reg", test_mean, test_labels, meta["y_mean"], meta["y_std"], meta["target_transform"])
    val_metrics["Members"] = ",".join(model_kind for model_kind, _ in top_items)
    test_metrics["Members"] = ",".join(model_kind for model_kind, _ in top_items)
    return val_metrics, test_metrics, threshold


def run_task_fold(task_cfg):
    print(f"\n========== {SPLIT_LABEL} | {task_cfg['Category']} | {task_cfg['Task']} ==========")
    trained_models, validation_metrics, test_metrics, metadata = {}, {}, {}, {}
    for model_kind in ACTIVE_MODEL_KINDS:
        print(f"Training variant: {model_kind}")
        train_fn = train_tree_model if MODEL_VARIANTS[model_kind]["architecture"] == "tree" else train_torch_model
        model, val_metrics, current_test_metrics, meta = train_fn(task_cfg, model_kind)
        trained_models[model_kind] = model
        validation_metrics[model_kind] = val_metrics
        test_metrics[model_kind] = current_test_metrics
        metadata[model_kind] = meta

    best_kind = max(validation_metrics, key=lambda name: model_selection_score(task_cfg["TaskType"], validation_metrics[name]))
    top_model_kinds = sorted(
        validation_metrics,
        key=lambda name: model_selection_score(task_cfg["TaskType"], validation_metrics[name]),
        reverse=True,
    )[: min(ENSEMBLE_TOP_K, len(validation_metrics))]

    top_items = [(model_kind, trained_models[model_kind]) for model_kind in top_model_kinds]
    ensemble_meta = metadata[top_model_kinds[0]]
    ensemble_val_metrics, ensemble_test_metrics, ensemble_threshold = calculate_ensemble_metrics(task_cfg, top_items, ensemble_meta)
    saved_paths = [save_model(task_cfg, trained_models[k], k, metadata[k], rank) for rank, k in enumerate(top_model_kinds, 1)]

    metric = "R2" if task_cfg["TaskType"] == "reg" else "AUC"
    baseline_model = "FP_MLP" if "FP_MLP" in test_metrics else None
    baseline_test_score = (
        test_metrics[baseline_model].get(metric, np.nan)
        if baseline_model is not None
        else np.nan
    )
    row = {
        "Category": task_cfg["Category"],
        "Task": task_cfg["Task"],
        "TaskType": task_cfg["TaskType"],
        "Fold": ACTIVE_FOLD_INDEX,
        "SplitMode": SPLIT_LABEL,
        "TrainedModels": ",".join(ACTIVE_MODEL_KINDS),
        "BestModel": best_kind,
        "BestValScore": model_selection_score(task_cfg["TaskType"], validation_metrics[best_kind]),
        "BestTestScore": model_selection_score(task_cfg["TaskType"], test_metrics[best_kind]),
        "EnsembleTopK": len(top_model_kinds),
        "EnsembleTopModels": ",".join(top_model_kinds),
        "EnsembleSavedModels": "|".join(saved_paths),
        "EnsembleValScore": model_selection_score(task_cfg["TaskType"], ensemble_val_metrics),
        "EnsembleTestScore": model_selection_score(task_cfg["TaskType"], ensemble_test_metrics),
        "EnsembleThreshold": ensemble_threshold,
        "BaselineModel": baseline_model,
        "BaselineTestScore": baseline_test_score,
        "EnsembleImprovement": model_selection_score(task_cfg["TaskType"], ensemble_test_metrics) - baseline_test_score,
        "ActiveDescriptors": ",".join(ensemble_meta["active_descriptors"]),
    }
    for model_kind, metrics in validation_metrics.items():
        for key, value in metrics.items():
            row[f"Val_{model_kind}_{key}"] = value
    for model_kind, metrics in test_metrics.items():
        for key, value in metrics.items():
            row[f"Test_{model_kind}_{key}"] = value
    for key, value in ensemble_val_metrics.items():
        row[f"Val_Top{len(top_model_kinds)}_Ensemble_{key}"] = value
    for key, value in ensemble_test_metrics.items():
        row[f"Test_Top{len(top_model_kinds)}_Ensemble_{key}"] = value
    return row


def build_ensemble_manifest(summary_df):
    rows = []
    for _, row in summary_df.iterrows():
        models = str(row["EnsembleTopModels"]).split(",")
        paths = str(row["EnsembleSavedModels"]).split("|")
        for rank, (model_kind, path) in enumerate(zip(models, paths), start=1):
            rows.append(
                {
                    "Category": row["Category"],
                    "Task": row["Task"],
                    "TaskType": row["TaskType"],
                    "Fold": row["Fold"],
                    "SplitMode": row.get("SplitMode", SPLIT_LABEL),
                    "TrainedModels": row.get("TrainedModels", ",".join(ACTIVE_MODEL_KINDS)),
                    "ModelRank": rank,
                    "ModelKind": model_kind,
                    "BestModel": row["BestModel"],
                    "SavedModel": path,
                    "EnsembleTopK": row["EnsembleTopK"],
                    "EnsembleTopModels": row["EnsembleTopModels"],
                    "EnsembleThreshold": row["EnsembleThreshold"],
                }
            )
    return pd.DataFrame(rows)


def save_metrics_overview_plot(summary_df):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except Exception as exc:
        print(f"Skipping metrics overview plot because matplotlib is unavailable: {exc}")
        return None

    if summary_df.empty:
        return None

    model_kind = ACTIVE_MODEL_KINDS[0]
    plot_df = summary_df.copy()
    plot_df = plot_df.sort_values(["Category", "Task"]).reset_index(drop=True)

    clf_df = plot_df[plot_df["TaskType"] == "clf"]
    reg_df = plot_df[plot_df["TaskType"] == "reg"]
    panel_count = int(not clf_df.empty) + int(not reg_df.empty)
    if panel_count == 0:
        return None

    fig_height = max(4.0, 0.44 * len(plot_df) + 2.3)
    fig, axes = plt.subplots(panel_count, 1, figsize=(17, fig_height), squeeze=False)
    axes = axes[:, 0]
    good_cmap = plt.get_cmap("Greens")
    bad_cmap = plt.get_cmap("Greens_r")

    def fmt_value(value):
        if pd.isna(value):
            return ""
        return f"{float(value):.4f}"

    def metric_color(values, value, higher_is_better=True):
        if pd.isna(value):
            return "#f3f4f6"
        finite = pd.to_numeric(values, errors="coerce").dropna()
        if finite.empty:
            return "#ffffff"
        vmin = float(finite.min())
        vmax = float(finite.max())
        if abs(vmax - vmin) < 1e-12:
            normalized = 0.65
        else:
            normalized = (float(value) - vmin) / (vmax - vmin)
        normalized = float(np.clip(normalized, 0.0, 1.0))
        if not higher_is_better:
            normalized = 1.0 - normalized
        return colors.to_hex(good_cmap(0.18 + 0.72 * normalized))

    def draw_table(ax, df, metric_specs, title):
        ax.axis("off")
        columns = ["Category", "Fold", "Task"] + [metric for metric, _ in metric_specs]
        cell_text, cell_colors = [], []
        for _, row in df.iterrows():
            row_text = [str(row["Category"]), str(row.get("Fold", "")), str(row["Task"])]
            row_colors = ["#e5e7eb", "#eef2ff", "#f9fafb"]
            for metric, higher_is_better in metric_specs:
                col = f"Test_{model_kind}_{metric}"
                value = pd.to_numeric(row.get(col, np.nan), errors="coerce")
                row_text.append(fmt_value(value))
                row_colors.append(metric_color(pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce"), value, higher_is_better))
            cell_text.append(row_text)
            cell_colors.append(row_colors)

        col_widths = [0.11, 0.05, 0.30] + [0.54 / max(1, len(metric_specs))] * len(metric_specs)
        table = ax.table(
            cellText=cell_text,
            colLabels=columns,
            cellColours=cell_colors,
            colColours=["#111827"] * len(columns),
            colWidths=col_widths,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.45)
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#d1d5db")
            if row_idx == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_height(cell.get_height() * 1.15)
            elif col_idx in (0, 2):
                cell.set_text_props(ha="left")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    axis_idx = 0
    if not clf_df.empty:
        draw_table(
            axes[axis_idx],
            clf_df,
            [("AUC", True), ("ACC", True), ("MCC", True), ("Specificity", True), ("Sensitivity", True), ("Threshold", True)],
            "Classification Tasks - Test Metrics",
        )
        axis_idx += 1
    if not reg_df.empty:
        draw_table(
            axes[axis_idx],
            reg_df,
            [("R2", True), ("MAE", False), ("RMSE", False), ("MedAE", False), ("Pearson", True), ("Spearman", True), ("ExplainedVariance", True)],
            "Regression Tasks - Test Metrics",
        )

    fig.suptitle(f"{','.join(ACTIVE_MODEL_KINDS)} Metrics Table ({SPLIT_LABEL})", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    plot_path = os.path.join(RESULT_DIR, "metrics_overview.png")
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics overview plot to: {plot_path}")
    return plot_path


def run_all_tasks():
    global ACTIVE_FOLD_INDEX
    seed_everything(42)
    print_runtime_info()

    tasks = discover_tasks(DATA_ROOT)
    if TASK_CATEGORIES:
        allowed_categories = {category.lower() for category in TASK_CATEGORIES}
        tasks = [
            task
            for task in tasks
            if task["Category"].lower() in allowed_categories
        ]
    if TASK_FILTER:
        tasks = [
            task
            for task in tasks
            if TASK_FILTER in task["Category"].lower()
            or TASK_FILTER in task["Task"].lower()
        ]
    print(f"Discovered {len(tasks)} endpoints.")
    print(f"Task categories: {','.join(TASK_CATEGORIES) if TASK_CATEGORIES else 'all'}")
    print(f"Training models: {','.join(ACTIVE_MODEL_KINDS)}")
    print(f"Split mode: {SPLIT_LABEL}; active folds: {','.join(str(fold) for fold in ACTIVE_FOLDS)}")

    pd.DataFrame(tasks).drop(columns=["SplitFiles"]).to_csv(
        os.path.join(RESULT_DIR, "discovered_endpoints.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    existing_rows, completed_keys = load_existing_results()
    rows = list(existing_rows)
    print(f"Loaded {len(completed_keys)} completed task rows from existing result files.")

    partial_path = os.path.join(RESULT_DIR, "all_task_metrics_summary_partial.csv")
    for fold_idx in ACTIVE_FOLDS:
        ACTIVE_FOLD_INDEX = fold_idx
        seed_everything(42 + fold_idx)
        for task_cfg in tasks:
            key = (str(task_cfg["Category"]), str(task_cfg["Task"]), int(ACTIVE_FOLD_INDEX))
            if key in completed_keys:
                print(f"Skipping completed: Fold {fold_idx} | {task_cfg['Category']} | {task_cfg['Task']}")
                continue
            try:
                rows.append(run_task_fold(task_cfg))
                completed_keys.add(key)
                pd.DataFrame(rows).to_csv(
                    partial_path,
                    index=False,
                    encoding="utf-8-sig",
                )
            finally:
                clear_task_caches(task_cfg)

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(RESULT_DIR, "all_task_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    build_ensemble_manifest(summary_df).to_csv(
        os.path.join(RESULT_DIR, "ensemble_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_metrics_overview_plot(summary_df)
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved models to: {MODEL_DIR}")
    return summary_df


if __name__ == "__main__":
    run_all_tasks()


