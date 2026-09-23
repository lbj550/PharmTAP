import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")


def safe_name(text):
    text = str(text).replace("\\", "_").replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "task"


def find_smiles_column(df):
    candidates = [
        "smiles_standarized",
        "smiles_standardized",
        "canonical_smiles",
        "smiles",
        "SMILES",
    ]
    for name in candidates:
        if name in df.columns:
            return name
    for name in df.columns:
        if name.lower() != "label":
            return name
    raise ValueError("Could not identify a SMILES column.")


def discover_tasks(data_root):
    tasks = []
    data_root = Path(data_root)
    for category_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for dirpath, _, files in os.walk(category_dir):
            csv_files = [f for f in files if f.lower().endswith(".csv")]
            frames = []
            for csv_name in csv_files:
                path = Path(dirpath) / csv_name
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                if "label" not in df.columns:
                    continue
                smiles_col = find_smiles_column(df)
                frames.append(df[[smiles_col, "label"]].rename(columns={smiles_col: "smiles"}))
            if not frames:
                continue
            merged = pd.concat(frames, ignore_index=True)
            labels = pd.to_numeric(merged["label"], errors="coerce").dropna()
            if labels.empty:
                continue
            unique = set(labels.unique().tolist())
            task_type = "clf" if unique.issubset({0, 1, 0.0, 1.0}) and len(unique) <= 2 else "reg"
            tasks.append(
                {
                    "category": category_dir.name,
                    "task": os.path.relpath(dirpath, category_dir),
                    "folder": Path(dirpath),
                    "task_type": task_type,
                    "frame": merged,
                }
            )
    return tasks


def scaffold_for_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    candidates = [mol]
    try:
        cleaned = Chem.Mol(mol)
        Chem.RemoveStereochemistry(cleaned)
        Chem.SanitizeMol(cleaned)
        candidates.append(cleaned)
    except Exception:
        pass

    for candidate in candidates:
        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(candidate)
            Chem.RemoveStereochemistry(scaffold_mol)
            scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=False)
            if scaffold:
                return scaffold
        except Exception:
            continue

    try:
        fallback = Chem.Mol(mol)
        Chem.RemoveStereochemistry(fallback)
        return Chem.MolToSmiles(fallback, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def prepare_frame(task):
    df = task["frame"].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["smiles", "label"])
    df = df.drop_duplicates(subset=["smiles", "label"]).reset_index(drop=True)
    scaffolds = []
    valid_mask = []
    for smiles in df["smiles"]:
        scaffold = scaffold_for_smiles(smiles)
        scaffolds.append(scaffold)
        valid_mask.append(scaffold is not None)
    invalid_count = int(len(df) - sum(valid_mask))
    df = df.loc[valid_mask].copy().reset_index(drop=True)
    df["scaffold"] = [s for s in scaffolds if s is not None]
    return df, invalid_count


def split_scaffolds(df, task_type, ratios, seed):
    if len(df) < 3:
        indices = np.arange(len(df), dtype=int)
        return {
            "train": indices,
            "val": indices[:0],
            "test": indices[:0],
        }, {
            "train": {"indices": indices.tolist(), "n": len(indices), "pos": 0, "scaffolds": df["scaffold"].nunique()},
            "val": {"indices": [], "n": 0, "pos": 0, "scaffolds": 0},
            "test": {"indices": [], "n": 0, "pos": 0, "scaffolds": 0},
        }

    rng = np.random.default_rng(seed)
    groups = []
    for scaffold, group in df.groupby("scaffold", sort=False):
        labels = group["label"].to_numpy(dtype=float)
        groups.append(
            {
                "scaffold": scaffold,
                "indices": group.index.to_numpy(dtype=int),
                "n": len(group),
                "pos": int(np.sum(labels == 1)) if task_type == "clf" else 0,
            }
        )

    rng.shuffle(groups)
    groups.sort(key=lambda item: item["n"], reverse=True)

    split_names = ["train", "val", "test"]
    target_counts = {name: ratios[i] * len(df) for i, name in enumerate(split_names)}
    total_pos = int(np.sum(df["label"].to_numpy(dtype=float) == 1)) if task_type == "clf" else 0
    target_pos = {name: ratios[i] * total_pos for i, name in enumerate(split_names)}
    buckets = {name: {"indices": [], "n": 0, "pos": 0, "scaffolds": 0} for name in split_names}

    for group in groups:
        best_name = None
        best_score = None
        for name in split_names:
            after_n = buckets[name]["n"] + group["n"]
            size_score = abs(after_n - target_counts[name]) / max(target_counts[name], 1.0)
            if task_type == "clf" and total_pos > 0:
                after_pos = buckets[name]["pos"] + group["pos"]
                pos_score = abs(after_pos - target_pos[name]) / max(target_pos[name], 1.0)
            else:
                pos_score = 0.0
            overflow = max(0.0, after_n - target_counts[name]) / max(target_counts[name], 1.0)
            score = size_score + 0.35 * pos_score + 0.5 * overflow
            if best_score is None or score < best_score:
                best_name = name
                best_score = score
        buckets[best_name]["indices"].extend(group["indices"].tolist())
        buckets[best_name]["n"] += group["n"]
        buckets[best_name]["pos"] += group["pos"]
        buckets[best_name]["scaffolds"] += 1

    for split_name in ["val", "test"]:
        if buckets[split_name]["n"] > 0 or buckets["train"]["scaffolds"] <= 1:
            continue
        largest_train_scaffold = None
        largest_train_indices = []
        for group in groups:
            if all(idx in buckets["train"]["indices"] for idx in group["indices"]):
                if len(group["indices"]) > len(largest_train_indices):
                    largest_train_scaffold = group
                    largest_train_indices = group["indices"].tolist()
        if largest_train_scaffold is None:
            continue
        move_set = set(largest_train_indices)
        buckets["train"]["indices"] = [idx for idx in buckets["train"]["indices"] if idx not in move_set]
        buckets["train"]["n"] -= largest_train_scaffold["n"]
        buckets["train"]["pos"] -= largest_train_scaffold["pos"]
        buckets["train"]["scaffolds"] -= 1
        buckets[split_name]["indices"].extend(largest_train_indices)
        buckets[split_name]["n"] += largest_train_scaffold["n"]
        buckets[split_name]["pos"] += largest_train_scaffold["pos"]
        buckets[split_name]["scaffolds"] += 1

    return {name: np.asarray(bucket["indices"], dtype=int) for name, bucket in buckets.items()}, buckets


def write_task(task, output_root, ratios, seed):
    df, invalid_count = prepare_frame(task)
    if len(df) < 3:
        return None
    split_indices, buckets = split_scaffolds(df, task["task_type"], ratios, seed)
    out_dir = Path(output_root) / task["category"] / task["task"]
    out_dir.mkdir(parents=True, exist_ok=True)
    task_file_stem = safe_name(task["task"])

    row = {
        "Category": task["category"],
        "Task": task["task"],
        "TaskType": task["task_type"],
        "TotalValid": len(df),
        "InvalidSmiles": invalid_count,
        "ScaffoldCount": df["scaffold"].nunique(),
    }
    for split_name in ["train", "val", "test"]:
        split_df = df.iloc[split_indices[split_name]][["smiles", "label"]].copy()
        split_df = split_df.rename(columns={"smiles": "smiles_standarized"})
        split_df.to_csv(out_dir / f"{task_file_stem}_{split_name}.csv", index=False)
        labels = split_df["label"].to_numpy(dtype=float)
        row[f"{split_name}_n"] = len(split_df)
        row[f"{split_name}_scaffolds"] = buckets[split_name]["scaffolds"]
        if task["task_type"] == "clf" and len(labels):
            row[f"{split_name}_positive_rate"] = float(np.mean(labels == 1))
        else:
            row[f"{split_name}_positive_rate"] = np.nan
    return row


def parse_ratios(text):
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError("--ratios must contain three positive values, e.g. 0.8,0.1,0.1")
    total = sum(values)
    return [v / total for v in values]


def main():
    parser = argparse.ArgumentParser(description="Build Bemis-Murcko scaffold train/val/test splits for ADMET endpoints.")
    parser.add_argument("--data-root", default=os.getenv("DATA_ROOT", "data/admet/data"))
    parser.add_argument("--output-root", default=os.getenv("SCAFFOLD_OUTPUT_ROOT", "data/admet/data_scaffold_8_1_1"))
    parser.add_argument("--ratios", default=os.getenv("SCAFFOLD_RATIOS", "0.8,0.1,0.1"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SCAFFOLD_SEED", "42")))
    args = parser.parse_args()

    ratios = parse_ratios(args.ratios)
    tasks = discover_tasks(args.data_root)
    rows = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task['category']} / {task['task']}")
        row = write_task(task, args.output_root, ratios, args.seed)
        if row is not None:
            rows.append(row)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stats = pd.DataFrame(rows)
    stats.to_csv(output_root / "scaffold_split_stats.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_data_root": args.data_root,
                "output_root": args.output_root,
                "ratios": ",".join(f"{r:.6f}" for r in ratios),
                "seed": args.seed,
                "task_count": len(rows),
            }
        ]
    ).to_csv(output_root / "scaffold_split_manifest.csv", index=False)
    print(f"Wrote scaffold splits for {len(rows)} tasks to {args.output_root}")


if __name__ == "__main__":
    main()
