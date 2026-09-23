import argparse
import os
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
FP_SIZE = 2048
RANDOM_SEED = 42

G_CLUSTER_CACHE_DIR = None
G_SHAPE_CACHE_DIR = None
G_ADAPTIVE_STATS_DIR = None
G_ENABLE_SHAPE = True
G_QUERIES = []

DEFAULT_CLUSTER_CACHE_DIR = "outputs/target_prediction/srcp_target_profiles"
DEFAULT_SHAPE_CACHE_DIR = "outputs/target_prediction/usrcat_target_profiles"
DEFAULT_ADAPTIVE_STATS_DIR = "outputs/target_prediction/adaptive_support_backgrounds"


def init_worker():
    RDLogger.DisableLog("rdApp.*")


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def load_pickle(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def collect_target_files(cluster_cache_dir, limit_targets=None):
    files = [
        f for f in os.listdir(cluster_cache_dir)
        if f.endswith(".pkl") and os.path.isfile(os.path.join(cluster_cache_dir, f))
    ]
    files.sort()
    if limit_targets:
        files = files[:limit_targets]
    return files


def standardize_smiles(smiles):
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
        te = rdMolStandardize.TautomerEnumerator()
        te.SetMaxTautomers(50)
        te.SetMaxTransforms(50)
        return te.Canonicalize(mol)
    except Exception:
        return None


def get_morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=MORGAN_RADIUS, nBits=FP_SIZE
    )


def bitvect_from_bytes(fp_bytes):
    if isinstance(fp_bytes, str):
        fp = DataStructs.ExplicitBitVect(len(fp_bytes) * 8)
        fp.InitFromString(fp_bytes)
        return fp
    return DataStructs.ExplicitBitVect(fp_bytes)


def get_scaffold_smiles(mol):
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold if scaffold else "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


def calc_properties(mol):
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "rotb": float(Lipinski.NumRotatableBonds(mol)),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "formal_charge": float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
    }


def property_similarity(query_props, cluster_property_profile):
    keys = [
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rotb",
        "ring_count",
        "aromatic_ring_count",
        "heavy_atoms",
        "formal_charge",
    ]
    z_values = []
    for key in keys:
        mean = cluster_property_profile.get(f"{key}_mean")
        std = cluster_property_profile.get(f"{key}_std")
        if mean is None or std is None or key not in query_props:
            continue
        denom = max(float(std), 1.0)
        z_values.append(abs(float(query_props[key]) - float(mean)) / denom)

    if not z_values:
        return 0.0
    return 1.0 / (1.0 + float(np.mean(z_values)))


def best_baseline_score(query, known_morgan_fps):
    if not known_morgan_fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(query["morgan_fp"], known_morgan_fps)
    return float(max(sims)) if sims else 0.0


def cluster_score(query, cluster, known_morgan_fps):
    centroid_fp = bitvect_from_bytes(cluster["centroid_morgan_fp_bytes"])
    centroid_sim = DataStructs.TanimotoSimilarity(query["morgan_fp"], centroid_fp)

    member_fps = [
        known_morgan_fps[i]
        for i in cluster.get("ligand_indices", [])
        if i < len(known_morgan_fps)
    ]
    if not member_fps:
        return 0.0
    member_sims = DataStructs.BulkTanimotoSimilarity(query["morgan_fp"], member_fps)
    top3_mean = float(np.mean(sorted(member_sims, reverse=True)[:3]))

    scaffold_score = (
        1.0
        if query.get("scaffold") != "NO_SCAFFOLD"
        and query.get("scaffold") == cluster.get("main_scaffold")
        else 0.0
    )
    prop_score = property_similarity(
        query.get("properties", {}),
        cluster.get("property_profile", {}),
    )
    confidence = min(np.log1p(cluster.get("size", 1)) / np.log1p(50.0), 1.0)
    return float(
        0.35 * centroid_sim
        + 0.30 * top3_mean
        + 0.15 * scaffold_score
        + 0.15 * prop_score
        + 0.05 * confidence
    )


def best_cluster_score(query, target_cache, known_morgan_fps):
    best = 0.0
    for cluster in target_cache.get("clusters", []):
        score = cluster_score(query, cluster, known_morgan_fps)
        if score > best:
            best = score
    return best


def calculate_usrcat_descriptor(mol, conf_id=0):
    try:
        descriptor = rdMolDescriptors.GetUSRCAT(mol, confId=conf_id)
        return np.array(descriptor, dtype=np.float32)
    except Exception:
        return None


def generate_query_shape_vectors(mol, num_confs=10):
    try:
        mol = Chem.Mol(mol)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED
        params.numThreads = 1
        params.pruneRmsThresh = 0.5
        params.maxIterations = 1000

        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
        if not conf_ids:
            return []

        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=150)
        vectors = []
        for conf_id in conf_ids:
            vector = calculate_usrcat_descriptor(mol, conf_id)
            if vector is not None and np.all(np.isfinite(vector)):
                vectors.append(vector)
        return vectors
    except Exception:
        return []


def vector_similarity(query_vector, ref_vector):
    query_vector = [float(x) for x in query_vector]
    ref_vector = [float(x) for x in ref_vector]
    return float(rdMolDescriptors.GetUSRScore(query_vector, ref_vector))


def representative_best_shape_score(query_vectors, representative):
    best = 0.0
    for q_vec in query_vectors:
        for ref_vec in representative.get("vectors", []):
            score = vector_similarity(q_vec, ref_vec)
            if score > best:
                best = score
    return best


def best_shape_score(query, shape_cache):
    if not shape_cache:
        return np.nan
    query_vectors = query.get("shape_vectors", [])
    if isinstance(query_vectors, tuple):
        query_vectors = query_vectors[0]
    if not query_vectors:
        return np.nan

    best = 0.0
    for cluster in shape_cache.get("clusters", []):
        rep_scores = [
            representative_best_shape_score(query_vectors, rep)
            for rep in cluster.get("representatives", [])
        ]
        rep_scores = [s for s in rep_scores if np.isfinite(s)]
        if not rep_scores:
            continue
        top3_mean = float(np.mean(sorted(rep_scores, reverse=True)[:3]))
        confidence = min(np.log1p(cluster.get("size", 1)) / np.log1p(50.0), 1.0)
        score = 0.85 * max(rep_scores) + 0.10 * top3_mean + 0.05 * confidence
        if score > best:
            best = score
    return float(best)


def adaptive_calibrated_score(raw_score, background_stats):
    if raw_score is None or not np.isfinite(raw_score) or not background_stats:
        return np.nan
    scores = background_stats.get("scores") or []
    mean = background_stats.get("mean")
    std = background_stats.get("std")
    if mean is None or std is None or not scores:
        return np.nan

    arr = np.array(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    percentile = float(np.sum(arr <= raw_score) / len(arr))
    z = (float(raw_score) - float(mean)) / max(float(std), 1e-6)
    if z >= 0:
        sig = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        sig = ez / (1.0 + ez)
    return 0.5 * percentile + 0.5 * float(sig)


def fuse_raw(primary_score, secondary_score, primary_weight=0.85):
    primary_ok = primary_score is not None and np.isfinite(primary_score)
    secondary_ok = secondary_score is not None and np.isfinite(secondary_score)
    if primary_ok and secondary_ok:
        return float(primary_weight * primary_score + (1.0 - primary_weight) * secondary_score)
    if primary_ok:
        return float(primary_score)
    if secondary_ok:
        return float(secondary_score)
    return np.nan


# -----------------------------------------------------------------------------
# Public single-molecule / batch prediction CLI
# -----------------------------------------------------------------------------

def support_label(value):
    if value is None or not np.isfinite(value):
        return "Unavailable"
    if value >= 0.95:
        return "High"
    if value >= 0.75:
        return "Moderate"
    return "Low"


def build_query_from_smiles(smiles, query_confs=10):
    mol = standardize_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    query = {
        "input_smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "morgan_fp": get_morgan_fp(mol),
        "scaffold": get_scaffold_smiles(mol),
        "properties": calc_properties(mol),
        "shape_vectors": generate_query_shape_vectors(mol, query_confs),
    }
    return query


def predict_targets_for_query(
    query,
    cluster_cache_dir=DEFAULT_CLUSTER_CACHE_DIR,
    shape_cache_dir=DEFAULT_SHAPE_CACHE_DIR,
    adaptive_stats_dir=DEFAULT_ADAPTIVE_STATS_DIR,
    top_n=100,
    srcp_weight=0.85,
    shape_prefilter_topk=300,
    limit_targets=None,
):
    target_files = collect_target_files(cluster_cache_dir, limit_targets)
    rows = []
    for cache_file in target_files:
        target_id = os.path.splitext(cache_file)[0]
        cluster_cache = load_pickle(os.path.join(cluster_cache_dir, cache_file))
        if not cluster_cache:
            continue
        known_morgan_fps = [
            bitvect_from_bytes(fp_bytes)
            for fp_bytes in cluster_cache.get("known_morgan_fps_bytes", [])
        ]
        srcp_score = best_cluster_score(query, cluster_cache, known_morgan_fps)
        rows.append({
            "target_id": target_id,
            "target_gene": cluster_cache.get("target_gene", ""),
            "protein_name": cluster_cache.get("protein_name", ""),
            "target_type": cluster_cache.get("target_type", ""),
            "organism": cluster_cache.get("organism", ""),
            "target_chembl": cluster_cache.get("target_chembl", ""),
            "srcp_score": srcp_score,
            "usrcat_score": np.nan,
            "fusion_score": srcp_score,
            "srcp_adaptive_support": np.nan,
            "usrcat_adaptive_support": np.nan,
        })

    rows.sort(key=lambda row: (-row["srcp_score"], row["target_id"]))
    shape_count = len(rows) if shape_prefilter_topk == 0 else min(shape_prefilter_topk, len(rows))
    for row in rows[:shape_count]:
        shape_cache = load_pickle(os.path.join(shape_cache_dir, f"{row['target_id']}.pkl"))
        row["usrcat_score"] = best_shape_score(query, shape_cache)
        row["fusion_score"] = fuse_raw(row["srcp_score"], row["usrcat_score"], srcp_weight)
        adaptive_cache = load_pickle(os.path.join(adaptive_stats_dir, f"{row['target_id']}.pkl"))
        if adaptive_cache:
            row["srcp_adaptive_support"] = adaptive_calibrated_score(
                row["srcp_score"], adaptive_cache.get("cluster_profile")
            )
            row["usrcat_adaptive_support"] = adaptive_calibrated_score(
                row["usrcat_score"], adaptive_cache.get("shape_profile")
            )

    rows.sort(key=lambda row: (-row["fusion_score"], row["target_id"]))
    for rank, row in enumerate(rows[:top_n], 1):
        row["rank"] = rank
        row["srcp_support_label"] = support_label(row.get("srcp_adaptive_support"))
        row["usrcat_support_label"] = support_label(row.get("usrcat_adaptive_support"))
    return rows[:top_n]


def read_smiles_inputs(args):
    smiles_values = []
    if args.smiles:
        smiles_values.append(args.smiles)
    if args.smiles_file:
        with open(args.smiles_file, encoding="utf-8") as handle:
            for line in handle:
                value = line.strip().split()[0] if line.strip() else ""
                if value:
                    smiles_values.append(value)
    if not smiles_values:
        raise ValueError("Provide --smiles or --smiles-file.")
    return smiles_values


def write_prediction_csv(results, output_csv):
    import csv
    fieldnames = [
        "molecule_index", "input_smiles", "canonical_smiles", "rank", "target_id",
        "target_gene", "protein_name", "target_type", "organism", "target_chembl",
        "srcp_score", "usrcat_score", "fusion_score", "srcp_support_label",
        "usrcat_support_label",
    ]
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mol_idx, item in enumerate(results, 1):
            for row in item["predictions"]:
                writer.writerow({
                    "molecule_index": mol_idx,
                    "input_smiles": item["input_smiles"],
                    "canonical_smiles": item["canonical_smiles"],
                    **{key: row.get(key, "") for key in fieldnames if key not in {"molecule_index", "input_smiles", "canonical_smiles"}},
                })


def parse_prediction_args():
    parser = argparse.ArgumentParser(description="Run PharmTAP SRCP + USRCAT target prediction for SMILES input.")
    parser.add_argument("--smiles", help="Single input SMILES string.")
    parser.add_argument("--smiles-file", help="Text/CSV file containing one SMILES per line.")
    parser.add_argument("--cluster-cache-dir", default=DEFAULT_CLUSTER_CACHE_DIR)
    parser.add_argument("--shape-cache-dir", default=DEFAULT_SHAPE_CACHE_DIR)
    parser.add_argument("--adaptive-stats-dir", default=DEFAULT_ADAPTIVE_STATS_DIR)
    parser.add_argument("--output-csv", default="outputs/target_predictions.csv")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--query-confs", type=int, default=10)
    parser.add_argument("--shape-prefilter-topk", type=int, default=300)
    parser.add_argument("--srcp-weight", type=float, default=0.85)
    parser.add_argument("--limit-targets", type=int)
    return parser.parse_args()


def main():
    args = parse_prediction_args()
    results = []
    for smiles in read_smiles_inputs(args):
        query = build_query_from_smiles(smiles, args.query_confs)
        predictions = predict_targets_for_query(
            query,
            cluster_cache_dir=args.cluster_cache_dir,
            shape_cache_dir=args.shape_cache_dir,
            adaptive_stats_dir=args.adaptive_stats_dir,
            top_n=args.top_n,
            srcp_weight=args.srcp_weight,
            shape_prefilter_topk=args.shape_prefilter_topk,
            limit_targets=args.limit_targets,
        )
        results.append({
            "input_smiles": smiles,
            "canonical_smiles": query["canonical_smiles"],
            "predictions": predictions,
        })
    write_prediction_csv(results, args.output_csv)
    print(f"Saved target predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
