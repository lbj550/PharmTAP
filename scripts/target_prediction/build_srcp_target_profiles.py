import argparse
import os
import pickle
import time
from collections import Counter, defaultdict
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

RDLogger.DisableLog("rdApp.*")

DATA_ROOT = "data/target_ligands"
CACHE_DIR = "outputs/target_prediction/srcp_target_profiles"

MORGAN_RADIUS = 2
FP_SIZE = 2048
BUTINA_DISTANCE_CUTOFF = 0.35
MAX_BUTINA_GROUP_SIZE = 2000
MIN_CLUSTER_SIZE_FOR_PROFILE = 1


def init_worker():
    RDLogger.DisableLog("rdApp.*")


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


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


def get_rdk_fp(mol):
    return Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=FP_SIZE)


def get_morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=MORGAN_RADIUS, nBits=FP_SIZE
    )


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


def property_profile(properties):
    keys = list(properties[0].keys())
    profile = {}
    for key in keys:
        values = np.array([p[key] for p in properties], dtype=float)
        profile[f"{key}_mean"] = round(float(np.mean(values)), 6)
        profile[f"{key}_std"] = round(float(np.std(values)), 6)
    return profile


def affinity_value(row):
    values = []
    for col in ["Ki (nM)", "IC50 (nM)", "Kd (nM)"]:
        if col in row and pd.notna(row[col]):
            try:
                val = float(row[col])
                if val > 0:
                    values.append(val)
            except Exception:
                pass
    return min(values) if values else None


def butina_clusters_for_indices(indices, fps):
    if len(indices) <= 1:
        return [tuple(indices)]

    dists = []
    for i in range(1, len(indices)):
        sims = DataStructs.BulkTanimotoSimilarity(
            fps[indices[i]], [fps[j] for j in indices[:i]]
        )
        dists.extend(1.0 - sim for sim in sims)

    local_clusters = Butina.ClusterData(
        dists,
        len(indices),
        BUTINA_DISTANCE_CUTOFF,
        isDistData=True,
        reordering=True,
    )
    return [tuple(indices[i] for i in cluster) for cluster in local_clusters]


def split_large_group(indices, props):
    sorted_indices = sorted(indices, key=lambda i: props[i]["mw"])
    return [
        sorted_indices[i:i + MAX_BUTINA_GROUP_SIZE]
        for i in range(0, len(sorted_indices), MAX_BUTINA_GROUP_SIZE)
    ]


def scaffold_then_butina_clusters(mols, fps, props):
    scaffold_groups = defaultdict(list)
    for idx, mol in enumerate(mols):
        scaffold_groups[get_scaffold_smiles(mol)].append(idx)

    clusters = []
    for indices in scaffold_groups.values():
        chunks = (
            split_large_group(indices, props)
            if len(indices) > MAX_BUTINA_GROUP_SIZE
            else [indices]
        )
        for chunk in chunks:
            clusters.extend(butina_clusters_for_indices(chunk, fps))

    return [
        tuple(cluster)
        for cluster in clusters
        if len(cluster) >= MIN_CLUSTER_SIZE_FOR_PROFILE
    ]


def choose_centroid(cluster_indices, fps):
    if len(cluster_indices) == 1:
        return cluster_indices[0], 1.0

    best_idx = cluster_indices[0]
    best_score = -1.0
    for idx in cluster_indices:
        others = [fps[j] for j in cluster_indices if j != idx]
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx], others)
        mean_sim = float(np.mean(sims)) if sims else 1.0
        if mean_sim > best_score:
            best_idx = idx
            best_score = mean_sim
    return best_idx, best_score


def mean_internal_similarity(cluster_indices, fps):
    if len(cluster_indices) <= 1:
        return 1.0

    sims = []
    for i in range(1, len(cluster_indices)):
        idx = cluster_indices[i]
        prev = [fps[j] for j in cluster_indices[:i]]
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[idx], prev))
    return float(np.mean(sims)) if sims else 1.0


def make_cluster_profile(cluster_id, cluster_indices, smiles, mols, fps, props, affinities):
    centroid_idx, centroid_mean_sim = choose_centroid(cluster_indices, fps)
    scaffold_counts = Counter(get_scaffold_smiles(mols[i]) for i in cluster_indices)
    main_scaffold, main_scaffold_count = scaffold_counts.most_common(1)[0]
    cluster_affinities = [
        affinities[i] for i in cluster_indices if affinities[i] is not None
    ]

    profile = {
        "cluster_id": cluster_id,
        "ligand_indices": list(cluster_indices),
        "size": len(cluster_indices),
        "centroid_index": centroid_idx,
        "centroid_smiles": smiles[centroid_idx],
        "centroid_morgan_fp_bytes": fps[centroid_idx].ToBinary(),
        "main_scaffold": main_scaffold,
        "main_scaffold_fraction": round(main_scaffold_count / len(cluster_indices), 6),
        "mean_internal_morgan_sim": round(
            mean_internal_similarity(cluster_indices, fps), 6
        ),
        "centroid_mean_morgan_sim": round(centroid_mean_sim, 6),
        "property_profile": property_profile([props[i] for i in cluster_indices]),
    }

    if cluster_affinities:
        profile["best_activity_nm"] = round(float(min(cluster_affinities)), 6)
        profile["median_activity_nm"] = round(float(np.median(cluster_affinities)), 6)
    else:
        profile["best_activity_nm"] = None
        profile["median_activity_nm"] = None

    return profile


def first_non_empty(df, column):
    if not column:
        return "N/A"
    values = df[column].dropna()
    return values.iloc[0] if not values.empty else "N/A"


def target_metadata(ligand_df):
    chembl_col = next((c for c in ligand_df.columns if "CHEMBL" in c.upper()), None)
    gene_col = next((c for c in ligand_df.columns if "GENE" in c.upper()), None)
    protein_col = next(
        (
            c
            for c in ligand_df.columns
            if "PROTEIN" in c.upper()
            or ("NAME" in c.upper() and "SMILES" not in c.upper())
        ),
        None,
    )
    type_col = next((c for c in ligand_df.columns if "TYPE" in c.upper()), None)
    org_col = next((c for c in ligand_df.columns if "ORGANISM" in c.upper()), None)

    return {
        "target_chembl": first_non_empty(ligand_df, chembl_col),
        "target_gene": first_non_empty(ligand_df, gene_col),
        "protein_name": first_non_empty(ligand_df, protein_col),
        "target_type": first_non_empty(ligand_df, type_col),
        "organism": first_non_empty(ligand_df, org_col),
    }


def process_single_target(target_id, data_root, cache_dir):
    ligand_file = os.path.join(data_root, target_id, f"{target_id}_ligands.csv")
    if not os.path.exists(ligand_file):
        return {"target_id": target_id, "status": "missing_csv"}

    try:
        ligand_df = pd.read_csv(ligand_file)
        smiles_col = next(c for c in ligand_df.columns if "SMILES" in c.upper())
    except Exception as exc:
        return {"target_id": target_id, "status": "read_failed", "error": str(exc)}

    metadata = target_metadata(ligand_df)
    smiles = []
    mols = []
    rdk_fps = []
    morgan_fps = []
    props = []
    affinities = []

    for _, row in ligand_df.iterrows():
        mol = standardize_smiles(row[smiles_col])
        if mol is None:
            continue
        smiles.append(Chem.MolToSmiles(mol, canonical=True))
        mols.append(mol)
        rdk_fps.append(get_rdk_fp(mol))
        morgan_fps.append(get_morgan_fp(mol))
        props.append(calc_properties(mol))
        affinities.append(affinity_value(row))

    if not mols:
        return {"target_id": target_id, "status": "no_valid_ligands"}

    clusters = scaffold_then_butina_clusters(mols, morgan_fps, props)
    cluster_profiles = [
        make_cluster_profile(i, cluster, smiles, mols, morgan_fps, props, affinities)
        for i, cluster in enumerate(clusters)
    ]
    cluster_profiles.sort(
        key=lambda c: (c["size"], c["mean_internal_morgan_sim"]),
        reverse=True,
    )

    cache_data = {
        **metadata,
        "schema_version": "target_cluster_profile_v1",
        "target_id": target_id,
        "known_smiles": smiles,
        "known_rdk_fps_bytes": [fp.ToBinary() for fp in rdk_fps],
        "known_morgan_fps_bytes": [fp.ToBinary() for fp in morgan_fps],
        "known_properties": props,
        "known_affinity_nm": affinities,
        "cluster_method": {
            "name": "murcko_scaffold_then_butina",
            "morgan_radius": MORGAN_RADIUS,
            "fp_size": FP_SIZE,
            "butina_distance_cutoff": BUTINA_DISTANCE_CUTOFF,
            "max_butina_group_size": MAX_BUTINA_GROUP_SIZE,
        },
        "clusters": cluster_profiles,
        "stats": {
            "input_rows": int(len(ligand_df)),
            "valid_ligands": int(len(mols)),
            "cluster_count": int(len(cluster_profiles)),
            "singleton_cluster_count": int(
                sum(1 for cluster in cluster_profiles if cluster["size"] == 1)
            ),
        },
    }

    os.makedirs(cache_dir, exist_ok=True)
    out_file = os.path.join(cache_dir, f"{target_id}.pkl")
    tmp_file = f"{out_file}.tmp.{os.getpid()}"
    with open(tmp_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_file, out_file)

    return {
        "target_id": target_id,
        "status": "ok",
        "valid_ligands": len(mols),
        "clusters": len(cluster_profiles),
    }


def process_target(target_id, data_root, cache_dir, overwrite):
    out_file = os.path.join(cache_dir, f"{target_id}.pkl")
    if os.path.exists(out_file) and not overwrite:
        return {"target_id": target_id, "status": "exists"}
    return process_single_target(target_id, data_root, cache_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build standalone Linux multi-core target cluster profile caches."
    )
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--workers", type=int, default=max(available_cpus() - 1, 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--maxtasksperchild", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    target_ids = [
        d
        for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    ]
    target_ids.sort()
    print(f"Data root: {args.data_root}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Targets: {len(target_ids)}")
    print(f"Workers: {args.workers}")

    worker = partial(
        process_target,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
    )

    counts = Counter()
    started = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=args.workers,
        initializer=init_worker,
        maxtasksperchild=args.maxtasksperchild,
    ) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, target_ids, chunksize=1), 1):
            status = result.get("status", "failed")
            counts[status] += 1
            if i % args.progress_every == 0 or i == len(target_ids):
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (len(target_ids) - i) / rate if rate > 0 else 0.0
                print(
                    f"Progress: {i}/{len(target_ids)} "
                    f"ok={counts['ok']} exists={counts['exists']} "
                    f"missing={counts['missing_csv']} no_valid={counts['no_valid_ligands']} "
                    f"read_failed={counts['read_failed']} failed={counts['failed']} "
                    f"rate={rate:.2f} targets/s eta={eta/60:.1f} min"
                )

    print(f"Done. Cluster profile caches saved to: {args.cache_dir}")


if __name__ == "__main__":
    main()
