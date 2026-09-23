import argparse
import os
import pickle
import time
from collections import Counter
from functools import partial
import multiprocessing as mp

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import RDLogger


# Keep native libraries from creating nested thread storms inside each process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

RDLogger.DisableLog("rdApp.*")

CLUSTER_PROFILE_CACHE_DIR = "outputs/target_prediction/srcp_target_profiles"
SHAPE_CACHE_DIR = "outputs/target_prediction/usrcat_target_profiles"

NUM_CONFS = 10
MAX_REPRESENTATIVES_PER_CLUSTER = 1
MAX_CLUSTERS_PER_TARGET = 80
MIN_CLUSTER_SIZE_FIRST = True
RANDOM_SEED = 42
OVERWRITE_EXISTING = False
PROGRESS_EVERY = 25


def init_worker():
    RDLogger.DisableLog("rdApp.*")


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def calculate_usrcat_descriptor(mol, conf_id=0):
    try:
        descriptor = rdMolDescriptors.GetUSRCAT(mol, confId=conf_id)
        return np.array(descriptor, dtype=np.float32)
    except Exception:
        return None


def generate_multiconf_usrcat_descriptors(smiles, num_confs=NUM_CONFS):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [], []

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED
        params.numThreads = 1
        params.pruneRmsThresh = 0.5
        params.maxIterations = 1000

        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        )
        if not conf_ids:
            return [], []

        opt_results = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=1, maxIters=150
        )

        descriptors = []
        energies = []
        for local_idx, conf_id in enumerate(conf_ids):
            descriptor = calculate_usrcat_descriptor(mol, conf_id)
            if descriptor is None or np.any(~np.isfinite(descriptor)):
                continue

            energy = None
            if local_idx < len(opt_results):
                energy = float(opt_results[local_idx][1])

            descriptors.append(descriptor)
            energies.append(energy)

        return descriptors, energies
    except Exception:
        return [], []


def representative_indices(cluster, max_representatives):
    indices = list(cluster["ligand_indices"])
    centroid = cluster["centroid_index"]
    ordered = [centroid] + [idx for idx in indices if idx != centroid]
    return ordered[:max_representatives]


def selected_clusters(clusters, max_clusters):
    if not MIN_CLUSTER_SIZE_FIRST:
        return clusters[:max_clusters]
    return sorted(
        clusters,
        key=lambda c: (c["size"], c["mean_internal_morgan_sim"]),
        reverse=True,
    )[:max_clusters]


def atomic_pickle_dump(data, out_path):
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, out_path)


def process_single_target_shape(
    cache_file,
    cluster_cache_dir,
    shape_cache_dir,
    num_confs,
    max_representatives,
    max_clusters,
    overwrite_existing,
):
    target_id = cache_file.replace(".pkl", "")
    cluster_path = os.path.join(cluster_cache_dir, cache_file)
    shape_path = os.path.join(shape_cache_dir, cache_file)

    if os.path.exists(shape_path) and not overwrite_existing:
        return {"target_id": target_id, "status": "exists"}

    started = time.time()
    try:
        with open(cluster_path, "rb") as f:
            cluster_cache = pickle.load(f)

        known_smiles = cluster_cache["known_smiles"]
        shape_clusters = []
        total_representatives = 0
        total_vectors = 0

        for cluster in selected_clusters(cluster_cache.get("clusters", []), max_clusters):
            reps = []
            for ligand_idx in representative_indices(cluster, max_representatives):
                if ligand_idx >= len(known_smiles):
                    continue

                smiles = known_smiles[ligand_idx]
                vectors, energies = generate_multiconf_usrcat_descriptors(
                    smiles, num_confs=num_confs
                )
                if not vectors:
                    continue

                reps.append({
                    "ligand_index": ligand_idx,
                    "smiles": smiles,
                    "is_centroid": ligand_idx == cluster["centroid_index"],
                    "vectors": [v.tolist() for v in vectors],
                    "energies": energies,
                    "conformer_count": len(vectors),
                })
                total_representatives += 1
                total_vectors += len(vectors)

            if reps:
                shape_clusters.append({
                    "cluster_id": cluster["cluster_id"],
                    "size": cluster["size"],
                    "main_scaffold": cluster["main_scaffold"],
                    "centroid_smiles": cluster["centroid_smiles"],
                    "mean_internal_morgan_sim": cluster["mean_internal_morgan_sim"],
                    "representatives": reps,
                })

        if not shape_clusters:
            return {
                "target_id": target_id,
                "status": "no_shape_vectors",
                "seconds": round(time.time() - started, 2),
            }

        shape_cache = {
            "schema_version": "target_3d_usrcat_profile_v1_linux",
            "target_id": target_id,
            "target_chembl": cluster_cache["target_chembl"],
            "target_gene": cluster_cache["target_gene"],
            "protein_name": cluster_cache["protein_name"],
            "target_type": cluster_cache["target_type"],
            "organism": cluster_cache["organism"],
            "shape_method": {
                "name": "multi_conformer_usrcat",
                "num_confs": num_confs,
                "max_representatives_per_cluster": max_representatives,
                "max_clusters_per_target": max_clusters,
                "random_seed": RANDOM_SEED,
                "platform": "linux_multiprocessing_fork",
            },
            "clusters": shape_clusters,
            "stats": {
                "shape_cluster_count": len(shape_clusters),
                "shape_representative_count": total_representatives,
                "shape_vector_count": total_vectors,
                "source_cluster_count": cluster_cache["stats"]["cluster_count"],
                "source_valid_ligands": cluster_cache["stats"]["valid_ligands"],
            },
        }

        atomic_pickle_dump(shape_cache, shape_path)

        return {
            "target_id": target_id,
            "status": "ok",
            "shape_clusters": len(shape_clusters),
            "vectors": total_vectors,
            "seconds": round(time.time() - started, 2),
        }
    except Exception as exc:
        return {
            "target_id": target_id,
            "status": "failed",
            "error": str(exc),
            "seconds": round(time.time() - started, 2),
        }


def list_pending_caches(cluster_cache_dir, shape_cache_dir, overwrite_existing):
    all_caches = [
        f for f in os.listdir(cluster_cache_dir) if f.endswith(".pkl")
    ]
    if not overwrite_existing:
        all_caches = [
            f for f in all_caches
            if not os.path.exists(os.path.join(shape_cache_dir, f))
        ]

    # Larger source pickle files usually represent heavier targets. Run them first
    # so long jobs start early and do not form a slow tail at the end.
    all_caches.sort(
        key=lambda f: os.path.getsize(os.path.join(cluster_cache_dir, f)),
        reverse=True,
    )
    return all_caches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Linux multi-core 3D shape caches from cluster profiles."
    )
    parser.add_argument("--cluster-cache-dir", default=CLUSTER_PROFILE_CACHE_DIR)
    parser.add_argument("--shape-cache-dir", default=SHAPE_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=max(available_cpus() - 1, 1))
    parser.add_argument("--num-confs", type=int, default=NUM_CONFS)
    parser.add_argument(
        "--max-representatives-per-cluster",
        type=int,
        default=MAX_REPRESENTATIVES_PER_CLUSTER,
    )
    parser.add_argument(
        "--max-clusters-per-target", type=int, default=MAX_CLUSTERS_PER_TARGET
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    parser.add_argument(
        "--maxtasksperchild",
        type=int,
        default=20,
        help="Recycle worker processes periodically to reduce long-run memory drift.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.cluster_cache_dir):
        raise FileNotFoundError(
            f"Cluster profile cache directory not found: {args.cluster_cache_dir}"
        )

    os.makedirs(args.shape_cache_dir, exist_ok=True)
    pending_caches = list_pending_caches(
        args.cluster_cache_dir, args.shape_cache_dir, args.overwrite
    )
    total_pending = len(pending_caches)

    print(f"Cluster cache dir: {args.cluster_cache_dir}")
    print(f"Shape cache dir: {args.shape_cache_dir}")
    print(f"Pending targets: {total_pending}")
    print(f"Workers: {args.workers}")
    print(
        f"num_confs={args.num_confs}, "
        f"max_representatives_per_cluster={args.max_representatives_per_cluster}, "
        f"max_clusters_per_target={args.max_clusters_per_target}"
    )

    if not pending_caches:
        print("Nothing to do. Existing shape caches are being kept.")
        return

    worker = partial(
        process_single_target_shape,
        cluster_cache_dir=args.cluster_cache_dir,
        shape_cache_dir=args.shape_cache_dir,
        num_confs=args.num_confs,
        max_representatives=args.max_representatives_per_cluster,
        max_clusters=args.max_clusters_per_target,
        overwrite_existing=args.overwrite,
    )

    counts = Counter()
    started = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=args.workers,
        initializer=init_worker,
        maxtasksperchild=args.maxtasksperchild,
    ) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, pending_caches, chunksize=1), 1):
            status = result.get("status", "unknown")
            counts[status] += 1

            if i % args.progress_every == 0 or i == total_pending:
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (total_pending - i) / rate if rate > 0 else 0.0
                print(
                    f"Progress: {i}/{total_pending} "
                    f"ok={counts['ok']} exists={counts['exists']} "
                    f"no_shape={counts['no_shape_vectors']} failed={counts['failed']} "
                    f"rate={rate:.2f} targets/s eta={eta/60:.1f} min"
                )

    elapsed = time.time() - started
    print(
        f"Done in {elapsed/60:.2f} min. "
        f"ok={counts['ok']}, exists={counts['exists']}, "
        f"no_shape={counts['no_shape_vectors']}, failed={counts['failed']}."
    )


if __name__ == "__main__":
    main()
