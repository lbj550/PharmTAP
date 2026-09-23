import argparse
import os
import pickle
import random
import time
from collections import Counter
import multiprocessing as mp

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger


# Avoid nested thread contention. Parallelism is handled by multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

RDLogger.DisableLog("rdApp.*")

CLUSTER_PROFILE_CACHE_DIR = "outputs/target_prediction/srcp_target_profiles"
SHAPE_CACHE_DIR = "outputs/target_prediction/usrcat_target_profiles"
ADAPTIVE_STATS_DIR = "outputs/target_prediction/adaptive_support_backgrounds"

DEFAULT_BACKGROUND_SAMPLES = 1000
DEFAULT_BACKGROUND_POOL_SIZE = 20000
RANDOM_SEED = 20260607
PROGRESS_EVERY = 25

G_CLUSTER_CACHE_DIR = None
G_SHAPE_CACHE_DIR = None
G_STATS_DIR = None
G_CLUSTER_BACKGROUND_ITEMS = None
G_SHAPE_BACKGROUND_ITEMS = None
G_SAMPLE_COUNT = None
G_OVERWRITE = None
G_ENABLE_CLUSTER = None
G_ENABLE_SHAPE = None


def init_worker():
    RDLogger.DisableLog("rdApp.*")


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def bitvect_from_bytes(fp_bytes):
    if isinstance(fp_bytes, str):
        fp = DataStructs.ExplicitBitVect(len(fp_bytes) * 8)
        fp.InitFromString(fp_bytes)
        return fp
    return DataStructs.ExplicitBitVect(fp_bytes)


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


def cluster_profile_score(query, cluster, known_morgan_fps):
    centroid_fp = bitvect_from_bytes(cluster["centroid_morgan_fp_bytes"])
    centroid_sim = DataStructs.TanimotoSimilarity(query["morgan_fp"], centroid_fp)

    member_fps = [
        known_morgan_fps[i]
        for i in cluster["ligand_indices"]
        if i < len(known_morgan_fps)
    ]
    if not member_fps:
        return 0.0
    member_sims = DataStructs.BulkTanimotoSimilarity(query["morgan_fp"], member_fps)
    top3_mean = float(np.mean(sorted(member_sims, reverse=True)[:3]))

    scaffold_score = 1.0 if query["scaffold"] == cluster["main_scaffold"] else 0.0
    prop_score = property_similarity(query["properties"], cluster["property_profile"])
    confidence = min(np.log1p(cluster["size"]) / np.log1p(50.0), 1.0)

    return (
        0.35 * centroid_sim
        + 0.30 * top3_mean
        + 0.15 * scaffold_score
        + 0.15 * prop_score
        + 0.05 * confidence
    )


def best_cluster_profile_score(query, target_cache, known_morgan_fps):
    best_score = 0.0
    for cluster in target_cache.get("clusters", []):
        score = cluster_profile_score(query, cluster, known_morgan_fps)
        if score > best_score:
            best_score = score
    return best_score


def vector_similarity(query_vector, ref_vector):
    query_vector = [float(x) for x in query_vector]
    ref_vector = [float(x) for x in ref_vector]
    return float(rdMolDescriptors.GetUSRScore(query_vector, ref_vector))


def representative_best_shape_score(query_vectors, representative):
    best_score = 0.0
    for q_vec in query_vectors:
        for ref_vec in representative.get("vectors", []):
            score = vector_similarity(q_vec, ref_vec)
            if score > best_score:
                best_score = score
    return best_score


def best_shape_profile_score(query, shape_cache):
    best_score = 0.0
    for cluster in shape_cache.get("clusters", []):
        rep_scores = [
            representative_best_shape_score(query["vectors"], rep)
            for rep in cluster.get("representatives", [])
        ]
        if not rep_scores:
            continue

        top3_mean = float(np.mean(sorted(rep_scores, reverse=True)[:3]))
        cluster_confidence = min(np.log1p(cluster["size"]) / np.log1p(50.0), 1.0)
        score = 0.85 * max(rep_scores) + 0.10 * top3_mean + 0.05 * cluster_confidence
        if score > best_score:
            best_score = score
    return best_score


def summary_stats(scores):
    arr = np.array(scores, dtype=np.float32)
    if len(arr) == 0:
        return {
            "scores": [],
            "n": 0,
            "mean": None,
            "std": None,
            "q50": None,
            "q90": None,
            "q95": None,
            "q99": None,
        }

    return {
        "scores": sorted(float(x) for x in arr),
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(max(np.std(arr), 1e-6)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
    }


def collect_cluster_background_items(cluster_cache_dir, max_items):
    items = []
    if not os.path.exists(cluster_cache_dir):
        return items

    files = [f for f in os.listdir(cluster_cache_dir) if f.endswith(".pkl")]
    random.Random(RANDOM_SEED).shuffle(files)

    for file_name in files:
        path = os.path.join(cluster_cache_dir, file_name)
        target_id = file_name.replace(".pkl", "")
        try:
            with open(path, "rb") as f:
                cache = pickle.load(f)
            known_properties = cache.get("known_properties", [])
            for cluster in cache.get("clusters", []):
                centroid_idx = cluster["centroid_index"]
                if centroid_idx >= len(known_properties):
                    continue
                items.append({
                    "source_target": target_id,
                    "morgan_fp_bytes": cluster["centroid_morgan_fp_bytes"],
                    "scaffold": cluster["main_scaffold"],
                    "properties": known_properties[centroid_idx],
                })
                if len(items) >= max_items:
                    return items
        except Exception:
            continue
    return items


def collect_shape_background_items(shape_cache_dir, max_items):
    items = []
    if not os.path.exists(shape_cache_dir):
        return items

    files = [f for f in os.listdir(shape_cache_dir) if f.endswith(".pkl")]
    random.Random(RANDOM_SEED).shuffle(files)

    for file_name in files:
        path = os.path.join(shape_cache_dir, file_name)
        target_id = file_name.replace(".pkl", "")
        try:
            with open(path, "rb") as f:
                cache = pickle.load(f)
            for cluster in cache.get("clusters", []):
                for rep in cluster.get("representatives", []):
                    vectors = rep.get("vectors")
                    if vectors:
                        items.append({
                            "source_target": target_id,
                            "vectors": vectors,
                        })
                    if len(items) >= max_items:
                        return items
        except Exception:
            continue
    return items


def choose_background(items, target_id, sample_count):
    candidates = [item for item in items if item["source_target"] != target_id]
    rng = random.Random(f"{RANDOM_SEED}:{target_id}")
    if len(candidates) <= sample_count:
        rng.shuffle(candidates)
        return candidates
    return rng.sample(candidates, sample_count)


def atomic_pickle_dump(data, out_path):
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, out_path)


def process_target_stats(cache_file):
    target_id = cache_file.replace(".pkl", "")
    out_path = os.path.join(G_STATS_DIR, cache_file)
    if os.path.exists(out_path) and not G_OVERWRITE:
        return {"target_id": target_id, "status": "exists"}

    started = time.time()
    cluster_stats = None
    shape_stats = None
    target_meta = {"target_id": target_id}

    if G_ENABLE_CLUSTER:
        try:
            cluster_path = os.path.join(G_CLUSTER_CACHE_DIR, cache_file)
            if os.path.exists(cluster_path):
                with open(cluster_path, "rb") as f:
                    cluster_cache = pickle.load(f)
                target_meta.update({
                    "target_chembl": cluster_cache.get("target_chembl"),
                    "target_gene": cluster_cache.get("target_gene"),
                    "protein_name": cluster_cache.get("protein_name"),
                    "target_type": cluster_cache.get("target_type"),
                    "organism": cluster_cache.get("organism"),
                    "known_ligands_count": cluster_cache.get("stats", {}).get(
                        "valid_ligands"
                    ),
                    "cluster_count": cluster_cache.get("stats", {}).get(
                        "cluster_count"
                    ),
                })

                known_morgan_fps = [
                    bitvect_from_bytes(fp_bytes)
                    for fp_bytes in cluster_cache["known_morgan_fps_bytes"]
                ]

                scores = []
                background = choose_background(
                    G_CLUSTER_BACKGROUND_ITEMS, target_id, G_SAMPLE_COUNT
                )
                for item in background:
                    query = {
                        "morgan_fp": bitvect_from_bytes(item["morgan_fp_bytes"]),
                        "scaffold": item["scaffold"],
                        "properties": item["properties"],
                    }
                    scores.append(
                        best_cluster_profile_score(
                            query, cluster_cache, known_morgan_fps
                        )
                    )
                cluster_stats = summary_stats(scores)
        except Exception as exc:
            cluster_stats = {"error": str(exc), "scores": [], "n": 0}

    if G_ENABLE_SHAPE:
        try:
            shape_path = os.path.join(G_SHAPE_CACHE_DIR, cache_file)
            if os.path.exists(shape_path):
                with open(shape_path, "rb") as f:
                    shape_cache = pickle.load(f)
                target_meta.update({
                    "shape_cluster_count": shape_cache.get("stats", {}).get(
                        "shape_cluster_count"
                    ),
                    "shape_representative_count": shape_cache.get("stats", {}).get(
                        "shape_representative_count"
                    ),
                    "shape_vector_count": shape_cache.get("stats", {}).get(
                        "shape_vector_count"
                    ),
                })

                scores = []
                background = choose_background(
                    G_SHAPE_BACKGROUND_ITEMS, target_id, G_SAMPLE_COUNT
                )
                for item in background:
                    scores.append(best_shape_profile_score(item, shape_cache))
                shape_stats = summary_stats(scores)
        except Exception as exc:
            shape_stats = {"error": str(exc), "scores": [], "n": 0}

    if cluster_stats is None and shape_stats is None:
        return {
            "target_id": target_id,
            "status": "no_source_cache",
            "seconds": round(time.time() - started, 2),
        }

    atomic_pickle_dump(
        {
            "schema_version": "target_adaptive_stats_v1_linux",
            "target": target_meta,
            "background_sample_count": G_SAMPLE_COUNT,
            "cluster_profile": cluster_stats,
            "shape_profile": shape_stats,
        },
        out_path,
    )

    return {
        "target_id": target_id,
        "status": "ok",
        "seconds": round(time.time() - started, 2),
    }


def list_target_files(cluster_cache_dir, shape_cache_dir, stats_dir, overwrite):
    cluster_files = set()
    if os.path.exists(cluster_cache_dir):
        cluster_files = {
            f for f in os.listdir(cluster_cache_dir) if f.endswith(".pkl")
        }
    shape_files = set()
    if os.path.exists(shape_cache_dir):
        shape_files = {
            f for f in os.listdir(shape_cache_dir) if f.endswith(".pkl")
        }
    all_files = list(cluster_files | shape_files)

    if not overwrite:
        all_files = [
            f for f in all_files
            if not os.path.exists(os.path.join(stats_dir, f))
        ]

    def source_size(file_name):
        paths = [
            os.path.join(cluster_cache_dir, file_name),
            os.path.join(shape_cache_dir, file_name),
        ]
        return sum(os.path.getsize(p) for p in paths if os.path.exists(p))

    all_files.sort(key=source_size, reverse=True)
    return all_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Linux multi-core adaptive background statistics."
    )
    parser.add_argument("--cluster-cache-dir", default=CLUSTER_PROFILE_CACHE_DIR)
    parser.add_argument("--shape-cache-dir", default=SHAPE_CACHE_DIR)
    parser.add_argument("--stats-dir", default=ADAPTIVE_STATS_DIR)
    parser.add_argument("--samples", type=int, default=DEFAULT_BACKGROUND_SAMPLES)
    parser.add_argument("--background-pool-size", type=int, default=DEFAULT_BACKGROUND_POOL_SIZE)
    parser.add_argument("--workers", type=int, default=max(available_cpus() - 1, 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cluster-only", action="store_true")
    parser.add_argument("--shape-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    parser.add_argument(
        "--maxtasksperchild",
        type=int,
        default=20,
        help="Recycle worker processes periodically to reduce long-run memory drift.",
    )
    return parser.parse_args()


def main():
    global G_CLUSTER_CACHE_DIR
    global G_SHAPE_CACHE_DIR
    global G_STATS_DIR
    global G_CLUSTER_BACKGROUND_ITEMS
    global G_SHAPE_BACKGROUND_ITEMS
    global G_SAMPLE_COUNT
    global G_OVERWRITE
    global G_ENABLE_CLUSTER
    global G_ENABLE_SHAPE

    args = parse_args()
    if args.cluster_only and args.shape_only:
        raise ValueError("Use at most one of --cluster-only or --shape-only.")

    enable_cluster = not args.shape_only
    enable_shape = not args.cluster_only

    os.makedirs(args.stats_dir, exist_ok=True)
    pending_files = list_target_files(
        args.cluster_cache_dir, args.shape_cache_dir, args.stats_dir, args.overwrite
    )

    print(f"Cluster cache dir: {args.cluster_cache_dir}")
    print(f"3D shape cache dir: {args.shape_cache_dir}")
    print(f"Adaptive stats dir: {args.stats_dir}")
    print(f"Pending targets: {len(pending_files)}")
    print(f"Workers: {args.workers}")
    print(
        f"samples={args.samples}, background_pool_size={args.background_pool_size}, "
        f"enable_cluster={enable_cluster}, enable_shape={enable_shape}"
    )

    if not pending_files:
        print("Nothing to do. Existing adaptive stats are being kept.")
        return

    cluster_background_items = []
    if enable_cluster:
        print("Collecting global cluster background pool...")
        cluster_background_items = collect_cluster_background_items(
            args.cluster_cache_dir, args.background_pool_size
        )
        print(f"Cluster background items: {len(cluster_background_items)}")

    shape_background_items = []
    if enable_shape:
        print("Collecting global 3D shape background pool...")
        shape_background_items = collect_shape_background_items(
            args.shape_cache_dir, args.background_pool_size
        )
        print(f"Shape background items: {len(shape_background_items)}")

    G_CLUSTER_CACHE_DIR = args.cluster_cache_dir
    G_SHAPE_CACHE_DIR = args.shape_cache_dir
    G_STATS_DIR = args.stats_dir
    G_CLUSTER_BACKGROUND_ITEMS = cluster_background_items
    G_SHAPE_BACKGROUND_ITEMS = shape_background_items
    G_SAMPLE_COUNT = args.samples
    G_OVERWRITE = args.overwrite
    G_ENABLE_CLUSTER = enable_cluster
    G_ENABLE_SHAPE = enable_shape

    counts = Counter()
    started = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=args.workers,
        initializer=init_worker,
        maxtasksperchild=args.maxtasksperchild,
    ) as pool:
        for i, result in enumerate(
            pool.imap_unordered(process_target_stats, pending_files, chunksize=1),
            1,
        ):
            status = result.get("status", "failed")
            counts[status] += 1
            if i % args.progress_every == 0 or i == len(pending_files):
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (len(pending_files) - i) / rate if rate > 0 else 0.0
                print(
                    f"Progress: {i}/{len(pending_files)} "
                    f"ok={counts['ok']} exists={counts['exists']} "
                    f"no_source={counts['no_source_cache']} failed={counts['failed']} "
                    f"rate={rate:.2f} targets/s eta={eta/60:.1f} min"
                )

    elapsed = time.time() - started
    print(
        f"Done in {elapsed/60:.2f} min. "
        f"ok={counts['ok']}, exists={counts['exists']}, "
        f"no_source={counts['no_source_cache']}, failed={counts['failed']}."
    )


if __name__ == "__main__":
    main()
