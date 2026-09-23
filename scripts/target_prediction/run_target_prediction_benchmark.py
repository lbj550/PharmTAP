import argparse
import hashlib
import os
import pickle
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from rdkit import DataStructs

import predict_targets as ab


PROFILE_ROOT = "outputs/target_prediction"
BENCHMARK_ROOT = "data/external_benchmark"
CLUSTER_CACHE_DIR = os.path.join(PROFILE_ROOT, "srcp_target_profiles")
SHAPE_CACHE_DIR = os.path.join(PROFILE_ROOT, "usrcat_target_profiles")
ADAPTIVE_STATS_DIR = os.path.join(PROFILE_ROOT, "adaptive_support_backgrounds")
OUTPUT_DIR = "outputs/target_prediction_benchmark"
DATASET_NAME = "External_BindingDB_ChEMBL_202604"
G_SHAPE_QUERY_FILTER_BY_TARGET = None
SEA_TC_THRESHOLD = 0.57
SEA_TOPK_PER_CLUSTER = 10
SEA_SIZE_NORMALIZATION_ALPHA = 0.5

METHODS = [
    "Baseline_2D_MaxTanimoto",
    "Cluster_Profile",
    "Cluster_Profile_Adaptive",
    "Cluster_SEA_Raw",
    "Cluster_SEA_Adaptive",
    "USRCAT_Shape",
    "USRCAT_Shape_Adaptive",
    "Cluster_Profile_Plus_USRCAT_Raw",
    "Cluster_SEA_Plus_USRCAT_Raw",
    "Final_Full_Adaptive",
]

ADAPTIVE_RANKING_METHODS = {
    "Cluster_Profile_Adaptive",
    "Cluster_SEA_Adaptive",
    "USRCAT_Shape_Adaptive",
    "Final_Full_Adaptive",
}


def log_step(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def query_signature(queries):
    digest = hashlib.md5()
    for query in queries:
        digest.update(str(query.get("source_index", "")).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(query.get("canonical_smiles", "")).encode("utf-8"))
        digest.update(b"\t")
        digest.update(";".join(query.get("true_targets", [])).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def atomic_pickle_dump(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def score_checkpoint_path(checkpoint_dir, cache_file):
    return os.path.join(checkpoint_dir, cache_file)


def valid_checkpoint_result(result, query_count):
    if not isinstance(result, dict) or "scores" not in result:
        return False
    for method in METHODS:
        scores = result["scores"].get(method)
        if scores is None or len(scores) != query_count:
            return False
    return True


def parse_weight_list(weight_text):
    weights = []
    for token in str(weight_text).split(","):
        token = token.strip()
        if not token:
            continue
        weight = float(token)
        if weight < 0.0 or weight > 1.0:
            raise ValueError(f"Fusion weight must be in [0, 1], got {weight}")
        weights.append(weight)
    if not weights:
        raise ValueError("At least one fusion weight is required.")
    return weights


def weight_label(weight):
    return f"W{int(round(weight * 100)):02d}"


def weighted_fusion_matrix(primary, secondary, primary_weight):
    primary = np.asarray(primary, dtype=np.float32)
    secondary = np.asarray(secondary, dtype=np.float32)
    fused = np.full(primary.shape, np.nan, dtype=np.float32)

    primary_ok = np.isfinite(primary)
    secondary_ok = np.isfinite(secondary)
    both_ok = primary_ok & secondary_ok
    fused[both_ok] = (
        primary_weight * primary[both_ok]
        + (1.0 - primary_weight) * secondary[both_ok]
    )
    fused[primary_ok & ~secondary_ok] = primary[primary_ok & ~secondary_ok]
    fused[~primary_ok & secondary_ok] = secondary[~primary_ok & secondary_ok]
    return fused


def rank_percentile_matrix(score_matrix):
    scores = np.asarray(score_matrix, dtype=np.float32)
    rank_scores = np.full(scores.shape, np.nan, dtype=np.float32)
    for q_idx in range(scores.shape[1]):
        col = scores[:, q_idx]
        finite = np.where(np.isfinite(col))[0]
        n = len(finite)
        if n == 0:
            continue
        order = finite[np.argsort(-col[finite], kind="mergesort")]
        if n == 1:
            rank_scores[order[0], q_idx] = 1.0
            continue
        values = 1.0 - (np.arange(n, dtype=np.float32) / float(n - 1))
        rank_scores[order, q_idx] = values
    return rank_scores


def add_fusion_score_matrices(score_mats, fusion_weights):
    derived_methods = []
    if "USRCAT_Shape" not in score_mats:
        return derived_methods

    shape = score_mats["USRCAT_Shape"]
    shape_rank = rank_percentile_matrix(shape)

    fusion_specs = [
        ("Cluster_Profile", "Cluster_Profile_Plus_USRCAT"),
        ("Cluster_SEA_Raw", "Cluster_SEA_Plus_USRCAT"),
    ]
    for primary_method, prefix in fusion_specs:
        if primary_method not in score_mats:
            continue
        primary = score_mats[primary_method]
        primary_rank = rank_percentile_matrix(primary)
        for weight in fusion_weights:
            label = weight_label(weight)
            fixed_name = f"{prefix}_FixedFusion_{label}"
            rank_name = f"{prefix}_RankFusion_{label}"
            score_mats[fixed_name] = weighted_fusion_matrix(primary, shape, weight)
            score_mats[rank_name] = weighted_fusion_matrix(
                primary_rank, shape_rank, weight
            )
            derived_methods.extend([fixed_name, rank_name])
    return derived_methods


def adaptive_level(score):
    if score is None or not np.isfinite(score):
        return "NA"
    if score >= 0.99:
        return "Very high background-adjusted support"
    if score >= 0.95:
        return "High background-adjusted support"
    if score >= 0.90:
        return "Moderate background-adjusted support"
    return "Not background-significant"


def choose_significance_method(score_mats, requested_method, derived_methods):
    if requested_method and requested_method in score_mats:
        return requested_method
    if requested_method:
        log_step(
            f"Requested significance method {requested_method} not found; "
            "falling back to available proposed method."
        )
    preferred = [
        "Cluster_Profile_Plus_USRCAT_FixedFusion_W95",
        "Cluster_Profile_Plus_USRCAT_FixedFusion_W98",
        "Cluster_Profile_Plus_USRCAT_RankFusion_W95",
        "Cluster_SEA_Plus_USRCAT_RankFusion_W90",
        "Cluster_SEA_Plus_USRCAT_FixedFusion_W90",
        "Cluster_SEA_Plus_USRCAT_Raw",
        "Cluster_SEA_Raw",
    ]
    for method in preferred + list(derived_methods):
        if method in score_mats:
            return method
    return None


def build_adaptive_significance_table(
    score_mats,
    target_ids,
    queries,
    ranking_method,
    top_n,
):
    if not ranking_method or top_n <= 0 or ranking_method not in score_mats:
        return pd.DataFrame()

    rows = []
    ranking_scores = score_mats[ranking_method]
    sea_scores = score_mats.get("Cluster_SEA_Raw")
    shape_scores = score_mats.get("USRCAT_Shape")
    sea_adaptive = score_mats.get("Cluster_SEA_Adaptive")
    shape_adaptive = score_mats.get("USRCAT_Shape_Adaptive")

    for q_idx, query in enumerate(queries):
        scores = np.array(ranking_scores[:, q_idx], dtype=float)
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        finite = np.where(np.isfinite(scores) & (scores > -np.inf))[0]
        if len(finite) == 0:
            continue
        n_top = min(top_n, len(finite))
        order = finite[np.argsort(-scores[finite], kind="mergesort")[:n_top]]
        true_targets = set(query["true_targets"])

        for rank, target_idx in enumerate(order, 1):
            target = target_ids[target_idx]
            sea_ad = (
                float(sea_adaptive[target_idx, q_idx])
                if sea_adaptive is not None
                and np.isfinite(sea_adaptive[target_idx, q_idx])
                else np.nan
            )
            shape_ad = (
                float(shape_adaptive[target_idx, q_idx])
                if shape_adaptive is not None
                and np.isfinite(shape_adaptive[target_idx, q_idx])
                else np.nan
            )
            combined_adaptive = np.nanmean([sea_ad, shape_ad])
            rows.append({
                "Ranking_Method": ranking_method,
                "Query_Index": query["source_index"],
                "SMILES": query["smiles"],
                "Rank": rank,
                "Predicted_Target": target,
                "Is_True_Target": target in true_targets,
                "Final_Ranking_Score": float(scores[target_idx]),
                "Cluster_SEA_Raw": (
                    float(sea_scores[target_idx, q_idx])
                    if sea_scores is not None
                    and np.isfinite(sea_scores[target_idx, q_idx])
                    else np.nan
                ),
                "USRCAT_Shape": (
                    float(shape_scores[target_idx, q_idx])
                    if shape_scores is not None
                    and np.isfinite(shape_scores[target_idx, q_idx])
                    else np.nan
                ),
                "Cluster_SEA_Adaptive_Calibrated": sea_ad,
                "USRCAT_Shape_Adaptive_Calibrated": shape_ad,
                "Mean_Adaptive_Calibrated": combined_adaptive,
                "Adaptive_Interpretation": adaptive_level(combined_adaptive),
            })
    return pd.DataFrame(rows)


def default_roc_methods(score_mats):
    preferred = [
        "Cluster_Profile",
        "Cluster_SEA_Raw",
        "USRCAT_Shape",
        "Cluster_Profile_Plus_USRCAT_Raw",
        "Cluster_SEA_Plus_USRCAT_Raw",
    ]
    return [method for method in preferred if method in score_mats]


def parse_method_list(text, score_mats):
    if not text:
        return default_roc_methods(score_mats)
    methods = [token.strip() for token in str(text).split(",") if token.strip()]
    missing = [method for method in methods if method not in score_mats]
    if missing:
        log_step("ROC methods not found and skipped: " + ", ".join(missing))
    return [method for method in methods if method in score_mats]


def macro_average_roc_curve(score_matrix, target_ids, queries, fpr_grid):
    target_index = {target: idx for idx, target in enumerate(target_ids)}
    tpr_sum = np.zeros_like(fpr_grid, dtype=np.float64)
    auc_values = []
    evaluable = 0

    for q_idx, query in enumerate(queries):
        true_indices = [
            target_index[target]
            for target in query["true_targets"]
            if target in target_index
        ]
        if not true_indices:
            continue

        scores = np.asarray(score_matrix[:, q_idx], dtype=np.float64)
        finite = np.isfinite(scores)
        if not finite.any():
            continue
        labels = np.zeros(len(target_ids), dtype=np.int8)
        labels[true_indices] = 1
        scores = np.where(finite, scores, -np.inf)
        order = np.argsort(-scores, kind="mergesort")
        ranked_labels = labels[order]
        positives = int(ranked_labels.sum())
        negatives = len(ranked_labels) - positives
        if positives == 0 or negatives == 0:
            continue

        tp = np.cumsum(ranked_labels, dtype=np.float64)
        fp = np.cumsum(1 - ranked_labels, dtype=np.float64)
        tpr = np.concatenate(([0.0], tp / positives, [1.0]))
        fpr = np.concatenate(([0.0], fp / negatives, [1.0]))
        # Keep monotonic FPR coordinates unique for interpolation.
        unique_fpr, unique_idx = np.unique(fpr, return_index=True)
        unique_tpr = np.maximum.accumulate(tpr[unique_idx])
        interp_tpr = np.interp(fpr_grid, unique_fpr, unique_tpr)
        tpr_sum += interp_tpr
        auc_values.append(float(np.trapz(unique_tpr, unique_fpr)))
        evaluable += 1

    if evaluable == 0:
        return np.full_like(fpr_grid, np.nan, dtype=np.float64), np.nan, 0
    return tpr_sum / evaluable, float(np.nanmean(auc_values)), evaluable


def write_macro_roc_outputs(
    score_mats,
    target_ids,
    queries,
    output_dir,
    roc_methods,
    grid_points,
):
    if not roc_methods:
        log_step("No ROC methods requested; skipping ROC output.")
        return None, None

    fpr_grid = np.linspace(0.0, 1.0, int(grid_points), dtype=np.float64)
    curve_rows = []
    summary_rows = []
    for method in roc_methods:
        log_step(f"Computing macro-average ROC curve: {method}")
        tpr, auc, evaluable = macro_average_roc_curve(
            score_mats[method],
            target_ids,
            queries,
            fpr_grid,
        )
        summary_rows.append({
            "Dataset": DATASET_NAME,
            "Method": method,
            "Macro_ROC_AUC": auc,
            "Queries_Evaluable": evaluable,
        })
        for fpr, value in zip(fpr_grid, tpr):
            curve_rows.append({
                "Dataset": DATASET_NAME,
                "Method": method,
                "FPR": float(fpr),
                "TPR": float(value),
            })

    roc_df = pd.DataFrame(curve_rows)
    summary_df = pd.DataFrame(summary_rows)
    roc_path = os.path.join(
        output_dir, "external_bindingdb_chembl202604_macro_roc_curve.csv"
    )
    summary_path = os.path.join(
        output_dir, "external_bindingdb_chembl202604_macro_roc_summary.csv"
    )
    roc_df.to_csv(roc_path, index=False, encoding="utf_8_sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf_8_sig")

    svg_path = os.path.join(
        output_dir, "external_bindingdb_chembl202604_macro_roc_curve.svg"
    )
    write_macro_roc_svg(roc_df, summary_df, svg_path)
    return roc_path, svg_path


def short_method_name(method):
    names = {
        "Cluster_Profile": "Cluster Profile",
        "Cluster_SEA_Raw": "Cluster SEA",
        "USRCAT_Shape": "USRCAT Shape",
        "Cluster_Profile_Plus_USRCAT_Raw": "Profile + USRCAT",
        "Cluster_SEA_Plus_USRCAT_Raw": "SEA + USRCAT",
    }
    return names.get(method, method)


def write_macro_roc_svg(roc_df, summary_df, path):
    colors = ["#4f7ecb", "#e07b39", "#8f8f8f", "#67a45b", "#b65aa5", "#2f9aa0"]
    width, height = 760, 520
    left, right, top, bottom = 82, 230, 58, 74
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(x):
        return left + x * plot_w

    def y_pos(y):
        return top + (1.0 - y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="92" y="42" font-family="Arial" font-size="38" fill="#000000">A</text>',
    ]
    for tick in np.linspace(0, 1, 6):
        x = x_pos(float(tick))
        y = y_pos(float(tick))
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#eef0f3"/>')
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e2e5e9"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="13">{tick:.1f}</text>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="13">{tick:.1f}</text>')
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#b8bec6"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top}" stroke="#a7adb5" stroke-width="1.5" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="15">False Positive Rate</text>')
    parts.append(f'<text x="22" y="{top+plot_h/2:.1f}" text-anchor="middle" font-family="Arial" font-size="15" transform="rotate(-90 22 {top+plot_h/2:.1f})">True Positive Rate</text>')
    parts.append(f'<text x="{left+plot_w-4}" y="{top+16}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">macro-average ROC across queries</text>')

    for idx, method in enumerate(summary_df["Method"].tolist()):
        color = colors[idx % len(colors)]
        group = roc_df[roc_df["Method"] == method]
        points = " ".join(
            f'{x_pos(float(row.FPR)):.2f},{y_pos(float(row.TPR)):.2f}'
            for row in group.itertuples(index=False)
        )
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{points}"/>')

    legend_x = left + plot_w + 22
    legend_y = top + 18
    auc_map = summary_df.set_index("Method")["Macro_ROC_AUC"].to_dict()
    for idx, method in enumerate(summary_df["Method"].tolist()):
        color = colors[idx % len(colors)]
        y = legend_y + idx * 34
        label = f'{short_method_name(method)} (AUC={auc_map[method]:.3f})'
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+36}" y="{y+5}" font-family="Arial" font-size="12">{label}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def cluster_sea_raw_score(query, cluster, known_morgan_fps):
    member_fps = [
        known_morgan_fps[i]
        for i in cluster.get("ligand_indices", [])
        if i < len(known_morgan_fps)
    ]
    if not member_fps:
        return 0.0

    sims = DataStructs.BulkTanimotoSimilarity(query["morgan_fp"], member_fps)
    strong = sorted((s for s in sims if s >= SEA_TC_THRESHOLD), reverse=True)
    if not strong:
        return 0.0

    strong = strong[:SEA_TOPK_PER_CLUSTER]
    raw_sum = float(np.sum(strong))
    support_count = len(strong)
    size_norm = max(len(member_fps), 1) ** SEA_SIZE_NORMALIZATION_ALPHA
    support_bonus = np.log1p(support_count) / np.log1p(SEA_TOPK_PER_CLUSTER)
    return (raw_sum / size_norm) * (0.75 + 0.25 * support_bonus)


def best_cluster_sea_score(query, cluster_cache, known_morgan_fps):
    best = 0.0
    for cluster in cluster_cache.get("clusters", []):
        score = cluster_sea_raw_score(query, cluster, known_morgan_fps)
        if score > best:
            best = score
    return best


def adaptive_score_against_values(raw_score, background_values):
    if raw_score is None or pd.isna(raw_score) or not background_values:
        return np.nan
    arr = np.array(background_values, dtype=float)
    if len(arr) == 0 or not np.all(np.isfinite(arr)):
        arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    percentile = float(np.sum(arr <= raw_score) / len(arr))
    std = max(float(np.std(arr)), 1e-6)
    z = (float(raw_score) - float(np.mean(arr))) / std
    if z >= 0:
        sig = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        sig = ez / (1.0 + ez)
    return 0.5 * percentile + 0.5 * float(sig)


def cluster_sea_background(query, cluster_cache, known_morgan_fps):
    values = [
        cluster_sea_raw_score(query, cluster, known_morgan_fps)
        for cluster in cluster_cache.get("clusters", [])
    ]
    return [v for v in values if np.isfinite(v)]


def build_contribution_summary_extended(summary):
    rows = []
    metric_cols = [
        "Top1_Recall",
        "Top3_Recall",
        "Top5_Recall",
        "Top10_Recall",
        "Top20_Recall",
        "Top30_Recall",
        "Top50_Recall",
        "Top100_Recall",
        "Top200_Recall",
        "MRR",
        "AUC",
        "AUPRC",
        "BEDROC20",
        "RIE20",
        "EF1pct",
        "EF5pct",
        "EF10pct",
    ]
    comparisons = [
        ("Cluster_Profile", "Baseline_2D_MaxTanimoto", "Cluster profile contribution"),
        ("Cluster_SEA_Raw", "Baseline_2D_MaxTanimoto", "SEA-like cluster ensemble contribution"),
        ("Cluster_SEA_Adaptive", "Cluster_SEA_Raw", "Adaptive correction on SEA-like cluster"),
        ("USRCAT_Shape_Adaptive", "USRCAT_Shape", "Adaptive correction on USRCAT shape"),
        ("Cluster_Profile_Plus_USRCAT_Raw", "Cluster_Profile", "USRCAT shape added to cluster profile"),
        ("Cluster_SEA_Plus_USRCAT_Raw", "Cluster_SEA_Raw", "USRCAT shape added to SEA-like cluster"),
        ("Cluster_Profile_Plus_USRCAT_FixedFusion_W95", "Cluster_Profile", "USRCAT fixed 95/5 fusion added to cluster profile"),
        ("Cluster_Profile_Plus_USRCAT_FixedFusion_W98", "Cluster_Profile", "USRCAT fixed 98/2 fusion added to cluster profile"),
        ("Cluster_Profile_Plus_USRCAT_RankFusion_W95", "Cluster_Profile", "USRCAT rank 95/5 fusion added to cluster profile"),
        ("Cluster_SEA_Plus_USRCAT_FixedFusion_W90", "Cluster_SEA_Raw", "Fixed 90/10 USRCAT fusion added to SEA-like cluster"),
        ("Cluster_SEA_Plus_USRCAT_RankFusion_W90", "Cluster_SEA_Raw", "Rank 90/10 USRCAT fusion added to SEA-like cluster"),
        ("Cluster_SEA_Plus_USRCAT_RankFusion_W90", "Cluster_SEA_Plus_USRCAT_Raw", "Rank fusion replacing raw score fusion"),
        ("Cluster_SEA_Plus_USRCAT_Raw", "Cluster_Profile_Plus_USRCAT_Raw", "SEA-like evidence replacing profile in fusion"),
        ("Final_Full_Adaptive", "Cluster_Profile_Plus_USRCAT_Raw", "Adaptive correction on original fused model"),
    ]

    for dataset, group in summary.groupby("Dataset"):
        by_method = group.set_index("Method")
        for method, reference, label in comparisons:
            if method not in by_method.index or reference not in by_method.index:
                continue
            row = {
                "Dataset": dataset,
                "Comparison": label,
                "Method": method,
                "Reference": reference,
            }
            for metric in metric_cols:
                if metric in by_method.columns:
                    row[f"Delta_{metric}"] = (
                        by_method.loc[method, metric]
                        - by_method.loc[reference, metric]
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def trapezoid_auc_from_binary_ranking(labels):
    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return np.nan

    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = np.concatenate([[0.0], tp / positives])
    fpr = np.concatenate([[0.0], fp / negatives])
    return float(np.trapz(tpr, fpr))


def average_precision_from_binary_ranking(labels):
    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    if positives == 0:
        return np.nan
    precision_at_i = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision_at_i * labels) / positives)


def rie_bedroc_from_binary_ranking(labels, alpha=20.0):
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    positives = int(labels.sum())
    if positives == 0 or positives == n:
        return np.nan, np.nan

    ranks = np.where(labels == 1)[0] + 1
    observed = float(np.sum(np.exp(-alpha * ranks / n)))
    random_expected = positives * (1.0 - np.exp(-alpha)) / alpha
    rie = observed / random_expected if random_expected > 0 else np.nan

    best_ranks = np.arange(1, positives + 1)
    worst_ranks = np.arange(n - positives + 1, n + 1)
    best = float(np.sum(np.exp(-alpha * best_ranks / n)))
    worst = float(np.sum(np.exp(-alpha * worst_ranks / n)))
    bedroc = (observed - worst) / (best - worst) if best > worst else np.nan
    return float(rie), float(bedroc)


def evaluate_method_extended(method, score_matrix, target_ids, queries, top_detail_n):
    query_rows = []
    top_rows = []
    ranks = []
    reciprocal_ranks = []
    auc_values = []
    auprc_values = []
    rie_values = []
    bedroc_values = []
    normalized_best_ranks = []
    evaluable = 0
    library_target_set = set(target_ids)

    top_ks = [1, 3, 5, 10, 20, 30, 50, 100, 200]
    ef_fracs = [0.01, 0.05, 0.10]
    top_hits = {k: 0 for k in top_ks}
    top_hit_counts = {k: [] for k in top_ks}
    ef_sums = {frac: 0.0 for frac in ef_fracs}
    ef_counts = {frac: 0 for frac in ef_fracs}

    for q_idx, query in enumerate(queries):
        true_targets = set(query["true_targets"])
        true_in_library = true_targets & library_target_set
        if not true_in_library:
            query_rows.append({
                "Method": method,
                "Query_Index": query["source_index"],
                "SMILES": query["smiles"],
                "True_Targets": ";".join(sorted(true_targets)),
                "True_Targets_In_Library": "",
                "Best_True_Rank": np.nan,
                "Best_True_Target": "",
                "Best_True_Score": np.nan,
                "Evaluable": False,
            })
            continue

        scores = np.array(score_matrix[:, q_idx], dtype=float)
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        order = np.argsort(-scores, kind="mergesort")
        ranked_targets = [target_ids[i] for i in order]
        ranked_scores = [float(scores[i]) for i in order]
        labels = np.array([1 if t in true_in_library else 0 for t in ranked_targets], dtype=int)

        rank_map = {target: rank + 1 for rank, target in enumerate(ranked_targets)}
        score_map = {target: score for target, score in zip(ranked_targets, ranked_scores)}
        best_true = min(true_in_library, key=lambda t: rank_map.get(t, 10**12))
        best_rank = rank_map[best_true]
        best_score = score_map[best_true]

        evaluable += 1
        ranks.append(best_rank)
        reciprocal_ranks.append(1.0 / best_rank)
        normalized_best_ranks.append(best_rank / len(target_ids))
        auc_values.append(trapezoid_auc_from_binary_ranking(labels))
        auprc_values.append(average_precision_from_binary_ranking(labels))
        rie, bedroc = rie_bedroc_from_binary_ranking(labels, alpha=20.0)
        rie_values.append(rie)
        bedroc_values.append(bedroc)

        n_targets = len(target_ids)
        n_positives = len(true_in_library)
        for k in top_ks:
            kk = min(k, n_targets)
            hits = int(labels[:kk].sum())
            top_hit_counts[k].append(hits)
            if hits > 0:
                top_hits[k] += 1

        for frac in ef_fracs:
            cutoff = max(1, int(np.ceil(frac * n_targets)))
            hits = int(labels[:cutoff].sum())
            expected = cutoff * n_positives / n_targets
            if expected > 0:
                ef_sums[frac] += hits / expected
                ef_counts[frac] += 1

        query_rows.append({
            "Method": method,
            "Query_Index": query["source_index"],
            "SMILES": query["smiles"],
            "True_Targets": ";".join(sorted(true_targets)),
            "True_Targets_In_Library": ";".join(sorted(true_in_library)),
            "Best_True_Rank": best_rank,
            "Best_True_Target": best_true,
            "Best_True_Score": best_score,
            "Evaluable": True,
            "Per_Query_AUC": auc_values[-1],
            "Per_Query_AUPRC": auprc_values[-1],
            "Per_Query_BEDROC20": bedroc_values[-1],
            "Per_Query_RIE20": rie_values[-1],
            "Normalized_Best_Rank": normalized_best_ranks[-1],
        })

        for rank, (target, score, label) in enumerate(
            zip(
                ranked_targets[:top_detail_n],
                ranked_scores[:top_detail_n],
                labels[:top_detail_n],
            ),
            1,
        ):
            top_rows.append({
                "Method": method,
                "Query_Index": query["source_index"],
                "Rank": rank,
                "Predicted_Target": target,
                "Score": score,
                "Is_True_Target": bool(label),
                "SMILES": query["smiles"],
            })

    metrics = {
        "Method": method,
        "Queries_Total": len(queries),
        "Queries_Evaluable": evaluable,
        "MRR": float(np.nanmean(reciprocal_ranks)) if reciprocal_ranks else np.nan,
        "Mean_Best_Rank": float(np.nanmean(ranks)) if ranks else np.nan,
        "Median_Best_Rank": float(np.nanmedian(ranks)) if ranks else np.nan,
        "Mean_Normalized_Best_Rank": (
            float(np.nanmean(normalized_best_ranks)) if normalized_best_ranks else np.nan
        ),
        "AUC": float(np.nanmean(auc_values)) if auc_values else np.nan,
        "AUPRC": float(np.nanmean(auprc_values)) if auprc_values else np.nan,
        "BEDROC20": float(np.nanmean(bedroc_values)) if bedroc_values else np.nan,
        "RIE20": float(np.nanmean(rie_values)) if rie_values else np.nan,
    }

    for k in top_ks:
        metrics[f"Top{k}_Recall"] = top_hits[k] / evaluable if evaluable else np.nan
        metrics[f"Top{k}_HitRate"] = metrics[f"Top{k}_Recall"]
        metrics[f"Top{k}_AccuracyPct"] = (
            100.0 * metrics[f"Top{k}_Recall"] if evaluable else np.nan
        )
        metrics[f"Top{k}_MeanHitCount"] = (
            float(np.mean(top_hit_counts[k])) if top_hit_counts[k] else np.nan
        )

    for frac in ef_fracs:
        metrics[f"EF{int(frac * 100)}pct"] = (
            ef_sums[frac] / ef_counts[frac] if ef_counts[frac] else np.nan
        )

    return metrics, query_rows, top_rows


def score_target_internal(cache_file):
    target_id = cache_file.replace(".pkl", "")
    result = {
        "target_id": target_id,
        "scores": {method: [] for method in METHODS},
    }

    cluster_cache = ab.load_pickle(os.path.join(ab.G_CLUSTER_CACHE_DIR, cache_file))
    if cluster_cache is None:
        return result

    known_morgan_fps = [
        ab.bitvect_from_bytes(fp_bytes)
        for fp_bytes in cluster_cache.get("known_morgan_fps_bytes", [])
    ]

    shape_query_filter = None
    if G_SHAPE_QUERY_FILTER_BY_TARGET is not None:
        shape_query_filter = G_SHAPE_QUERY_FILTER_BY_TARGET.get(target_id, set())

    shape_cache = None
    if (
        ab.G_ENABLE_SHAPE
        and (shape_query_filter is None or len(shape_query_filter) > 0)
    ):
        shape_cache = ab.load_pickle(os.path.join(ab.G_SHAPE_CACHE_DIR, cache_file))

    adaptive_stats = ab.load_pickle(os.path.join(ab.G_ADAPTIVE_STATS_DIR, cache_file))
    cluster_stats = adaptive_stats.get("cluster_profile") if adaptive_stats else None
    shape_stats = adaptive_stats.get("shape_profile") if adaptive_stats else None

    for q_idx, query in enumerate(ab.G_QUERIES):
        baseline = ab.best_baseline_score(query, known_morgan_fps)
        cluster = ab.best_cluster_score(query, cluster_cache, known_morgan_fps)
        cluster_sea_values = cluster_sea_background(query, cluster_cache, known_morgan_fps)
        cluster_sea = max(cluster_sea_values) if cluster_sea_values else 0.0
        cluster_sea_adaptive = adaptive_score_against_values(cluster_sea, cluster_sea_values)
        should_score_shape = (
            shape_cache is not None
            and (shape_query_filter is None or q_idx in shape_query_filter)
        )
        shape = ab.best_shape_score(query, shape_cache) if should_score_shape else np.nan

        cluster_adaptive = ab.adaptive_calibrated_score(cluster, cluster_stats)
        shape_adaptive = ab.adaptive_calibrated_score(shape, shape_stats)
        raw_fusion = ab.fuse_raw(cluster, shape)
        full_adaptive = ab.fuse_raw(cluster_adaptive, shape_adaptive)

        result["scores"]["Baseline_2D_MaxTanimoto"].append(baseline)
        result["scores"]["Cluster_Profile"].append(cluster)
        result["scores"]["Cluster_Profile_Adaptive"].append(cluster_adaptive)
        result["scores"]["Cluster_SEA_Raw"].append(cluster_sea)
        result["scores"]["Cluster_SEA_Adaptive"].append(cluster_sea_adaptive)
        result["scores"]["USRCAT_Shape"].append(shape)
        result["scores"]["USRCAT_Shape_Adaptive"].append(shape_adaptive)
        result["scores"]["Cluster_Profile_Plus_USRCAT_Raw"].append(raw_fusion)
        result["scores"]["Cluster_SEA_Plus_USRCAT_Raw"].append(
            ab.fuse_raw(cluster_sea, shape)
        )
        result["scores"]["Final_Full_Adaptive"].append(full_adaptive)

    return result


def feature_one_query(task):
    idx, row_dict, targets, num_query_confs, enable_shape = task
    mol = ab.standardize_smiles(row_dict["SMILES"])
    if mol is None:
        return None

    query = {
        "query_index": idx,
        "source_index": row_dict["Query_ID"],
        "smiles": str(row_dict["SMILES"]),
        "canonical_smiles": row_dict.get(
            "Canonical_SMILES", ab.Chem.MolToSmiles(mol, canonical=True)
        ),
        "true_targets": targets,
        "morgan_fp": ab.get_morgan_fp(mol),
        "scaffold": row_dict.get("Scaffold", ab.get_scaffold_smiles(mol)),
        "properties": ab.calc_properties(mol),
        "shape_vectors": [],
    }
    if enable_shape:
        query["shape_vectors"] = ab.generate_query_shape_vectors(mol, num_query_confs)
    return query


def query_cache_path(benchmark_root, num_query_confs, enable_shape):
    suffix = "usrcat_shape" if enable_shape else "cluster_only"
    return os.path.join(
        benchmark_root,
        f"test_query_features_{suffix}_confs{num_query_confs}.pkl",
    )


def load_external_queries(
    query_file,
    truth_file,
    benchmark_root,
    num_query_confs,
    enable_shape,
    workers,
    progress_every,
    rebuild_cache,
):
    cache_path = query_cache_path(benchmark_root, num_query_confs, enable_shape)
    if os.path.exists(cache_path) and not rebuild_cache:
        started = time.time()
        log_step(f"Loading cached query features: {cache_path}")
        with open(cache_path, "rb") as f:
            queries = pickle.load(f)
        log_step(
            f"Loaded cached query features: {len(queries)} "
            f"in {time.time() - started:.1f}s"
        )
        return queries

    started_read = time.time()
    log_step(f"Reading query file: {query_file}")
    query_df = pd.read_csv(query_file, encoding="utf_8_sig")
    log_step(f"Reading truth file: {truth_file}")
    truth_df = pd.read_csv(truth_file, encoding="utf_8_sig")
    log_step(
        f"Loaded query/truth tables: query_rows={len(query_df)}, "
        f"truth_rows={len(truth_df)} in {time.time() - started_read:.1f}s"
    )

    truth_map = (
        truth_df.groupby("Query_ID")["UniProt"]
        .apply(lambda s: sorted(set(str(x).strip() for x in s if pd.notna(x))))
        .to_dict()
    )

    tasks = []
    for idx, row in query_df.iterrows():
        query_id = row["Query_ID"]
        targets = truth_map.get(query_id, [])
        if not targets:
            continue
        tasks.append((len(tasks), row.to_dict(), targets, num_query_confs, enable_shape))

    log_step(
        f"Featurizing {len(tasks)} queries "
        f"(enable_shape={enable_shape}, query_confs={num_query_confs}, workers={workers})..."
    )
    queries = []
    started = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=workers,
        initializer=ab.init_worker,
        maxtasksperchild=20,
    ) as pool:
        for i, query in enumerate(pool.imap_unordered(feature_one_query, tasks, chunksize=1), 1):
            if query is not None:
                queries.append(query)
            if i % progress_every == 0 or i == len(tasks):
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (len(tasks) - i) / rate if rate > 0 else 0.0
                log_step(
                    f"Query features: {i}/{len(tasks)} processed, "
                    f"valid={len(queries)} rate={rate:.2f} queries/s eta={eta/60:.1f} min"
                )

    log_step("Sorting and indexing query features...")
    queries.sort(key=lambda q: q["source_index"])
    for idx, query in enumerate(queries):
        query["query_index"] = idx

    log_step(f"Writing query feature cache: {cache_path}")
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(queries, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    log_step(f"Saved query feature cache: {cache_path}")
    return queries


def run_external(args):
    global G_SHAPE_QUERY_FILTER_BY_TARGET

    fusion_weights = parse_weight_list(args.fusion_cluster_weights)
    ab.G_CLUSTER_CACHE_DIR = args.cluster_cache_dir
    ab.G_SHAPE_CACHE_DIR = args.shape_cache_dir
    ab.G_ADAPTIVE_STATS_DIR = args.adaptive_stats_dir
    ab.G_ENABLE_SHAPE = not args.cluster_only

    query_file = os.path.join(args.benchmark_root, "test_queries.csv")
    truth_file = os.path.join(args.benchmark_root, "test_ground_truth.csv")

    total_started = time.time()
    log_step("Loading external benchmark queries...")
    queries = load_external_queries(
        query_file,
        truth_file,
        benchmark_root=args.benchmark_root,
        num_query_confs=args.query_confs,
        enable_shape=not args.cluster_only,
        workers=args.query_workers,
        progress_every=args.query_progress_every,
        rebuild_cache=args.rebuild_query_cache,
    )
    log_step(f"Valid internal queries: {len(queries)}")
    if not queries:
        raise ValueError("No valid internal benchmark queries were loaded.")

    log_step("Collecting target cache files...")
    target_files = ab.collect_target_files(args.cluster_cache_dir, args.limit_targets)
    target_ids = [f.replace(".pkl", "") for f in target_files]
    target_index = {target_id: idx for idx, target_id in enumerate(target_ids)}
    log_step(f"Candidate targets: {len(target_ids)}")

    ab.G_QUERIES = queries
    qsig = query_signature(queries)
    checkpoint_root = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoints")
    log_step(f"Query signature: {qsig}")
    log_step(f"Score checkpoint root: {checkpoint_root}")
    log_step(
        f"Allocating score matrices: methods={len(METHODS)}, "
        f"targets={len(target_files)}, queries={len(queries)}"
    )
    score_mats = {
        method: np.full((len(target_files), len(queries)), np.nan, dtype=np.float32)
        for method in METHODS
    }
    log_step("Score matrices allocated.")

    def fill_score_matrices(result):
        row_idx = target_index.get(result["target_id"])
        if row_idx is None:
            return False
        for method in METHODS:
            score_mats[method][row_idx, :] = np.array(
                result["scores"][method], dtype=np.float32
            )
        return True

    def run_scoring_pass(pass_name, files_to_score, checkpoint_name):
        checkpoint_dir = os.path.join(checkpoint_root, qsig, checkpoint_name)
        log_step(f"{pass_name}: checkpoint dir: {checkpoint_dir}")

        pending_files = []
        loaded = 0
        invalid = 0
        if not args.rebuild_score_checkpoints:
            load_started = time.time()
            for i, cache_file in enumerate(files_to_score, 1):
                path = score_checkpoint_path(checkpoint_dir, cache_file)
                if os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            result = pickle.load(f)
                        if valid_checkpoint_result(result, len(queries)):
                            fill_score_matrices(result)
                            loaded += 1
                        else:
                            pending_files.append(cache_file)
                            invalid += 1
                    except Exception:
                        pending_files.append(cache_file)
                        invalid += 1
                else:
                    pending_files.append(cache_file)

                if i % args.progress_every == 0 or i == len(files_to_score):
                    log_step(
                        f"{pass_name}: checkpoint scan {i}/{len(files_to_score)} "
                        f"loaded={loaded} pending={len(pending_files)} invalid={invalid}"
                    )
            log_step(
                f"{pass_name}: checkpoint scan finished in "
                f"{time.time() - load_started:.1f}s"
            )
        else:
            pending_files = list(files_to_score)
            log_step(f"{pass_name}: rebuilding score checkpoints by request.")

        if not pending_files:
            log_step(f"{pass_name}: all target scores loaded from checkpoints.")
            return

        log_step(
            f"{pass_name}: starting scoring pass for {len(pending_files)} pending targets "
            f"with workers={args.workers}"
        )
        started = time.time()
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=args.workers,
            initializer=ab.init_worker,
            maxtasksperchild=args.maxtasksperchild,
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(score_target_internal, pending_files, chunksize=1), 1
            ):
                fill_score_matrices(result)
                cache_file = f"{result['target_id']}.pkl"
                atomic_pickle_dump(
                    {
                        "checkpoint_version": "score_target_v1",
                        "pass_name": pass_name,
                        "query_signature": qsig,
                        "target_id": result["target_id"],
                        "scores": result["scores"],
                    },
                    score_checkpoint_path(checkpoint_dir, cache_file),
                )
                if i % args.progress_every == 0 or i == len(pending_files):
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0.0
                    eta = (len(pending_files) - i) / rate if rate > 0 else 0.0
                    log_step(
                        f"{pass_name}: scored {i}/{len(pending_files)} pending targets "
                        f"rate={rate:.2f} targets/s eta={eta/60:.1f} min"
                    )
        log_step(f"{pass_name}: finished in {(time.time() - started)/60:.2f} min")

    if args.shape_prefilter_topk > 0 and not args.cluster_only:
        log_step(
            f"Stage 1/2: fast cluster scoring for shape prefilter "
            f"(topK={args.shape_prefilter_topk})"
        )
        ab.G_ENABLE_SHAPE = False
        G_SHAPE_QUERY_FILTER_BY_TARGET = {}
        run_scoring_pass(
            "External prefilter",
            target_files,
            checkpoint_name=f"prefilter_{args.shape_prefilter_method}",
        )

        log_step("Building target-query shape filter...")
        filter_started = time.time()
        shape_filter = {}
        prefilter_scores = score_mats[args.shape_prefilter_method]
        for q_idx in range(len(queries)):
            scores = prefilter_scores[:, q_idx]
            finite = np.where(np.isfinite(scores))[0]
            if len(finite) == 0:
                continue
            topk = min(args.shape_prefilter_topk, len(finite))
            top_local = finite[np.argpartition(-scores[finite], topk - 1)[:topk]]
            for target_row_idx in top_local:
                target_id = target_ids[target_row_idx]
                shape_filter.setdefault(target_id, set()).add(q_idx)
            if (
                (q_idx + 1) % args.filter_progress_every == 0
                or (q_idx + 1) == len(queries)
            ):
                elapsed = time.time() - filter_started
                rate = (q_idx + 1) / elapsed if elapsed > 0 else 0.0
                eta = (len(queries) - q_idx - 1) / rate if rate > 0 else 0.0
                log_step(
                    f"Shape filter: {q_idx + 1}/{len(queries)} queries, "
                    f"targets_with_shape={len(shape_filter)} "
                    f"rate={rate:.2f} queries/s eta={eta/60:.1f} min"
                )

        filtered_target_files = [f"{target_id}.pkl" for target_id in shape_filter]
        total_pairs = sum(len(qs) for qs in shape_filter.values())
        log_step(
            f"Stage 2/2: 3D shape scoring for {len(filtered_target_files)} "
            f"targets with {total_pairs} target-query pairs"
        )
        ab.G_ENABLE_SHAPE = True
        G_SHAPE_QUERY_FILTER_BY_TARGET = shape_filter
        run_scoring_pass(
            "External filtered shape",
            filtered_target_files,
            checkpoint_name=(
                f"filtered_shape_top{args.shape_prefilter_topk}_"
                f"{args.shape_prefilter_method}"
            ),
        )
    else:
        G_SHAPE_QUERY_FILTER_BY_TARGET = None
        checkpoint_name = "cluster_only" if args.cluster_only else "full_all_targets"
        run_scoring_pass("External", target_files, checkpoint_name=checkpoint_name)

    log_step("Building derived fixed-weight and rank-fusion score matrices...")
    derived_methods = [] if args.cluster_only else add_fusion_score_matrices(
        score_mats,
        fusion_weights,
    )
    if derived_methods:
        log_step("Derived fusion methods: " + ", ".join(derived_methods))
    else:
        log_step("No derived fusion methods were created.")

    log_step("Evaluating metrics and query ranks...")
    eval_started = time.time()
    metrics_rows = []
    query_rows = []
    top_rows = []
    evaluation_methods = list(METHODS) + derived_methods
    if not args.include_adaptive_ranking_methods:
        evaluation_methods = [
            method
            for method in evaluation_methods
            if method not in ADAPTIVE_RANKING_METHODS
        ]
        log_step(
            "Adaptive ranking methods are excluded from final ranking evaluation; "
            "they will be used for significance explanation only."
        )

    for method in evaluation_methods:
        if args.cluster_only and ("Shape" in method or "USRCAT" in method):
            continue
        method_started = time.time()
        log_step(f"Evaluating method: {method}")
        metrics, method_query_rows, method_top_rows = evaluate_method_extended(
            method,
            score_mats[method],
            target_ids,
            queries,
            args.top_detail_n,
        )
        metrics_rows.append(metrics)
        query_rows.extend(method_query_rows)
        top_rows.extend(method_top_rows)
        log_step(f"Finished {method} evaluation in {time.time() - method_started:.1f}s")
    log_step(f"All evaluations finished in {(time.time() - eval_started)/60:.2f} min")

    log_step(f"Preparing output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_rows).assign(Dataset=DATASET_NAME)
    cols = ["Dataset"] + [c for c in metrics_df.columns if c != "Dataset"]
    metrics_df = metrics_df[cols]
    query_df = pd.DataFrame(query_rows)
    top_df = pd.DataFrame(top_rows)
    contribution_df = build_contribution_summary_extended(metrics_df)
    significance_method = choose_significance_method(
        score_mats,
        args.significance_method,
        derived_methods,
    )
    significance_df = build_adaptive_significance_table(
        score_mats,
        target_ids,
        queries,
        significance_method,
        args.significance_top_n,
    )
    roc_methods = parse_method_list(args.roc_methods, score_mats)

    metrics_path = os.path.join(
        args.output_dir, "external_bindingdb_chembl202604_ablation_metrics.csv"
    )
    query_path = os.path.join(
        args.output_dir, "external_bindingdb_chembl202604_query_ranks.csv"
    )
    top_path = os.path.join(
        args.output_dir,
        f"external_bindingdb_chembl202604_top{args.top_detail_n}.csv",
    )
    contribution_path = os.path.join(
        args.output_dir,
        "external_bindingdb_chembl202604_module_contribution_summary.csv",
    )
    significance_path = os.path.join(
        args.output_dir,
        "external_bindingdb_chembl202604_adaptive_significance_explanations.csv",
    )
    roc_curve_path = None
    roc_svg_path = None
    if args.write_roc_curve:
        roc_curve_path, roc_svg_path = write_macro_roc_outputs(
            score_mats,
            target_ids,
            queries,
            args.output_dir,
            roc_methods,
            args.roc_grid_points,
        )

    log_step(f"Writing metrics CSV: {metrics_path}")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf_8_sig")
    log_step(f"Writing query ranks CSV: {query_path} rows={len(query_df)}")
    query_df.to_csv(query_path, index=False, encoding="utf_8_sig")
    if args.top_detail_n > 0:
        log_step(f"Writing top prediction CSV: {top_path} rows={len(top_df)}")
        top_df.to_csv(top_path, index=False, encoding="utf_8_sig")
    log_step(f"Writing contribution summary CSV: {contribution_path}")
    contribution_df.to_csv(contribution_path, index=False, encoding="utf_8_sig")
    if not significance_df.empty:
        log_step(
            f"Writing adaptive significance explanation CSV: {significance_path} "
            f"rows={len(significance_df)}"
        )
        significance_df.to_csv(significance_path, index=False, encoding="utf_8_sig")
    else:
        log_step("Skipped adaptive significance explanation output.")

    log_step(f"Saved metrics: {metrics_path}")
    log_step(f"Saved query ranks: {query_path}")
    if args.top_detail_n > 0:
        log_step(f"Saved top predictions: {top_path}")
    else:
        log_step("Skipped top prediction detail output (--top-detail-n 0).")
    log_step(f"Saved contribution summary: {contribution_path}")
    if not significance_df.empty:
        log_step(f"Saved adaptive significance explanations: {significance_path}")
    if roc_curve_path:
        log_step(f"Saved macro ROC curve CSV: {roc_curve_path}")
    if roc_svg_path:
        log_step(f"Saved macro ROC curve SVG: {roc_svg_path}")
    log_step(f"Total run finished in {(time.time() - total_started)/60:.2f} min")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PharmTAP target-prediction benchmark using SRCP + USRCAT evidence."
    )
    parser.add_argument("--benchmark-root", default=BENCHMARK_ROOT)
    parser.add_argument("--cluster-cache-dir", default=CLUSTER_CACHE_DIR)
    parser.add_argument("--shape-cache-dir", default=SHAPE_CACHE_DIR)
    parser.add_argument("--adaptive-stats-dir", default=ADAPTIVE_STATS_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=max(ab.available_cpus() - 1, 1))
    parser.add_argument(
        "--query-workers",
        type=int,
        default=max(min(ab.available_cpus() - 1, 16), 1),
        help="Workers for query 3D conformer generation before target scoring.",
    )
    parser.add_argument("--query-confs", type=int, default=10)
    parser.add_argument("--query-progress-every", type=int, default=25)
    parser.add_argument("--rebuild-query-cache", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        help=(
            "Directory for per-target score checkpoints. Defaults to "
            "<output-dir>/checkpoints."
        ),
    )
    parser.add_argument(
        "--rebuild-score-checkpoints",
        action="store_true",
        help="Ignore existing per-target score checkpoints and recompute them.",
    )
    parser.add_argument("--top-detail-n", type=int, default=100)
    parser.add_argument(
        "--fusion-cluster-weights",
        default="0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99",
        help=(
            "Comma-separated primary weights for Cluster_Profile/Cluster_SEA_Raw "
            "in derived fixed-weight and rank-fusion methods. USRCAT Shape "
            "receives 1-weight."
        ),
    )
    parser.add_argument(
        "--include-adaptive-ranking-methods",
        action="store_true",
        help=(
            "Include adaptive-calibrated methods in final ranking metrics. "
            "By default adaptive scores are used only for significance explanation."
        ),
    )
    parser.add_argument(
        "--significance-method",
        default="Cluster_Profile_Plus_USRCAT_FixedFusion_W95",
        help=(
            "Ranking method whose top predictions receive adaptive significance "
            "annotations. Falls back to an available proposed method if missing."
        ),
    )
    parser.add_argument(
        "--significance-top-n",
        type=int,
        default=20,
        help="Number of top predictions per query to annotate with adaptive scores.",
    )
    parser.add_argument(
        "--write-roc-curve",
        action="store_true",
        help=(
            "Write macro-average ROC curve CSV/SVG from full in-memory score "
            "matrices. Existing score checkpoints can be reused."
        ),
    )
    parser.add_argument(
        "--roc-methods",
        default="Cluster_Profile,Cluster_SEA_Raw,USRCAT_Shape,Cluster_Profile_Plus_USRCAT_Raw,Cluster_SEA_Plus_USRCAT_Raw",
        help=(
            "Comma-separated methods for ROC output. Defaults to the main "
            "external comparison methods without 2D MaxTanimoto."
        ),
    )
    parser.add_argument(
        "--roc-grid-points",
        type=int,
        default=501,
        help="Number of FPR grid points for macro-average ROC interpolation.",
    )
    parser.add_argument(
        "--shape-prefilter-topk",
        type=int,
        default=0,
        help=(
            "If >0, run 3D shape only for each query's top-K targets from a "
            "fast prefilter score. This greatly accelerates large ablations."
        ),
    )
    parser.add_argument(
        "--shape-prefilter-method",
        choices=[
            "Baseline_2D_MaxTanimoto",
            "Cluster_Profile",
            "Cluster_Profile_Adaptive",
            "Cluster_SEA_Raw",
            "Cluster_SEA_Adaptive",
        ],
        default="Cluster_Profile",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--filter-progress-every", type=int, default=1000)
    parser.add_argument("--maxtasksperchild", type=int, default=20)
    parser.add_argument("--limit-targets", type=int)
    parser.add_argument("--cluster-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    log_step(f"Benchmark root: {args.benchmark_root}")
    log_step(f"Cluster cache dir: {args.cluster_cache_dir}")
    log_step(f"Shape cache dir: {args.shape_cache_dir}")
    log_step(f"Adaptive stats dir: {args.adaptive_stats_dir}")
    log_step(f"Output dir: {args.output_dir}")
    log_step(f"Workers: {args.workers}")
    log_step(f"Query workers: {args.query_workers}")
    log_step(f"Query conformers: {args.query_confs}")
    log_step(f"Cluster only: {args.cluster_only}")
    log_step(f"Shape prefilter topK: {args.shape_prefilter_topk}")
    log_step(f"Shape prefilter method: {args.shape_prefilter_method}")
    log_step(f"Top detail N: {args.top_detail_n}")
    log_step(f"Fusion cluster weights: {args.fusion_cluster_weights}")
    log_step(
        f"Include adaptive ranking methods: {args.include_adaptive_ranking_methods}"
    )
    log_step(f"Significance method: {args.significance_method}")
    log_step(f"Significance top N: {args.significance_top_n}")
    log_step(f"Checkpoint dir: {args.checkpoint_dir or os.path.join(args.output_dir, 'checkpoints')}")
    log_step(f"Rebuild score checkpoints: {args.rebuild_score_checkpoints}")
    run_external(args)


if __name__ == "__main__":
    main()
