# PharmTAP

PharmTAP is a cheminformatics workflow for chemotype-resolved target prediction and multimodal multitask ADMET prediction. This public repository contains the reproducibility scripts for the core methods described in the PharmTAP manuscript:

- SRCP + USRCAT ligand-based target prediction.
- Adaptive support background generation for target-specific support annotation.
- Multimodal ADMET scaffold splitting, single-task baselines, and graph-descriptor-ECFP multitask fusion ablation.

The academic web server is available at https://www.pharm-tap.com/.

## Repository layout

```text
scripts/
  target_prediction/
    build_srcp_target_profiles.py
    build_usrcat_target_profiles.py
    build_adaptive_support_backgrounds.py
    predict_targets.py
    run_target_prediction_benchmark.py
  admet/
    build_admet_scaffold_splits.py
    admet_graph_desc_baseline.py
    admet_multimodal_core.py
    train_admet_single_task_baseline.py
    train_admet_fusion_ablation.py
data/
  README.md
```

## Data

The curated target-prediction and ADMET datasets are distributed through the PharmTAP website rather than GitHub. Download the data package from https://www.pharm-tap.com/ and place the extracted files under `data/`.

Recommended local layout:

```text
data/
  target_ligands/
  external_benchmark/
  admet/
    data/
    data_scaffold_8_1_1_seed42/
    data_scaffold_8_1_1_seed43/
    data_scaffold_8_1_1_seed44/
```

Large generated files, model checkpoints, caches, and outputs should be written to `outputs/`, which is ignored by git.

## Environment

A conda environment file is provided:

```bash
conda env create -f environment.yml
conda activate pharmtap
```

Alternatively, install the Python dependencies from `requirements.txt`. RDKit, PyTorch, and PyTorch Geometric are often easiest to install through conda because their binary builds depend on the operating system and CUDA version.

## Target prediction workflow

Build scaffold-resolved chemotype profiles:

```bash
python scripts/target_prediction/build_srcp_target_profiles.py \
  --data-root data/target_ligands \
  --cache-dir outputs/target_prediction/srcp_target_profiles
```

Build multi-conformer USRCAT shape-pharmacophore profiles:

```bash
python scripts/target_prediction/build_usrcat_target_profiles.py \
  --cluster-cache-dir outputs/target_prediction/srcp_target_profiles \
  --shape-cache-dir outputs/target_prediction/usrcat_target_profiles
```

Build adaptive support background statistics:

```bash
python scripts/target_prediction/build_adaptive_support_backgrounds.py \
  --cluster-cache-dir outputs/target_prediction/srcp_target_profiles \
  --shape-cache-dir outputs/target_prediction/usrcat_target_profiles \
  --stats-dir outputs/target_prediction/adaptive_support_backgrounds
```

Run target prediction for a SMILES input:

```bash
python scripts/target_prediction/predict_targets.py \
  --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1" \
  --cluster-cache-dir outputs/target_prediction/srcp_target_profiles \
  --shape-cache-dir outputs/target_prediction/usrcat_target_profiles \
  --adaptive-stats-dir outputs/target_prediction/adaptive_support_backgrounds \
  --output-csv outputs/example_target_predictions.csv \
  --top-n 100
```

Run the external target-prediction benchmark:

```bash
python scripts/target_prediction/run_target_prediction_benchmark.py \
  --benchmark-root data/external_benchmark \
  --cluster-cache-dir outputs/target_prediction/srcp_target_profiles \
  --shape-cache-dir outputs/target_prediction/usrcat_target_profiles \
  --adaptive-stats-dir outputs/target_prediction/adaptive_support_backgrounds \
  --output-dir outputs/target_prediction_benchmark
```

## ADMET workflow

Create scaffold splits for ADMET endpoints:

```bash
python scripts/admet/build_admet_scaffold_splits.py \
  --data-root data/admet/data \
  --output-root data/admet/data_scaffold_8_1_1_seed42 \
  --seed 42
```

Train the independent single-task graph-descriptor-ECFP baseline:

```bash
python scripts/admet/train_admet_single_task_baseline.py \
  --data-root data/admet/data_scaffold_8_1_1_seed42 \
  --model-dir outputs/admet/single_task_models \
  --result-dir outputs/admet/single_task_results \
  --seeds 42
```

Train the multimodal multitask fusion ablation:

```bash
python scripts/admet/train_admet_fusion_ablation.py \
  --data-root data/admet/data_scaffold_8_1_1_seed42 \
  --model-dir outputs/admet/fusion_models \
  --result-dir outputs/admet/fusion_results \
  --fusion-modes mt_graph_desc_ecfp_attention,mt_graph_desc_ecfp_gated,mt_graph_desc_ecfp_gated_attention \
  --seeds 42
```

Use `--help` on each script to inspect additional options.

## Citation

If you use this code or the associated data, please cite the PharmTAP manuscript.

