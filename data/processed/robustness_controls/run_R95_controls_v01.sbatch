#!/bin/bash
#SBATCH -J r95ctrl
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/r95ctrl_%j.out
#SBATCH -e logs/r95ctrl_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source ~/.bashrc || true
mamba activate dsmc-gpu || conda activate dsmc-gpu || true
mkdir -p logs

python3 22_protrusion_R95_robustness_controls.py \
  --audit-dir ./protrusion_01_audit_v2 \
  --out ./protrusion_22_R95_controls_v01 \
  --nonlocal-script ./16_protrusion_nonlocal_bulk_to_wall_footprint.py \
  --field-base-script ./10_protrusion_train_smooth_field_operator_v4_geomfix.py \
  --surface-base-script ./06_protrusion_train_unified_operator_v8_geomfix.py \
  --radii 0.05,0.1,0.2,0.35,0.5,0.75,1,1.5,2,3 \
  --models extratrees,ridge \
  --thresholds 0.025,0.05,0.10 \
  --max-gas-points 60000 \
  --trees 250 \
  --jobs "${SLURM_CPUS_PER_TASK:-16}" \
  --make-plots

echo "DONE R95 controls"
