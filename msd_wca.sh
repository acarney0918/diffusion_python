#!/bin/bash
set -euo pipefail

BASE_DIR="/scratch/hsg2ke/wca_fse/input"
MSD_DIR="${BASE_DIR}/msd"
PYTHON_DIR="${BASE_DIR}/python"

# === MSD script (streaming, multi-origin) ===
MSD_SCRIPT="${PYTHON_DIR}/msd_streaming_multi_origin.py"

# Tunables for MSD
TIMESTEP_FS="0.25"          # fs; your MD timestep
MAX_TAU_FRAMES="100000"     # longest lag in frames
MAX_ACTIVE_ORIGINS="8"      # memory knob (4–16 typical)
DESIRED_ATOM_ID="1"         # single-atom trace (doesn't affect all-atoms MSD)

mkdir -p "$MSD_DIR"

# Check for required script
if [ ! -f "$MSD_SCRIPT" ]; then
    echo "Missing script: $MSD_SCRIPT"
    exit 1
fi

for rep in {3..10}; do
    REP_DIR="${BASE_DIR}/rep${rep}_fixmom"
    REP_MSD_DIR="${MSD_DIR}/rep${rep}_fixmom"
    mkdir -p "$REP_MSD_DIR"
    job_ids=()

    for trial in {0..9}; do
        TRIAL_DIR="${REP_DIR}/trial${trial}"
        if [ -d "$TRIAL_DIR" ]; then
            # Collect any existing unwrapped dumps (stage1..3)
            FILES=()
            for s in 1 2 3; do
                f="${TRIAL_DIR}/dump_unwrapped_stage${s}.lammpstrj"
                [[ -f "$f" ]] && FILES+=("$f")
            done
            # Fallback: single-file naming (optional)
            f_default="${TRIAL_DIR}/dump_unwrapped.lammpstrj"
            [[ ${#FILES[@]} -eq 0 && -f "$f_default" ]] && FILES+=("$f_default")

            if [[ ${#FILES[@]} -gt 0 ]]; then
                SLURM_SCRIPT="${TRIAL_DIR}/run_msd.slurm"

                # Build a Python list literal from FILES
                PY_LIST=$(printf "'%s', " "${FILES[@]}")
                PY_LIST="[${PY_LIST%, }]"

                cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH -J msd_r${rep}_fixmom_t${trial}
#SBATCH -o ${TRIAL_DIR}/msd.out
#SBATCH -e ${TRIAL_DIR}/msd.err
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 6-00:00:00
#SBATCH -A dubay-carney

module load miniforge
export PYTHONPATH="${PYTHON_DIR}:\$PYTHONPATH"
cd "$TRIAL_DIR"

python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "${PYTHON_DIR}")
from msd_streaming_multi_origin import msd_streaming, save_msd_to_txt

file_paths = ${PY_LIST}
timestep_fs = float("${TIMESTEP_FS}")
max_tau_frames = int("${MAX_TAU_FRAMES}")
max_active_origins = int("${MAX_ACTIVE_ORIGINS}")
desired_atom_id = int("${DESIRED_ATOM_ID}")

time_fs, msd, s_msd, dump_stride = msd_streaming(
    file_paths,
    timestep_fs=timestep_fs,
    max_tau_frames=max_tau_frames,
    max_active_origins=max_active_origins,
    desired_atom_id=desired_atom_id,
    assume_sorted_by_id=True,
)

out = Path("${REP_MSD_DIR}") / f"msd_rep${rep}_trial${trial}.txt"
save_msd_to_txt(time_fs, msd, s_msd, filename=str(out))
print(f"Inferred dump stride: {dump_stride} MD steps")
print(f"Saved: {out}")
PY
EOF

                jobid=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')
                echo "Submitted MSD job: $jobid for rep${rep}_fixmom/trial${trial}"
                job_ids+=("$jobid")
            else
                echo "No unwrapped dumps found in $TRIAL_DIR (expected dump_unwrapped_stage[1-3].lammpstrj), skipping."
            fi
        fi
    done

    # Optional post step (depends on all per-trial MSD jobs)
    if [ ${#job_ids[@]} -gt 0 ]; then
        DEPENDENCY=$(IFS=:; echo "${job_ids[*]}")
        SLURM_POST="${REP_DIR}/run_msd_post.slurm"

        cat > "$SLURM_POST" <<EOF
#!/bin/bash
#SBATCH -J msd_post_r${rep}_fixmom
#SBATCH -o ${REP_DIR}/msd_post.out
#SBATCH -e ${REP_DIR}/msd_post.err
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 1-00:00:00
#SBATCH -A dubay-carney
#SBATCH --dependency=afterok:${DEPENDENCY}

module load miniforge
cd "${REP_MSD_DIR}"

echo "MSD results for rep${rep}_fixmom:"
ls -1 msd_rep${rep}_trial*.txt || true

# (Optional) add a reducer here to average across trials or fit D from the linear tail.
EOF

        postid=$(sbatch "$SLURM_POST" | awk '{print $4}')
        echo "Submitted MSD post job: $postid for rep${rep}_fixmom"
    else
        echo "No MSD jobs for rep${rep}_fixmom, skipping post job."
    fi
done

