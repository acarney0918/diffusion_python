#!/bin/bash
set -euo pipefail

BASE_DIR="/scratch/hsg2ke/wca_fse/input"
V2_DIR="${BASE_DIR}/v2"
PYTHON_DIR="${BASE_DIR}/python"

# Python scripts
VACF_SCRIPT="${PYTHON_DIR}/vacf_wca_windows.py"   # updated script that writes _windowN.txt + _windowavg.txt
ANALYSIS_SCRIPT="${PYTHON_DIR}/vacf_analysis.py"
DIFFUSION_SCRIPT="${PYTHON_DIR}/vacf_to_diffusion.py"

mkdir -p "$V2_DIR"

# Sanity checks
for script in "$VACF_SCRIPT" "$ANALYSIS_SCRIPT" "$DIFFUSION_SCRIPT"; do
    if [ ! -f "$script" ]; then
        echo "Missing script: $script"
        exit 1
    fi
done

for rep in {3..10}; do
    REP_DIR="${BASE_DIR}/rep${rep}_fixmom"
    REP_V2_DIR="${V2_DIR}/rep${rep}_fixmom"
    mkdir -p "$REP_V2_DIR"
    job_ids=()

    for trial in {0..9}; do
        TRIAL_DIR="${REP_DIR}/trial${trial}"
        if [ -d "$TRIAL_DIR" ]; then
            D2="${TRIAL_DIR}/dump_velocity_stage1.lammpstrj"

            if [[ -f "$D2" ]]; then
                SLURM_SCRIPT="${TRIAL_DIR}/run_vacf.slurm"

                cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH -J vacf_r${rep}_fixmom_t${trial}
#SBATCH -o ${TRIAL_DIR}/vacf.out
#SBATCH -e ${TRIAL_DIR}/vacf.err
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 6-00:00:00
#SBATCH -A dubay-carney

module load miniforge
cd "$TRIAL_DIR"

# Let NumPy/FFTs use threads
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK

# The VACF script will emit:
#   vacf_rep${rep}_trial${trial}_window1.txt, _window2.txt, ..., and _windowavg.txt
python "$VACF_SCRIPT" \\
  --input "$D2" \\
  --output "${REP_V2_DIR}/vacf_rep${rep}_trial${trial}.txt" \\
  --dt 0.25 \\
  --windows 5 \\
  --overlap 0.5 \\
  --compare-vacf
  # add --windows/--overlap tweaks if needed
  # remove --compare-vacf if you don't want the PNG
EOF

                jobid=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')
                echo "Submitted VACF job: $jobid for rep${rep}_fixmom/trial${trial}"
                job_ids+=("$jobid")
            else
                echo "Missing velocity dump stage2 in $TRIAL_DIR, skipping."
            fi
        fi
    done

    if [ ${#job_ids[@]} -gt 0 ]; then
        DEPENDENCY=$(IFS=:; echo "${job_ids[*]}")
        SLURM_ANALYSIS="${REP_DIR}/run_vacf_analysis.slurm"

        cat > "$SLURM_ANALYSIS" <<EOF
#!/bin/bash
#SBATCH -J vacf_analysis_r${rep}_fixmom
#SBATCH -o ${REP_DIR}/vacf_analysis.out
#SBATCH -e ${REP_DIR}/vacf_analysis.err
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 1-00:00:00
#SBATCH -A dubay-carney
#SBATCH --dependency=afterok:${DEPENDENCY}

module load miniforge
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK

cd "$BASE_DIR"

# Aggregate and plot VACF curves across trials for this replicate
python "$ANALYSIS_SCRIPT"

# Compute diffusion coefficient from VACF via Green-Kubo relation
python "$DIFFUSION_SCRIPT"
EOF

        postid=$(sbatch "$SLURM_ANALYSIS" | awk '{print $4}')
        echo "Submitted VACF analysis job: $postid for rep${rep}_fixmom"
    else
        echo "No valid VACF jobs for rep${rep}_fixmom, skipping analysis job."
    fi
done

