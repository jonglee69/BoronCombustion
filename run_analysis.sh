#!/usr/bin/env bash
# Reproduce the quantitative MD analyses and figures of the revised manuscript (Figure 10, Figures S6-S9, Table S2).
# Usage:  BORON_DATA=/path/to/extracted/zenodo/data ./run_analysis.sh
# Results (npz/npy/json and figures) are written to ./results (override with BORON_RESULTS).
set -euo pipefail
cd "$(dirname "$0")"
: "${BORON_DATA:?set BORON_DATA to the directory containing 6.8nm/ 10nm/ 15nm/ B_only/ controls/}"
A=analysis
declare -A R=( [6.8]=33.9 [10]=48.9 [15]=73.4 )      # core radius used for counting (Å)
declare -A D=( [6.8]=6.8 [10]=10.0 [15]=15.0 )         # diameter tag in the file names
for s in 6.8 10 15; do
  for c in 1 2; do
    traj="${s}nm/case${c}_${D[$s]}nm_prod.lammpstrj"
    python $A/o_uptake.py "$traj" --rcore ${R[$s]} --stride 10 --label c${c}_${s} &
    python $A/shell_analysis.py "$traj" --rcore ${R[$s]} --stride 20 --label c${c}_${s} &
  done
done
# alpha-rhombohedral starting core (10 nm; dumps every 0.5 ps) and Stage-1 void tracking (FCC-type vs alpha)
for c in 1 2; do
  python $A/o_uptake.py controls/alpha_rhombohedral_core/run_case${c}/prod.lammpstrj --rcore 55 --stride 2 --label aB_c${c}_10 &
  python $A/void_tracking.py controls/alpha_rhombohedral_core/alpha_case${c}_10nm.data controls/alpha_rhombohedral_core/run_case${c}/melt.lammpstrj --ncore 58660 --label aB_c${c}_10 &
done
python $A/void_tracking.py 10nm/case1_10.0nm_B_B2O3.data 10nm/case1_10.0nm_melt.lammpstrj --ncore 37995 --label fcc_c1_10 &
python $A/void_tracking.py 10nm/case2_10.0nm_B_H3BO3_B2O3.data 10nm/case2_10.0nm_melt.lammpstrj --ncore 37995 --label fcc_c2_10 &
wait
python figures/figure10.py
python figures/figureS6_S8.py
python figures/figureS9.py
# documentation of the error in the submitted Figure 9 (original origin vs particle centroid), 10 nm Case 2
python $A/reproduce_original_fig9.py 10nm/case2_10.0nm_prod.lammpstrj --lcase1 208.3381 --rcore 48.9 --stride 20
