"""Documents the post-processing error in the original Figure 9 (submitted version).

The original notebook (legacy/analysis_size_extrapolation.ipynb) counted O atoms inside a sphere centred at L_case1/2 for both
cases, but the Case 2 boxes are 15 Å larger, so the Case 2 sphere was displaced by ~13 Å from the particle.  This script
evaluates the uptake rate with (i) the original origin, (ii) the trajectory's own box centre and (iii) the B centroid.

example:  python reproduce_original_fig9.py 10nm/case2_10.0nm_prod.lammpstrj --lcase1 208.3381 --rcore 48.9 --stride 20"""
import argparse, os, numpy as np
from common import frames, center_of, data_path

ap = argparse.ArgumentParser()
ap.add_argument('traj'); ap.add_argument('--lcase1', type=float, required=True); ap.add_argument('--rcore', type=float, required=True)
ap.add_argument('--stride', type=int, default=20)
a = ap.parse_args()
traj = a.traj if os.path.isabs(a.traj) else data_path(a.traj)
T, N = [], []
for k, (step, ids, t, x, L, lo) in enumerate(frames(traj)):
    if k % a.stride:
        continue
    O = x[t == 3]; c = center_of(x[t == 2], L, lo); d = O - c; d -= L * np.round(d / L)
    T.append(step * 0.25e-3)
    N.append([(np.linalg.norm(O - a.lcase1 / 2, axis=1) < a.rcore).sum(), (np.linalg.norm(O - (lo + L / 2), axis=1) < a.rcore).sum(),
              (np.linalg.norm(d, axis=1) < a.rcore).sum()])
T, N = np.array(T), np.array(N, float); m = np.arange(len(T)) >= max(1, len(T) // 10)
r = [np.polyfit(T[m], N[m, j], 1)[0] for j in range(3)]
print(f'box = {L[0]:.1f} Å | rate: original origin {r[0]:.2f}, own box centre {r[1]:.2f}, B centroid {r[2]:.2f} atoms/ps')
