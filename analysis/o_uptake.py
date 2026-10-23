"""Particle-centred O uptake (Figure 10a-c, Figures S6-S8 b,c, Figure S9c).

For every analysed frame the number of O atoms inside spheres of radius R_core - delta (delta = 0...8 Å) centred on the
periodic centroid of the B atoms is counted, and the radial O/B densities at the first and last analysed frames are stored.
Uptake rates are slopes of linear fits after discarding the first 10 % of the run.

example:  python o_uptake.py 10nm/case2_10.0nm_prod.lammpstrj --rcore 48.9 --stride 10 --label c2_10"""
import argparse, os, numpy as np
from common import frames, centred, data_path, RESULTS, TIMESTEP_FS

ap = argparse.ArgumentParser()
ap.add_argument('traj', help='dump file (absolute, or relative to BORON_DATA)')
ap.add_argument('--rcore', type=float, required=True); ap.add_argument('--stride', type=int, default=1)
ap.add_argument('--label', required=True)
a = ap.parse_args()
traj = a.traj if os.path.isabs(a.traj) else data_path(a.traj)
DELTA = np.array([0, 1, 2, 3, 4, 5, 6, 8]); RB = np.arange(0, 141, 1.0); SV = 4 / 3 * np.pi * (RB[1:]**3 - RB[:-1]**3)
T, N, step0 = [], [], None
for k, (step, ids, t, x, L, lo) in enumerate(frames(traj)):
    if k % a.stride:
        continue
    step0 = step if step0 is None else step0
    r = np.linalg.norm(centred(x, t, L, lo), axis=1)
    T.append((step - step0) * TIMESTEP_FS * 1e-3); N.append([((t == 3) & (r < a.rcore - d)).sum() for d in DELTA])
    last = (np.histogram(r[t == 3], RB)[0] / SV, np.histogram(r[t == 2], RB)[0] / SV)
    if len(T) == 1:
        first = last
T, N = np.array(T), np.array(N, float)
m = np.arange(len(T)) >= max(1, len(T) // 10)
rates = [np.polyfit(T[m], N[m, j], 1, cov=True) for j in range(len(DELTA))]
np.savez(os.path.join(RESULTS, f'uptake_{a.label}.npz'), t=T, N=N, delta=DELTA, R0=a.rcore, rb=RB,
         rhoO0=first[0], rhoB0=first[1], rhoO1=last[0], rhoB1=last[1],
         rate=np.array([r[0][0] for r in rates]), rate_err=np.array([np.sqrt(r[1][0, 0]) for r in rates]))
print(a.label, 'rate(R_core) = %.2f +/- %.2f atoms/ps' % (rates[0][0][0], np.sqrt(rates[0][1][0, 0])),
      '| R_core-3 Å: %.2f' % rates[3][0][0], '| R_core-6 Å: %.2f' % rates[6][0][0])
