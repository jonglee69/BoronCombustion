"""Oxide-shell porosity / composition and H2O location (Figures S6-S8 d-f, Table S2).

Shell = fixed radial window [R_core - 3, R_core + 12] Å around the B centroid.  Per analysed frame:
  porosity  : fraction of 20 000 random probe points in the shell whose nearest atom is farther than 2.0 Å
  n_O, n_B  : O and B atoms in the shell
  H2O       : molecular water (O with two H within 1.2 Å and no B within 2.0 Å), inside the shell / outside it

example:  python shell_analysis.py 10nm/case2_10.0nm_prod.lammpstrj --rcore 48.9 --stride 20 --label c2_10"""
import argparse, os, numpy as np
from scipy.spatial import cKDTree
from common import frames, centred, data_path, RESULTS, TIMESTEP_FS

ap = argparse.ArgumentParser()
ap.add_argument('traj'); ap.add_argument('--rcore', type=float, required=True)
ap.add_argument('--stride', type=int, default=1); ap.add_argument('--label', required=True)
a = ap.parse_args()
traj = a.traj if os.path.isabs(a.traj) else data_path(a.traj)
lo_r, hi_r = a.rcore - 3.0, a.rcore + 12.0
rng = np.random.default_rng(1)
u, ct, ph = rng.random(20000), rng.uniform(-1, 1, 20000), rng.uniform(0, 2 * np.pi, 20000)
rr = (lo_r**3 + u * (hi_r**3 - lo_r**3)) ** (1 / 3); st = np.sqrt(1 - ct**2)
probe = np.c_[rr * st * np.cos(ph), rr * st * np.sin(ph), rr * ct]
rows, step0 = [], None
for k, (step, ids, t, x, L, lo) in enumerate(frames(traj)):
    if k % a.stride:
        continue
    step0 = step if step0 is None else step0
    d = centred(x, t, L, lo); r = np.linalg.norm(d, axis=1)
    near = (r > lo_r - 6) & (r < hi_r + 6)
    dmin, _ = cKDTree(d[near]).query(probe, k=1)
    shell = (r >= lo_r) & (r < hi_r)
    tree = cKDTree(d); iO = np.where(t == 3)[0]
    nH = np.array([sum(t[j] == 1 for j in nb) for nb in tree.query_ball_point(d[iO], 1.2)])
    nB = np.array([sum(t[j] == 2 for j in nb) for nb in tree.query_ball_point(d[iO], 2.0)])
    w = iO[(nH == 2) & (nB == 0)]
    rows.append(((step - step0) * TIMESTEP_FS * 1e-3, (dmin > 2.0).mean(), ((t == 3) & shell).sum(), ((t == 2) & shell).sum(),
                 (r[w] < hi_r).sum(), (r[w] >= hi_r).sum()))
A = np.array(rows)
np.save(os.path.join(RESULTS, f'shell_{a.label}.npy'), A)   # columns: t_ps, porosity, n_O_shell, n_B_shell, H2O_in, H2O_out
print(f'{a.label}: porosity {A[0,1]:.3f}->{A[-1,1]:.3f}  O_shell {A[0,2]:.0f}->{A[-1,2]:.0f}  B_shell {A[0,3]:.0f}->{A[-1,3]:.0f}'
      f'  H2O in/out {A[-1,4]:.0f}/{A[-1,5]:.0f}')
