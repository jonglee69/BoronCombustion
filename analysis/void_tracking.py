"""Central-void tracking during the preparation stages (Figure 11g-j discussion, Figure S9b).

The core atoms are the first n_core atoms of the initial data file (both structure generators write the core first).
Per frame: radial B density, void radius (first radial bin where the B density reaches half of the core plateau),
mean radius of the core atoms that started at r < 10 Å, and the number of O atoms inside the initial core radius.

example:  python void_tracking.py 10nm/case1_10.0nm_B_B2O3.data 10nm/case1_10.0nm_melt.lammpstrj --ncore 37995 --label fcc_c1_10"""
import argparse, os, json, numpy as np
from common import read_data, frames, center_of, data_path, RESULTS, TIMESTEP_FS

ap = argparse.ArgumentParser()
ap.add_argument('init'); ap.add_argument('traj'); ap.add_argument('--ncore', type=int, required=True)
ap.add_argument('--label', required=True)
a = ap.parse_args()
P = lambda f: f if os.path.isabs(f) else data_path(f)
ids0, t0, x0, _, _ = read_data(P(a.init))
core_ids = ids0[:a.ncore]; assert (t0[:a.ncore] == 2).all()
c0 = x0[:a.ncore].mean(0); r0 = np.linalg.norm(x0 - c0, axis=1); r_core = r0[:a.ncore].max(); r_init = dict(zip(ids0, r0))
RB = np.arange(0, 121, 2.0); SV = 4 / 3 * np.pi * (RB[1:]**3 - RB[:-1]**3)
rows, prof = [], []
for step, ids, t, x, L, lo in frames(P(a.traj)):
    c = center_of(x[t == 2], L, lo); dv = x - c; dv -= L * np.round(dv / L); r = np.linalg.norm(dv, axis=1)
    rho = np.histogram(r[t == 2], RB)[0] / SV; prof.append(rho)
    plateau = np.median(rho[(RB[:-1] > 0.45 * r_core) & (RB[:-1] < 0.75 * r_core)])
    inc = np.isin(ids, core_ids); rc = r[inc]; ri = np.array([r_init[i] for i in ids[inc]])
    rows.append(dict(t_ps=step * TIMESTEP_FS * 1e-3, r_void=float(RB[np.argmax(rho >= 0.5 * plateau)]),
                     inner_tracer_r=float(rc[ri < 10].mean()), n_O_inside_rcore=int(((t == 3) & (r < r_core)).sum())))
json.dump(dict(r_core=float(r_core), rows=rows), open(os.path.join(RESULTS, f'void_{a.label}.json'), 'w'), indent=1)
np.save(os.path.join(RESULTS, f'void_profile_{a.label}.npy'), np.array(prof))
for w in rows[::max(1, len(rows) // 10)]:
    print('t=%6.1f ps  r_void=%4.0f Å  inner tracers at %.1f Å  O inside core %d' % (w['t_ps'], w['r_void'], w['inner_tracer_r'], w['n_O_inside_rcore']))
