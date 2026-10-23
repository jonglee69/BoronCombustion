"""Extract whole particles (largest connected cluster, bond cutoff 2.3 Å) from ReaxFF snapshots for Blender rendering.
Output: results/figure11_snapshots/<key>.npz (types, positions relative to the B centroid)."""
import sys, os, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'analysis'))
from common import read_data, last_frame, centred, data_path, RESULTS
OUT = os.path.join(RESULTS, 'figure11_snapshots'); os.makedirs(OUT, exist_ok=True)
SNAPS = {   # key: (kind, file in BORON_DATA)
    'a_c1_6.8_final': ('data', '6.8nm/case1_6.8nm_final.data'), 'b_c2_6.8_final': ('data', '6.8nm/case2_6.8nm_final.data'),
    'c_c2_10_final': ('data', '10nm/case2_10.0nm_final.data'), 'd_c2_15_final': ('data', '15nm/case2_15.0nm_final.data'),
    'g_c2_10_init': ('data', '10nm/case2_10.0nm_B_H3BO3_B2O3.data'), 'h_c2_10_melt': ('dump', '10nm/case2_10.0nm_melt.lammpstrj'),
    'i_c2_10_equil': ('data', '10nm/case2_10.0nm_equil.data'), 'j_c2_10_final': ('data', '10nm/case2_10.0nm_final.data')}
for k in (sys.argv[1:] or SNAPS):
    kind, f = SNAPS[k]
    if kind == 'data':
        _, t, x, L, lo = read_data(data_path(f))
    else:
        _, _, t, x, L, lo = last_frame(data_path(f))
    d = centred(x, t, L, lo); r = np.linalg.norm(d, axis=1)
    if k == 'g_c2_10_init':          # the initial shell is not yet bonded to the core: keep everything inside the shell
        keep = r < 63.0
    else:
        lab = connected_components(cKDTree(d).sparse_distance_matrix(cKDTree(d), 2.3), directed=False)[1]
        keep = lab == np.bincount(lab).argmax()
    np.savez(os.path.join(OUT, f'{k}.npz'), t=t[keep], x=d[keep], n_total=len(t))
    print(k, 'atoms kept', keep.sum(), 'of', len(t))
