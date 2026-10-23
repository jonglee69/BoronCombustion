"""Control structure: same B2O3 shell / O2 gas as Case 1, but the core is a sphere carved
from pre-equilibrated amorphous B (2500 K, 0 atm) with the same atom count as the original
Mackay-icosahedral seed.  Shell/gas placement reuses the original generator (same seed)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import generate_structures_multisize as g
from ase.cluster import Icosahedron

def read_bulk(f):
    L = open(f).read().split('\n')
    box = np.array([[float(v) for v in l.split()[:2]] for l in L[:30] if l.rstrip().endswith(('xhi', 'yhi', 'zhi'))])
    i = [k for k, l in enumerate(L) if l.startswith('Atoms')][0]
    x = np.array([[float(v) for v in l.split()[3:6]] for l in L[i + 2:] if len(l.split()) >= 6])
    return x - box[:, 0], box[:, 1] - box[:, 0]

def build(size_nm, case, noshells, seed, out, n_core=None):
    n_core = n_core or len(Icosahedron('B', noshells=noshells, latticeconstant=g.B_LATTICE_CONST))
    x, L = read_bulk('bulkB_2500K.data')
    rep = int(np.ceil(2 * 1.3 * (3 * n_core / (4 * np.pi * len(x) / L.prod())) ** (1 / 3) / L.min())) | 1
    sh = np.array([[i, j, k] for i in range(rep) for j in range(rep) for k in range(rep)]) * L
    big = (x[None] + sh[:, None]).reshape(-1, 3); big -= rep * L / 2
    core = big[np.argsort(np.linalg.norm(big, axis=1))[:n_core]]
    core -= core.mean(0)
    r_core = np.linalg.norm(core, axis=1).max()
    print(f'core {n_core} atoms, rho={len(x)/L.prod():.4f}, r_core={r_core:.2f} Å, tiles={rep}^3')
    np.random.seed(seed)
    r_b_in = r_core + 2.5
    if case == 1:
        layers = [(r_b_in, r_b_in + g.B2O3_THICKNESS, [(g.B2O3_B_DENSITY, 'B', g.MIN_DIST), (g.B2O3_O_DENSITY, 'O', g.MIN_DIST)])]
    else:
        r_h_out = r_b_in + g.H3BO3_THICKNESS
        layers = [(r_b_in, r_h_out, [(g.H3BO3_B_DENSITY, 'B', g.MIN_DIST), (g.H3BO3_O_DENSITY, 'O', g.MIN_DIST), (g.H3BO3_H_DENSITY, 'H', 2.0)]),
                  (r_h_out + 2.5, r_h_out + 2.5 + g.B2O3_THICKNESS, [(g.B2O3_B_DENSITY, 'B', g.MIN_DIST), (g.B2O3_O_DENSITY, 'O', g.MIN_DIST)])]
    pos, sym = [core], ['B'] * n_core
    for ri, ro, spec in layers:
        p, s = g.build_shells(ri, ro, list(np.vstack(pos)), spec)
        pos.append(p); sym += s
    r_out = layers[-1][1]
    r_o2_in = r_out + 5.0; r_o2_out = r_o2_in + g.O2_THICKNESS
    n_o2 = int(4 / 3 * np.pi * (r_o2_out**3 - r_o2_in**3) * g.O2_DENSITY)
    p, s = g.place_o2(r_o2_in, r_o2_out, n_o2, list(np.vstack(pos)))
    pos.append(p); sym += s
    g.write_lammps_data(out, np.vstack(pos), sym, 2 * (r_o2_out + g.BOX_PADDING), size_nm)

if __name__ == '__main__':
    size, case = float(sys.argv[1]), int(sys.argv[2])
    if size in g.SIZE_TO_NOSHELLS or size == 10.0:
        build(size, case, g.SIZE_TO_NOSHELLS[size if size != 10.0 else 10], 42 if case == 1 else 123,
              f'ctrl_case{case}_{size}nm_amorphcore.data')
    else:   # small test particles: core diameter = size, filled at the amorphous-B density
        build(size, case, None, 42 if case == 1 else 123, f'ctrl_case{case}_{size}nm_amorphcore.data',
              n_core=int(0.0897 * 4 / 3 * np.pi * (size * 5) ** 3))
