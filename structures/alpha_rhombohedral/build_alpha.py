"""10 nm particle with an alpha-rhombohedral (B12-icosahedral) boron core: sphere of the same outer radius as the
original Mackay-icosahedral core (46.7 Å); shell / O2 gas built with the original generator and seeds."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import generate_structures_multisize as g
from alphaB import alpha_boron

R_CORE = 46.7
def core_sphere(R):
    a = alpha_boron(); n = np.ceil(2 * (R + 5) / np.array([4.9065, 4.9065 * np.sqrt(3) / 2, 12.5658])).astype(int) + 1
    s = a * tuple(n); x = s.positions - s.get_cell().sum(0) / 2
    return x[np.linalg.norm(x, axis=1) <= R]

def build(case):
    core = core_sphere(R_CORE); core -= core.mean(0)
    r_core = np.linalg.norm(core, axis=1).max(); np.random.seed(42 if case == 1 else 123)
    print(f'alpha-B core: {len(core)} atoms, r_core={r_core:.2f} Å')
    r_in = r_core + 2.5
    if case == 1:
        layers = [(r_in, r_in + g.B2O3_THICKNESS, [(g.B2O3_B_DENSITY, 'B', g.MIN_DIST), (g.B2O3_O_DENSITY, 'O', g.MIN_DIST)])]
    else:
        r_h = r_in + g.H3BO3_THICKNESS
        layers = [(r_in, r_h, [(g.H3BO3_B_DENSITY, 'B', g.MIN_DIST), (g.H3BO3_O_DENSITY, 'O', g.MIN_DIST), (g.H3BO3_H_DENSITY, 'H', 2.0)]),
                  (r_h + 2.5, r_h + 2.5 + g.B2O3_THICKNESS, [(g.B2O3_B_DENSITY, 'B', g.MIN_DIST), (g.B2O3_O_DENSITY, 'O', g.MIN_DIST)])]
    pos, sym = [core], ['B'] * len(core)
    for ri, ro, spec in layers:
        p, s = g.build_shells(ri, ro, list(np.vstack(pos)), spec); pos.append(p); sym += s
    r_o2 = layers[-1][1] + 5.0
    n_o2 = int(4 / 3 * np.pi * ((r_o2 + g.O2_THICKNESS)**3 - r_o2**3) * g.O2_DENSITY)
    p, s = g.place_o2(r_o2, r_o2 + g.O2_THICKNESS, n_o2, list(np.vstack(pos))); pos.append(p); sym += s
    g.write_lammps_data(f'alpha_case{case}_10nm.data', np.vstack(pos), sym, 2 * (r_o2 + g.O2_THICKNESS + g.BOX_PADDING), 10.0)
    return len(core)

if __name__ == '__main__':
    print('n_core', build(int(sys.argv[1])))
