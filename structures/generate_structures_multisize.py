"""
다중 크기 B 나노입자 구조 생성기
Usage:
  python generate_structures_multisize.py --size 10   # 10nm
  python generate_structures_multisize.py --size 15   # 15nm
  python generate_structures_multisize.py --size 6.8  # current (default)

생성 파일:
  case1_{size}nm_B_B2O3.data
  case2_{size}nm_B_H3BO3_B2O3.data
"""

import argparse
import numpy as np
from scipy.spatial import cKDTree
from ase.cluster import Icosahedron
from ase.io import write
from ase import Atoms

# ── Size → noshells mapping ──────────────────────────────────
# Icosahedron radius ≈ noshells * lattice_const * 0.951
# For lattice_const=3.0: radius ≈ noshells * 2.85 Å
# diameter = 2 * radius
SIZE_TO_NOSHELLS = {
    6.8:  17,   # ~6.8 nm (current)
    10:   23,   # ~10 nm
    15:   34,   # ~15 nm
    20:   45,   # ~20 nm
}

B_LATTICE_CONST  = 3.0
B2O3_THICKNESS   = 10.0   # 1 nm
H3BO3_THICKNESS  = 5.0    # 0.5 nm
O2_THICKNESS     = 30.0
BOX_PADDING      = 10.0
MIN_DIST         = 2.2
O2_MIN_DIST      = 3.0
FILL_FRACTION    = 0.30

B2O3_B_DENSITY   = 0.04254 * FILL_FRACTION
B2O3_O_DENSITY   = 0.06381 * FILL_FRACTION
H3BO3_B_DENSITY  = 0.01397 * FILL_FRACTION
H3BO3_O_DENSITY  = 0.04191 * FILL_FRACTION
H3BO3_H_DENSITY  = 0.04191 * FILL_FRACTION
O2_DENSITY       = 5e-3


# ── Helpers ──────────────────────────────────────────────────

def random_point_in_shell(r_inner, r_outer):
    u   = np.random.uniform(0, 1)
    r   = (r_inner**3 + u * (r_outer**3 - r_inner**3)) ** (1/3)
    ct  = np.random.uniform(-1, 1)
    st  = np.sqrt(max(0, 1 - ct**2))
    phi = np.random.uniform(0, 2 * np.pi)
    return np.array([r*st*np.cos(phi), r*st*np.sin(phi), r*ct])


def place_atoms(r_inner, r_outer, n_target, existing, min_d=MIN_DIST, label=""):
    placed, recent, tree = [], [], None
    for _ in range(n_target * 300):
        if len(placed) >= n_target:
            break
        c    = random_point_in_shell(r_inner, r_outer)
        c3   = c.reshape(1, 3)
        skip = False
        if existing or placed:
            if tree is None:
                tree = cKDTree(np.array(existing + placed)) if existing else cKDTree(np.array(placed))
            d, _ = tree.query(c3, k=1)
            if d[0] < min_d:
                skip = True
        if not skip and recent:
            d2, _ = cKDTree(np.array(recent)).query(c3, k=1)
            if d2[0] < min_d:
                skip = True
        if skip:
            continue
        placed.append(c.tolist())
        recent.append(c.tolist())
        if len(placed) % 100 == 0:
            tree = cKDTree(np.array(existing + placed)) if existing else cKDTree(np.array(placed))
            recent = []
        if label and len(placed) % 500 == 0:
            print(f"    {label}: {len(placed)}/{n_target}")
    if len(placed) < n_target:
        print(f"  WARNING [{label}]: {len(placed)}/{n_target}")
    return np.array(placed) if placed else np.zeros((0,3)), len(placed)


def place_o2(r_inner, r_outer, n_target, existing):
    pos, syms, placed, recent, tree = [], [], 0, [], None
    all_pos = list(existing)
    for _ in range(n_target * 300):
        if placed >= n_target:
            break
        c   = random_point_in_shell(r_inner, r_outer)
        d   = np.random.randn(3); d /= np.linalg.norm(d)
        o1, o2 = c - 0.605*d, c + 0.605*d
        cands   = np.array([o1, o2])
        skip    = False
        if all_pos:
            if tree is None:
                tree = cKDTree(np.array(all_pos))
            ds, _ = tree.query(cands, k=1)
            if ds.min() < O2_MIN_DIST:
                skip = True
        if not skip and recent:
            ds2, _ = cKDTree(np.array(recent)).query(cands, k=1)
            if ds2.min() < O2_MIN_DIST:
                skip = True
        if skip:
            continue
        pos.extend([o1.tolist(), o2.tolist()])
        syms.extend(['O', 'O'])
        all_pos.extend([o1.tolist(), o2.tolist()])
        recent.extend([o1.tolist(), o2.tolist()])
        placed += 1
        if placed % 100 == 0:
            tree = cKDTree(np.array(all_pos)); recent = []
        if placed % 1000 == 0:
            print(f"    O2: {placed}/{n_target}")
    if placed < n_target:
        print(f"  WARNING [O2]: {placed}/{n_target}")
    print(f"  O2: {placed} molecules = {placed*2} O atoms")
    return np.array(pos) if pos else np.zeros((0,3)), syms


def build_shells(r_inner, r_outer, existing, densities_syms):
    """Place multiple atom species sequentially."""
    all_pos, positions, symbols = list(existing), [], []
    for (density, sym, min_d) in densities_syms:
        vol = (4/3)*np.pi*(r_outer**3 - r_inner**3)
        n   = int(vol * density)
        p, _ = place_atoms(r_inner, r_outer, n, all_pos, min_d=min_d,
                            label=f"{sym}")
        if len(p):
            all_pos.extend(p.tolist())
            positions.append(p)
            symbols += [sym] * len(p)
    return (np.vstack(positions) if positions else np.zeros((0,3))), symbols


def write_lammps_data(fname, positions, symbols, box_size, size_nm):
    type_map = {'H':1, 'B':2, 'O':3}
    mass_map = {'H':1.0080, 'B':10.811, 'O':15.999}
    half     = box_size / 2.0
    shifted  = np.array(positions) + half
    n        = len(symbols)
    with open(fname, 'w') as f:
        f.write(f"LAMMPS data: B nanoparticle {size_nm}nm\n\n")
        f.write(f"{n} atoms\n3 atom types\n\n")
        f.write(f"0.0 {box_size:.4f} xlo xhi\n")
        f.write(f"0.0 {box_size:.4f} ylo yhi\n")
        f.write(f"0.0 {box_size:.4f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for s, tid in sorted(type_map.items(), key=lambda x: x[1]):
            f.write(f"{tid} {mass_map[s]:.4f}  # {s}\n")
        f.write("\nAtoms  # charge\n\n")
        for i,(pos,sym) in enumerate(zip(shifted, symbols)):
            f.write(f"{i+1} {type_map[sym]} 0.0 {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")
    cnt = {s: symbols.count(s) for s in ['H','B','O'] if symbols.count(s)}
    print(f"Written {fname}: {n} atoms, box={box_size:.1f} Å  {cnt}")


def write_lammps_input(size_nm, case, r_core, box_size):
    """Generate LAMMPS input file for given size and case."""
    tag  = f"{size_nm}nm"
    data = f"case{case}_{tag}_B_B2O3.data" if case==1 else f"case{case}_{tag}_B_H3BO3_B2O3.data"
    fname = f"in.case{case}_{tag}"

    with open(fname, 'w') as f:
        f.write(f"""\
# Case {case} — {size_nm} nm B nanoparticle
# ReaxFF melt-and-quench protocol

package         omp 24
units           real
atom_style      charge
boundary        p p p

read_data       {data}

pair_style      reaxff NULL safezone 5.0 mincap 200
pair_coeff      * * ffield.reax.BOH H B O

fix             qeq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff
neighbor        2.0 bin
neigh_modify    every 5 delay 0 check yes

variable        dt equal 0.25

# Stage 0: Damped pre-relax at 100 K
timestep        0.05
velocity        all create 10.0 {42+case} dist gaussian mom yes
fix             pre1 all nve/limit 0.03
fix             pre2 all langevin 100.0 100.0 5.0 {42+case}
thermo          100
thermo_style    custom step temp pe ke etotal press
run             1000
unfix pre1
unfix pre2
reset_timestep  0

# Stage 1: Melt at 3000 K (10 ps) — sufficient for B2O3 amorphization, below B vaporization
timestep        ${{dt}}
velocity        all create 3000.0 {12345+case} dist gaussian mom yes rot yes
fix             nvt1 all nvt temp 3000.0 3000.0 25.0
thermo          200
thermo_style    custom step temp pe ke etotal press
dump            melt all custom 2000 case{case}_{tag}_melt.lammpstrj id type x y z q
dump_modify     melt sort id element H B O
run             40000
unfix nvt1
undump melt

# Stage 2: Quench to 2500 K (10 ps)
fix             nvt2 all nvt temp 3000.0 2500.0 25.0
run             40000
unfix nvt2

# Stage 3: Equilibrate at 2500 K (10 ps)
fix             nvt3 all nvt temp 2500.0 2500.0 25.0
run             40000
unfix nvt3
write_data      case{case}_{tag}_equil.data

# Stage 4: Production MD at 2500 K (100 ps)
fix             nvt4 all nvt temp 2500.0 2500.0 25.0
thermo          200
thermo_style    custom step temp pe ke etotal press vol
dump            prod all custom 400 case{case}_{tag}_prod.lammpstrj id type x y z q
dump_modify     prod sort id element H B O
fix             bonds all reaxff/bonds 200 case{case}_{tag}_bonds.txt
fix             species all reaxff/species 200 1 200 case{case}_{tag}_species.txt
run             400000
write_data      case{case}_{tag}_final.data
print "Case {case} {size_nm}nm complete!"
""")
    print(f"Written {fname}")


# ── Main builders ─────────────────────────────────────────────

def build_case1(size_nm, noshells, seed=42):
    print(f"\n{'='*60}")
    print(f"CASE 1 — {size_nm} nm: B core + B2O3 shell + O2")
    print(f"{'='*60}")
    np.random.seed(seed)

    core   = Icosahedron('B', noshells=noshells, latticeconstant=B_LATTICE_CONST)
    core.positions -= core.positions.mean(axis=0)
    r_core = np.linalg.norm(core.positions, axis=1).max()
    n_core = len(core)
    print(f"B core: {n_core} atoms, radius={r_core:.1f} Å ({2*r_core/10:.1f} nm)")

    r_b_in  = r_core + 2.5
    r_b_out = r_b_in + B2O3_THICKNESS
    b2o3_pos, b2o3_sym = build_shells(r_b_in, r_b_out, list(core.positions), [
        (B2O3_B_DENSITY, 'B', MIN_DIST),
        (B2O3_O_DENSITY, 'O', MIN_DIST),
    ])

    r_o2_in  = r_b_out + 5.0
    r_o2_out = r_o2_in + O2_THICKNESS
    vol_o2   = (4/3)*np.pi*(r_o2_out**3 - r_o2_in**3)
    n_o2     = int(vol_o2 * O2_DENSITY)
    print(f"O2 gas: target {n_o2} molecules")
    existing = list(core.positions) + (b2o3_pos.tolist() if len(b2o3_pos) else [])
    o2_pos, o2_sym = place_o2(r_o2_in, r_o2_out, n_o2, existing)

    all_pos = np.vstack([core.positions] +
                        ([b2o3_pos] if len(b2o3_pos) else []) +
                        ([o2_pos]   if len(o2_pos)   else []))
    all_sym = ['B']*n_core + b2o3_sym + o2_sym

    box     = 2*(r_o2_out + BOX_PADDING)
    tag     = f"{size_nm}nm"
    fname   = f"case1_{tag}_B_B2O3.data"
    write_lammps_data(fname, all_pos, all_sym, box, size_nm)
    write_lammps_input(size_nm, 1, r_core, box)
    return r_core, box


def build_case2(size_nm, noshells, seed=123):
    print(f"\n{'='*60}")
    print(f"CASE 2 — {size_nm} nm: B core + H3BO3 + B2O3 + O2")
    print(f"{'='*60}")
    np.random.seed(seed)

    core   = Icosahedron('B', noshells=noshells, latticeconstant=B_LATTICE_CONST)
    core.positions -= core.positions.mean(axis=0)
    r_core = np.linalg.norm(core.positions, axis=1).max()
    n_core = len(core)
    print(f"B core: {n_core} atoms, radius={r_core:.1f} Å")

    # H3BO3 interlayer
    r_h_in  = r_core + 2.5
    r_h_out = r_h_in + H3BO3_THICKNESS
    h_pos, h_sym = build_shells(r_h_in, r_h_out, list(core.positions), [
        (H3BO3_B_DENSITY, 'B', MIN_DIST),
        (H3BO3_O_DENSITY, 'O', MIN_DIST),
        (H3BO3_H_DENSITY, 'H', 2.0),
    ])

    # B2O3 outer shell
    r_b_in  = r_h_out + 2.5
    r_b_out = r_b_in + B2O3_THICKNESS
    existing2 = list(core.positions) + (h_pos.tolist() if len(h_pos) else [])
    b_pos, b_sym = build_shells(r_b_in, r_b_out, existing2, [
        (B2O3_B_DENSITY, 'B', MIN_DIST),
        (B2O3_O_DENSITY, 'O', MIN_DIST),
    ])

    # O2 gas
    r_o2_in  = r_b_out + 5.0
    r_o2_out = r_o2_in + O2_THICKNESS
    vol_o2   = (4/3)*np.pi*(r_o2_out**3 - r_o2_in**3)
    n_o2     = int(vol_o2 * O2_DENSITY)
    print(f"O2 gas: target {n_o2} molecules")
    existing3 = existing2 + (b_pos.tolist() if len(b_pos) else [])
    o2_pos, o2_sym = place_o2(r_o2_in, r_o2_out, n_o2, existing3)

    all_pos = np.vstack([core.positions] +
                        ([h_pos]  if len(h_pos)  else []) +
                        ([b_pos]  if len(b_pos)  else []) +
                        ([o2_pos] if len(o2_pos) else []))
    all_sym = ['B']*n_core + h_sym + b_sym + o2_sym

    box   = 2*(r_o2_out + BOX_PADDING)
    tag   = f"{size_nm}nm"
    fname = f"case2_{tag}_B_H3BO3_B2O3.data"
    write_lammps_data(fname, all_pos, all_sym, box, size_nm)
    write_lammps_input(size_nm, 2, r_core, box)
    return r_core, box


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=float, required=True,
                        help='Target diameter in nm (e.g. 10, 15)')
    parser.add_argument('--case', type=int, default=0,
                        help='1=Case1 only, 2=Case2 only, 0=both (default)')
    args = parser.parse_args()

    size = args.size
    # Find closest noshells
    if size in SIZE_TO_NOSHELLS:
        noshells = SIZE_TO_NOSHELLS[size]
    else:
        # Estimate: radius ≈ noshells * 2.85 Å → noshells ≈ diameter*5/2/2.85
        noshells = max(5, int(size * 10 / 2 / 2.85))
        print(f"Estimated noshells={noshells} for {size}nm")

    print(f"\nGenerating {size} nm structures (noshells={noshells})")

    if args.case in (0, 1):
        build_case1(size, noshells)
    if args.case in (0, 2):
        build_case2(size, noshells)

    print(f"\nDone! Run with:")
    print(f"  OMP_NUM_THREADS=24 lmp -sf omp -in in.case1_{size}nm -log case1_{size}nm.log &")
    print(f"  OMP_NUM_THREADS=24 lmp -sf omp -in in.case2_{size}nm -log case2_{size}nm.log &")
