"""
B 나노입자 + B2O3/H3BO3 셸 구조 생성기 (v2 - 개별 원자 배치)

전략:
  - B2O3/H3BO3 셸: 분자 단위가 아닌 개별 B, O, H 원자를 30% 초기밀도로 배치
  - LAMMPS melt-and-quench으로 올바른 유리 네트워크 구조 형성
  - O2: 분자 단위 유지 (실제 반응물)
  - min_dist = 2.2 Å (ReaxFF 안정 영역)
"""

import numpy as np
from scipy.spatial import cKDTree
from ase.cluster import Icosahedron
from ase.io import write
from ase import Atoms

# ============================================================
# Parameters
# ============================================================
B_CORE_SHELLS   = 17        # ~10nm B core
B_LATTICE_CONST = 3.0       # Icosahedron lattice constant (Å)
B2O3_THICKNESS  = 10.0      # B2O3 shell thickness (Å) = 1nm
H3BO3_THICKNESS = 5.0       # H3BO3 interlayer thickness (Å) = 0.5nm
O2_THICKNESS    = 30.0      # O2 gas region thickness (Å)
BOX_PADDING     = 10.0      # Extra vacuum padding (Å)
MIN_DIST        = 2.2       # Min interatomic distance for shell atoms (Å)
O2_MIN_DIST     = 3.0       # Min inter-O2-molecule distance (Å)

# Densities at 30% of crystal density (for safe initial packing)
FILL_FRACTION   = 0.30

# B2O3: rho=2.46 g/cm3, M=69.62 → 0.02127 formula units/Å³ → B:0.04254, O:0.06381 atoms/Å³
B2O3_B_DENSITY  = 0.04254 * FILL_FRACTION   # B atoms/Å³
B2O3_O_DENSITY  = 0.06381 * FILL_FRACTION   # O atoms/Å³

# H3BO3: rho=1.435 g/cm3, M=61.83 → 0.01397 formula units/Å³ → B:0.01397, O:0.04191, H:0.04191 atoms/Å³
H3BO3_B_DENSITY = 0.01397 * FILL_FRACTION
H3BO3_O_DENSITY = 0.04191 * FILL_FRACTION
H3BO3_H_DENSITY = 0.04191 * FILL_FRACTION

# O2 gas: ~1 atm at 2500K ≈ 5e-6 molecules/Å³ → 1e-5 O atoms/Å³ (sparse)
O2_DENSITY      = 5e-3      # O2 molecules/Å³ (relatively sparse)


# ============================================================
# Core
# ============================================================

def create_b_core():
    atoms = Icosahedron('B', noshells=B_CORE_SHELLS, latticeconstant=B_LATTICE_CONST)
    atoms.positions -= atoms.positions.mean(axis=0)
    print(f"B core: {len(atoms)} atoms")
    return atoms


def get_core_radius(atoms):
    return np.linalg.norm(atoms.positions, axis=1).max()


# ============================================================
# Atom placement helpers
# ============================================================

def random_point_in_shell(r_inner, r_outer):
    u = np.random.uniform(0, 1)
    r = (r_inner**3 + u * (r_outer**3 - r_inner**3)) ** (1.0/3.0)
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(max(0.0, 1 - cos_theta**2))
    phi = np.random.uniform(0, 2 * np.pi)
    return np.array([r * sin_theta * np.cos(phi),
                     r * sin_theta * np.sin(phi),
                     r * cos_theta])


def place_atoms_random(r_inner, r_outer, n_target, existing_positions=None,
                       min_dist=MIN_DIST, max_attempts_factor=300, label=""):
    """
    Place n_target individual atoms in spherical shell with min_dist check.
    Returns (positions_array, n_placed).
    """
    all_placed = list(existing_positions) if existing_positions is not None else []
    positions  = []
    placed     = 0
    tree       = None
    recent     = []
    max_att    = n_target * max_attempts_factor

    for attempt in range(max_att):
        if placed >= n_target:
            break

        cand = random_point_in_shell(r_inner, r_outer)

        # distance check
        too_close = False
        if all_placed:
            if tree is None:
                tree = cKDTree(np.array(all_placed))
            d, _ = tree.query(cand.reshape(1, 3), k=1)
            if d[0] < min_dist:
                too_close = True
        if not too_close and recent:
            d2, _ = cKDTree(np.array(recent)).query(cand.reshape(1, 3), k=1)
            if d2[0] < min_dist:
                too_close = True
        if too_close:
            continue

        # accept
        positions.append(cand)
        all_placed.append(cand)
        recent.append(cand)
        placed += 1

        # rebuild tree every 100 placements
        if placed % 100 == 0:
            tree   = cKDTree(np.array(all_placed))
            recent = []
            if label and placed % 500 == 0:
                print(f"    {label}: {placed}/{n_target} atoms placed...")

    if placed < n_target:
        print(f"  WARNING [{label}]: placed {placed}/{n_target} (shell full at min_dist={min_dist:.1f} Å)")

    return np.array(positions) if positions else np.zeros((0, 3)), placed


def place_o2_molecules(r_inner, r_outer, n_target, existing_positions=None):
    """Place O2 diatomic molecules with bond 1.21 Å, inter-molecule min 3.0 Å."""
    all_placed = list(existing_positions) if existing_positions is not None else []
    positions  = []
    symbols    = []
    placed     = 0
    tree       = None
    recent     = []
    max_att    = n_target * 300

    for attempt in range(max_att):
        if placed >= n_target:
            break

        center    = random_point_in_shell(r_inner, r_outer)
        direction = np.random.randn(3); direction /= np.linalg.norm(direction)
        o1 = center - 0.605 * direction
        o2 = center + 0.605 * direction
        cands = np.array([o1, o2])

        too_close = False
        if all_placed:
            if tree is None:
                tree = cKDTree(np.array(all_placed))
            d, _ = tree.query(cands, k=1)
            if d.min() < O2_MIN_DIST:
                too_close = True
        if not too_close and recent:
            d2, _ = cKDTree(np.array(recent)).query(cands, k=1)
            if d2.min() < O2_MIN_DIST:
                too_close = True
        if too_close:
            continue

        positions.extend([o1, o2])
        symbols.extend(['O', 'O'])
        for p in [o1, o2]:
            all_placed.append(p)
            recent.append(p)
        placed += 1

        if placed % 100 == 0:
            tree   = cKDTree(np.array(all_placed))
            recent = []
        if placed % 1000 == 0:
            print(f"    O2: {placed}/{n_target} molecules placed...")

    if placed < n_target:
        print(f"  WARNING [O2]: placed {placed}/{n_target}")

    print(f"  O2: placed {placed} molecules = {placed*2} O atoms")
    return np.array(positions) if positions else np.zeros((0, 3)), symbols


# ============================================================
# Shell builders
# ============================================================

def build_b2o3_shell(r_inner, r_outer, existing_positions):
    """Place individual B and O atoms for amorphous B2O3 shell."""
    vol = (4/3) * np.pi * (r_outer**3 - r_inner**3)
    n_B = int(vol * B2O3_B_DENSITY)
    n_O = int(vol * B2O3_O_DENSITY)
    print(f"B2O3 shell (individual atoms): target B={n_B}, O={n_O}")
    print(f"  r_inner={r_inner:.1f} Å, r_outer={r_outer:.1f} Å, vol={vol:.0f} Å³, "
          f"fill={FILL_FRACTION*100:.0f}% of crystal density")

    all_pos = list(existing_positions)
    b_pos, nb = place_atoms_random(r_inner, r_outer, n_B, all_pos, label="B(B2O3)")
    all_pos  += list(b_pos)
    o_pos, no = place_atoms_random(r_inner, r_outer, n_O, all_pos, label="O(B2O3)")

    positions = []
    symbols   = []
    if nb > 0:
        positions.append(b_pos);   symbols += ['B'] * nb
    if no > 0:
        positions.append(o_pos);   symbols += ['O'] * no

    pos_arr = np.vstack(positions) if positions else np.zeros((0, 3))
    print(f"  Placed: {nb} B + {no} O = {nb+no} atoms")
    return pos_arr, symbols


def build_h3bo3_layer(r_inner, r_outer, existing_positions):
    """Place individual B, O, H atoms for amorphous H3BO3 layer."""
    vol = (4/3) * np.pi * (r_outer**3 - r_inner**3)
    n_B = int(vol * H3BO3_B_DENSITY)
    n_O = int(vol * H3BO3_O_DENSITY)
    n_H = int(vol * H3BO3_H_DENSITY)
    print(f"H3BO3 layer (individual atoms): target B={n_B}, O={n_O}, H={n_H}")
    print(f"  r_inner={r_inner:.1f} Å, r_outer={r_outer:.1f} Å, vol={vol:.0f} Å³")

    all_pos = list(existing_positions)
    b_pos, nb = place_atoms_random(r_inner, r_outer, n_B, all_pos, label="B(H3BO3)")
    all_pos  += list(b_pos)
    o_pos, no = place_atoms_random(r_inner, r_outer, n_O, all_pos, label="O(H3BO3)")
    all_pos  += list(o_pos)
    # H atoms can be slightly closer (1.0 Å O-H bond, but inter-molecular ≥ 2.0 Å)
    h_pos, nh = place_atoms_random(r_inner, r_outer, n_H, all_pos,
                                    min_dist=2.0, label="H(H3BO3)")

    positions = []
    symbols   = []
    if nb > 0: positions.append(b_pos); symbols += ['B'] * nb
    if no > 0: positions.append(o_pos); symbols += ['O'] * no
    if nh > 0: positions.append(h_pos); symbols += ['H'] * nh

    pos_arr = np.vstack(positions) if positions else np.zeros((0, 3))
    print(f"  Placed: {nb} B + {no} O + {nh} H = {nb+no+nh} atoms")
    return pos_arr, symbols


# ============================================================
# Validation and I/O
# ============================================================

def validate_structure(all_positions, label):
    tree = cKDTree(all_positions)
    dists, _ = tree.query(all_positions, k=2)
    nn = dists[:, 1]
    print(f"\n  Validation ({label}):")
    print(f"    Total atoms: {len(all_positions)}")
    print(f"    Min distance: {nn.min():.3f} Å")
    print(f"    Atoms < 1.0 Å: {np.sum(nn < 1.0)}")
    print(f"    Atoms < 1.5 Å: {np.sum(nn < 1.5)}")
    if nn.min() < 1.0:
        print("    WARNING: Very close atoms found!")


def write_lammps_data(filename, all_positions, all_symbols, box_size):
    type_map = {'H': 1, 'B': 2, 'O': 3}
    mass_map = {'H': 1.0080, 'B': 10.811, 'O': 15.999}

    half    = box_size / 2.0
    shifted = np.array(all_positions) + half
    n_atoms = len(all_symbols)

    # determine which types present
    types_present = sorted(set(all_symbols), key=lambda s: type_map[s])
    n_types = len(types_present)

    with open(filename, 'w') as f:
        f.write("LAMMPS data file for Boron nanoparticle\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"3 atom types\n\n")
        f.write(f"0.0 {box_size:.4f} xlo xhi\n")
        f.write(f"0.0 {box_size:.4f} ylo yhi\n")
        f.write(f"0.0 {box_size:.4f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for sym, tid in sorted(type_map.items(), key=lambda x: x[1]):
            f.write(f"{tid} {mass_map[sym]:.4f}  # {sym}\n")
        f.write("\nAtoms  # charge\n\n")
        for i, (pos, sym) in enumerate(zip(shifted, all_symbols)):
            tid = type_map[sym]
            f.write(f"{i+1} {tid} 0.0 {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")

    print(f"Written {filename}: {n_atoms} atoms, box={box_size:.1f} Å")
    for sym in ['H', 'B', 'O']:
        cnt = all_symbols.count(sym)
        if cnt: print(f"  {sym}: {cnt}")


# ============================================================
# Main builders
# ============================================================

def build_case1():
    print("\n" + "="*60)
    print("CASE 1: B core + B2O3 shell (atoms) + O2 gas")
    print("="*60)
    np.random.seed(42)

    core   = create_b_core()
    r_core = get_core_radius(core)
    print(f"Core radius: {r_core:.1f} Å ({2*r_core/10:.1f} nm diameter)")

    # B2O3 shell
    r_b2o3_in  = r_core + 2.5
    r_b2o3_out = r_b2o3_in + B2O3_THICKNESS
    b2o3_pos, b2o3_sym = build_b2o3_shell(r_b2o3_in, r_b2o3_out,
                                            list(core.positions))

    # O2 gas region
    r_o2_in  = r_b2o3_out + 5.0
    r_o2_out = r_o2_in + O2_THICKNESS
    vol_o2   = (4/3) * np.pi * (r_o2_out**3 - r_o2_in**3)
    n_o2     = int(vol_o2 * O2_DENSITY)
    print(f"O2 gas: target {n_o2} molecules in r=[{r_o2_in:.1f},{r_o2_out:.1f}] Å")
    existing_for_o2 = list(core.positions) + list(b2o3_pos)
    o2_pos, o2_sym  = place_o2_molecules(r_o2_in, r_o2_out, n_o2, existing_for_o2)

    # Combine
    all_pos = np.vstack([core.positions,
                          b2o3_pos if len(b2o3_pos) else np.zeros((0,3)),
                          o2_pos   if len(o2_pos)   else np.zeros((0,3))])
    all_sym = ['B']*len(core) + b2o3_sym + o2_sym

    validate_structure(all_pos, "Case 1")

    box = 2 * (r_o2_out + BOX_PADDING)
    write_lammps_data('case1_B_B2O3.data', all_pos, all_sym, box)
    atoms = Atoms(symbols=all_sym, positions=all_pos + box/2,
                  cell=[box, box, box])
    write('case1_B_B2O3.cif', atoms)
    print(f"Saved case1_B_B2O3.cif")
    return box


def build_case2():
    print("\n" + "="*60)
    print("CASE 2: B core + H3BO3 layer (atoms) + B2O3 shell (atoms) + O2 gas")
    print("="*60)
    np.random.seed(123)

    core   = create_b_core()
    r_core = get_core_radius(core)
    print(f"Core radius: {r_core:.1f} Å ({2*r_core/10:.1f} nm diameter)")

    # H3BO3 interlayer
    r_h_in   = r_core + 2.5
    r_h_out  = r_h_in + H3BO3_THICKNESS
    h_pos, h_sym = build_h3bo3_layer(r_h_in, r_h_out, list(core.positions))

    # B2O3 outer shell
    r_b_in   = r_h_out + 2.5
    r_b_out  = r_b_in + B2O3_THICKNESS
    existing2 = list(core.positions) + list(h_pos)
    b_pos, b_sym = build_b2o3_shell(r_b_in, r_b_out, existing2)

    # O2 gas
    r_o2_in  = r_b_out + 5.0
    r_o2_out = r_o2_in + O2_THICKNESS
    vol_o2   = (4/3) * np.pi * (r_o2_out**3 - r_o2_in**3)
    n_o2     = int(vol_o2 * O2_DENSITY)
    print(f"O2 gas: target {n_o2} molecules in r=[{r_o2_in:.1f},{r_o2_out:.1f}] Å")
    existing3 = existing2 + list(b_pos)
    o2_pos, o2_sym = place_o2_molecules(r_o2_in, r_o2_out, n_o2, existing3)

    # Combine
    all_pos = np.vstack([core.positions,
                          h_pos  if len(h_pos)  else np.zeros((0,3)),
                          b_pos  if len(b_pos)  else np.zeros((0,3)),
                          o2_pos if len(o2_pos) else np.zeros((0,3))])
    all_sym = ['B']*len(core) + h_sym + b_sym + o2_sym

    validate_structure(all_pos, "Case 2")

    box = 2 * (r_o2_out + BOX_PADDING)
    write_lammps_data('case2_B_H3BO3_B2O3.data', all_pos, all_sym, box)
    atoms = Atoms(symbols=all_sym, positions=all_pos + box/2,
                  cell=[box, box, box])
    write('case2_B_H3BO3_B2O3.cif', atoms)
    print(f"Saved case2_B_H3BO3_B2O3.cif")
    return box


if __name__ == '__main__':
    box1 = build_case1()
    box2 = build_case2()

    print("\n" + "="*60)
    print("Structure generation complete!")
    print(f"Case 1 box: {box1:.1f} Å  |  Case 2 box: {box2:.1f} Å")
    print("Next: run LAMMPS with in.case1 and in.case2 (melt-and-quench)")
    print("="*60)
