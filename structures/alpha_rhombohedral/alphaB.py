"""alpha-rhombohedral boron (R-3m, hexagonal setting; Will & Kiefer, Z. Anorg. Allg. Chem. 2001, 627, 2100)."""
from ase.spacegroup import crystal
def alpha_boron():
    return crystal(['B', 'B'], basis=[(0.11886, 0.88114, 0.89133), (0.19686, 0.80314, 0.02432)], spacegroup=166,
                   cellpar=[4.9065, 4.9065, 12.5658, 90, 90, 120])
