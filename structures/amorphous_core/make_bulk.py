"""Random pure-B box (40 Å cube, 0.09 Å^-3) as start for an amorphous B core."""
import numpy as np
from scipy.spatial import cKDTree
rng = np.random.default_rng(7); L = 40.0; N = int(0.09 * L**3); pts = []
while len(pts) < N:
    c = rng.uniform(0, L, 3)
    if pts:
        d = np.array(pts) - c; d -= L * np.round(d / L)
        if (np.einsum('ij,ij->i', d, d) < 1.6**2).any(): continue
    pts.append(c)
with open('bulkB_init.data', 'w') as f:
    f.write(f"random B box\n\n{N} atoms\n3 atom types\n\n0 {L} xlo xhi\n0 {L} ylo yhi\n0 {L} zlo zhi\n\nMasses\n\n1 1.008\n2 10.811\n3 15.999\n\nAtoms # charge\n\n")
    for i, p in enumerate(pts): f.write(f"{i+1} 2 0.0 {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
print(N)
