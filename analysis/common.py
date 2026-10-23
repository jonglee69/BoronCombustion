"""Shared I/O and geometry helpers for the ReaxFF boron-nanoparticle analyses.

Data location: set the environment variable BORON_DATA to the directory into which the Zenodo archives were extracted
(it contains 6.8nm/, 10nm/, 15nm/, B_only/, controls/, forcefield/).  Results are written to ./results by default."""
import os
import numpy as np

DATA = os.environ.get('BORON_DATA', os.path.join(os.path.dirname(__file__), '..', 'data'))
RESULTS = os.environ.get('BORON_RESULTS', os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(RESULTS, exist_ok=True)
TIMESTEP_FS = 0.25
# core radius used to count O uptake (radius of the initial Mackay-icosahedral B core, as in the original analysis)
R_CORE = {'6.8': 33.9, '10': 48.9, '15': 73.4}


def data_path(*parts):
    return os.path.join(DATA, *parts)


def read_data(f):
    """LAMMPS data file (atom_style charge) -> ids, types, positions, box lengths, box lower bounds (sorted by id)."""
    L = open(f).read().split('\n')
    box = np.array([[float(v) for v in l.split()[:2]] for l in L[:40] if l.rstrip().endswith(('xhi', 'yhi', 'zhi'))])
    i = [k for k, l in enumerate(L) if l.startswith('Atoms')][0]
    a = np.array([[float(v) for v in l.split()[:6]] for l in L[i + 2:] if len(l.split()) >= 6])
    a = a[np.argsort(a[:, 0])]
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 3:6], box[:, 1] - box[:, 0], box[:, 0]


def frames(f):
    """Iterate over a LAMMPS custom dump (id type x y z ...): yields step, ids, types, positions, box lengths, lower bounds."""
    with open(f) as fh:
        while fh.readline():
            step = int(fh.readline()); fh.readline(); n = int(fh.readline()); fh.readline()
            box = np.array([[float(v) for v in fh.readline().split()[:2]] for _ in range(3)])
            cols = fh.readline().split()[2:]
            a = np.loadtxt([fh.readline() for _ in range(n)], usecols=[cols.index(c) for c in ('id', 'type', 'x', 'y', 'z')])
            a = a[np.argsort(a[:, 0])]
            yield step, a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2:5], box[:, 1] - box[:, 0], box[:, 0]


def last_frame(f, chunk=80_000_000):
    """Last frame of a (large) dump file without reading the whole file."""
    with open(f, 'rb') as fh:
        fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - chunk)); txt = fh.read().decode()
    txt = txt[txt.rfind('ITEM: TIMESTEP'):]
    import io, tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.lammpstrj', delete=False) as t:
        t.write(txt)
    out = next(frames(t.name)); os.unlink(t.name)
    return out


def center_of(x, L, lo):
    """Periodic-safe centroid (circular mean per axis)."""
    th = 2 * np.pi * (x - lo) / L
    return lo + (np.arctan2(np.sin(th).mean(0), np.cos(th).mean(0)) % (2 * np.pi)) * L / (2 * np.pi)


def centred(x, t, L, lo):
    """Positions relative to the periodic centroid of the B atoms (type 2), minimum image."""
    c = center_of(x[t == 2], L, lo)
    d = x - c
    return d - L * np.round(d / L)


# production trajectories / species files inside BORON_DATA (Case 2, 10 nm: complete 100 ps restart run)
PROD = {('6.8', 1): '6.8nm/case1_6.8nm_prod.lammpstrj', ('6.8', 2): '6.8nm/case2_6.8nm_prod.lammpstrj',
        ('10', 1): '10nm/case1_10.0nm_prod.lammpstrj', ('10', 2): '10nm/case2_10.0nm_prod.lammpstrj',
        ('15', 1): '15nm/case1_15.0nm_prod.lammpstrj', ('15', 2): '15nm/case2_15.0nm_prod.lammpstrj'}
SPECIES_CASE2 = {'6.8': '6.8nm/case2_6.8nm_species.txt', '10': '10nm/case2_10.0nm_species_prod.txt',
                 '15': '15nm/case2_15.0nm_species.txt'}
PS_PER_FRAME = 0.1        # production dumps every 400 steps
