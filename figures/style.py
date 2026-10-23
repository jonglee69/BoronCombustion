import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
C1, C2, INK, MUTED = '#2166ac', '#d6604d', '#222222', '#6b6b6b'      # Case 1 / Case 2 (CVD-checked pair)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.6, 'legend.frameon': False})


def h2o(f):
    """Molecular H2O count vs time (ps, from the first record) from a fix reaxff/species output file."""
    import numpy as np
    T, N = [], []
    for line in open(f):
        if line.startswith('#'):
            head = line[1:].split(); continue
        v = line.split(); T.append(int(v[0]) * 0.25e-3); N.append(int(v[head.index('H2O')]) if 'H2O' in head else 0)
    T = np.array(T); return T - T[0], np.array(N)
