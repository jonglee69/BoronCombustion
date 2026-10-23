"""Figures S6-S8. Requires results/uptake_*.npz and results/shell_*.npy for the three sizes."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
from common import RESULTS, SPECIES_CASE2, data_path
from style import plt, C1, C2, INK, MUTED, h2o
for s, name, lab in (('6.8', 'S6', '6.8 nm'), ('10', 'S7', '10.0 nm'), ('15', 'S8', '15.0 nm')):
    D = {c: np.load(os.path.join(RESULTS, f'uptake_c{c}_{s}.npz')) for c in (1, 2)}
    S = {c: np.load(os.path.join(RESULTS, f'shell_c{c}_{s}.npy')) for c in (1, 2)}
    R0 = float(D[1]['R0']); rb = D[1]['rb'][:-1] + 0.5
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6)); plt.subplots_adjust(hspace=0.5, wspace=0.42)
    ax = axs[0, 0]
    for c, col in ((1, C1), (2, C2)):
        ax.plot(rb, D[c]['rhoB0'], color=col, lw=1.3, label=f'B, Case {c}'); ax.plot(rb, D[c]['rhoO0'], color=col, lw=1.0, ls=':', label=f'O, Case {c}')
    ax.axvline(R0, color=MUTED, lw=0.6, ls='--'); ax.set(xlabel='r (Å)', ylabel='Number density (Å$^{-3}$)', xlim=(0, R0 + 30))
    ax.legend(fontsize=6, loc='upper left'); ax.set_title('Start of production', fontsize=8, color=INK)
    ax = axs[0, 1]
    for c, col in ((1, C1), (2, C2)):
        ax.plot(D[c]['t'], D[c]['N'][:, 0] - D[c]['N'][0, 0], color=col, lw=1.3, label=f'Case {c}')
    ax.set(xlabel='Production time (ps)', ylabel='ΔN$_O$ inside R$_{core}$'); ax.legend(fontsize=7)
    ax = axs[0, 2]
    for c, col in ((1, C1), (2, C2)):
        ax.plot(rb, D[c]['rhoO0'], color=col, lw=1.0, ls=':'); ax.plot(rb, D[c]['rhoO1'], color=col, lw=1.4, label=f'Case {c}, 100 ps')
    ax.axvline(R0, color=MUTED, lw=0.6, ls='--'); ax.set(xlabel='r (Å)', ylabel='O density (Å$^{-3}$)', xlim=(R0 - 25, R0 + 25))
    ax.legend(fontsize=6, loc='upper right'); ax.set_title('dotted: 0 ps', fontsize=7, color=MUTED)
    ax = axs[1, 0]
    for c, col in ((1, C1), (2, C2)):
        ax.plot(S[c][:, 0], S[c][:, 1], color=col, lw=1.3, label=f'Case {c}')
    ax.set(xlabel='Production time (ps)', ylabel='Shell porosity'); ax.legend(fontsize=7)
    ax.set_title(f'shell: {R0 - 3:.0f}–{R0 + 12:.0f} Å', fontsize=7, color=MUTED)
    ax = axs[1, 1]
    for c, col in ((1, C1), (2, C2)):
        ax.plot(S[c][:, 0], S[c][:, 2] / S[c][0, 2] * 100, color=col, lw=1.3, label=f'Case {c}')
    ax.axhline(100, color=MUTED, lw=0.6, ls=':'); ax.set(xlabel='Production time (ps)', ylabel='Shell O content (% of initial)'); ax.legend(fontsize=7)
    ax = axs[1, 2]; T, N = h2o(data_path(SPECIES_CASE2[s])); m = T <= 100.0
    ax.plot(T[m], N[m], color=C2, lw=1.3, label='total (species)')
    ax.plot(S[2][:, 0], S[2][:, 5], color=C2, lw=1.0, ls='--', label='outside particle')
    ax.plot(S[2][:, 0], S[2][:, 4], color=C2, lw=1.0, ls=':', label='inside shell')
    ax.set(xlabel='Production time (ps)', ylabel='H$_2$O molecules (Case 2)'); ax.legend(fontsize=6)
    for a, l in zip(axs.flat, 'abcdef'):
        a.text(-0.3, 1.06, f'({l})', transform=a.transAxes, fontweight='bold', fontsize=9, color=INK)
    fig.suptitle(f'ReaxFF-MD, {lab} particles (particle-centred analysis)', fontsize=9, color=INK, y=0.99)
    fig.savefig(os.path.join(RESULTS, f'Figure{name}.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(name, 'written')
