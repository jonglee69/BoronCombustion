"""Figure 10 of the revised manuscript. Requires results/uptake_c{1,2}_{6.8,10,15}.npz (analysis/o_uptake.py)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
from common import RESULTS, SPECIES_CASE2, data_path
from style import plt, C1, C2, INK, MUTED, h2o
SIZES = [('6.8', '6.8 nm'), ('10', '10.0 nm'), ('15', '15.0 nm')]
D = {(c, s): np.load(os.path.join(RESULTS, f'uptake_c{c}_{s}.npz')) for c in (1, 2) for s, _ in SIZES}
fig = plt.figure(figsize=(7.2, 5.6)); gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42)
for i, (s, lab) in enumerate(SIZES):
    ax = fig.add_subplot(gs[0, i])
    for c, col in ((1, C1), (2, C2)):
        d = D[c, s]; ax.plot(d['t'], d['N'][:, 0] - d['N'][0, 0], color=col, label=f'Case {c}')
    ax.set_title(lab, color=INK, fontsize=8.5); ax.set_xlabel('Production time (ps)')
    if i == 0:
        ax.set_ylabel('ΔN$_O$ inside core boundary'); ax.legend(loc='upper left')
        ax.text(-0.3, 1.08, '(a)', transform=ax.transAxes, fontweight='bold', fontsize=10)
ax = fig.add_subplot(gs[1, 0])
for (s, lab), mk in zip(SIZES, 'os^'):
    for c, col, ls in ((1, C1, '-'), (2, C2, '--')):
        ax.plot(D[c, s]['delta'], D[c, s]['rate'], ls, color=col, marker=mk, ms=4, label=lab if c == 1 else None)
ax.axhline(0, color=MUTED, lw=0.6); ax.set_xlabel('Boundary depth below R$_{core}$ (Å)'); ax.set_ylabel('O uptake rate (atoms ps$^{-1}$)')
ax.legend(title='solid: Case 1, dashed: Case 2', title_fontsize=6.5, fontsize=6.5, loc='upper right')
ax.text(-0.3, 1.08, '(b)', transform=ax.transAxes, fontweight='bold', fontsize=10)
ax = fig.add_subplot(gs[1, 1]); rb = D[1, '10']['rb'][:-1] + 0.5
for c, col in ((1, C1), (2, C2)):
    ax.plot(rb, D[c, '10']['rhoO0'], color=col, lw=1.0, ls=':', label=f'Case {c}, 0 ps')
    ax.plot(rb, D[c, '10']['rhoO1'], color=col, label=f'Case {c}, 100 ps')
ax.plot(rb, D[1, '10']['rhoB1'] * 0.5, color=MUTED, lw=0.8, label='B (Case 1, ×0.5)')
ax.axvline(float(D[1, '10']['R0']), color=MUTED, lw=0.6, ls='--')
ax.set_xlim(30, 70); ax.set_ylim(0, 0.062); ax.set_xlabel('r (Å)'); ax.set_ylabel('Number density (Å$^{-3}$)')
ax.set_title('10.0 nm', color=INK, fontsize=8.5); ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.02, 1.0))
ax.text(-0.3, 1.08, '(c)', transform=ax.transAxes, fontweight='bold', fontsize=10)
ax = fig.add_subplot(gs[1, 2])
for s, lab in SIZES:
    T, N = h2o(data_path(SPECIES_CASE2[s])); m = T <= 100.0
    ax.plot(T[m], N[m], color=C2, lw=1.2); ax.text(T[m][-1] + 3, N[m][-1], lab, color=INK, fontsize=7, va='center')
ax.set_xlim(0, 125); ax.set_xticks([0, 50, 100]); ax.set_xlabel('Production time (ps)'); ax.set_ylabel('H$_2$O molecules (Case 2)')
ax.text(-0.3, 1.08, '(d)', transform=ax.transAxes, fontweight='bold', fontsize=10)
for ext, dpi in (('png', 300), ('pdf', 300)):
    fig.savefig(os.path.join(RESULTS, f'Figure10.{ext}'), dpi=dpi, bbox_inches='tight', facecolor='white')
for s, lab in SIZES:
    print(lab, ' '.join(f'Case {c}: {D[c, s]["rate"][0]:.2f} ± {D[c, s]["rate_err"][0]:.2f}' for c in (1, 2)), 'atoms/ps')
