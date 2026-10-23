# Legacy analysis (submitted version) — superseded

`analysis_size_extrapolation.ipynb` produced Figure 9 (O penetration rate, O flux, enhancement factor and shrinking-core
extrapolation) of the originally submitted manuscript. It is kept for transparency only; **do not use it**.

It contains two errors that were found during the revision:

1. **Coordinate origin for Case 2.** O atoms inside the core were counted in a sphere centred at `box_L/2` with the Case 1
   box length for both cases (`SIZE_CONFIG[...]['box_L']`). The Case 2 boxes are 15 Å larger in each direction, so the
   counting sphere for Case 2 was displaced by ≈13 Å from the particle. This produced the apparent size-dependent
   enhancement (6.8 nm: 1.93 vs 1.26; 10 nm: 3.55 vs 4.65; 15 nm: 3.70 vs 7.82 atoms ps⁻¹).
2. **Flux units.** The flux panel was labelled atoms nm⁻² ps⁻¹ but plotted values 100× too large.

`analysis/reproduce_original_fig9.py` reproduces the original numbers and shows the effect of the origin. The corrected,
particle-centred analysis (`analysis/o_uptake.py`) gives 0.98/1.71, 4.04/3.97 and 8.50/9.30 atoms ps⁻¹ (Case 1/Case 2), and
shows that the uptake is confined to within a few ångströms of the core–shell interface (Figure 10 of the revised manuscript).
