# BoronCombustion

ReaxFF reactive molecular dynamics (ReaxFF-MD) inputs, structure generators, and analysis scripts for boron nanoparticles
(BNPs) with a B₂O₃ shell (Case 1) or an H₃BO₃ interlayer + B₂O₃ shell (Case 2), as reported in

> J. H. Lee, S. H. Joo, Y. H. Lee, H. Nersisyan, *Degradable nanoscale interface engineering and particle-size refinement
> synergistically unlock high-energy boron reactivity*, Energy & Fuels (submitted).

The simulation data (structures, trajectories, species and log files; ~12 GB compressed) are archived in Zenodo:
**https://doi.org/10.5281/zenodo.23589942**.

## Contents

| Path | Content |
|---|---|
| `structures/` | `generate_structures.py` (6.8 nm) and `generate_structures_multisize.py` (10 and 15 nm): Mackay-icosahedral (FCC-type, a = 3.0 Å) B core + stochastically placed shell atoms + O₂ gas. `make_B_only.py`: oxidizer-free control. `amorphous_core/`: equilibrated amorphous-B core (control). `alpha_rhombohedral/`: α-rhombohedral (B₁₂) core (control, Figure S9). |
| `lammps/production/` | Inputs of the production simulations (four-stage melt–quench protocol + 100 ps at 2500 K). |
| `lammps/controls/` | Inputs of the control simulations: amorphous core (4000 K and 2500 K only), α-rhombohedral core, GPU re-run of the FCC-type seed, bulk energies of amorphous / α-rhombohedral / FCC-type boron. |
| `analysis/` | Particle-centred analyses: O uptake (`o_uptake.py`), oxide-shell porosity/composition and H₂O location (`shell_analysis.py`), central-void tracking (`void_tracking.py`), and `reproduce_original_fig9.py`, which documents the error in the originally submitted Figure 9. |
| `figures/` | Scripts for Figure 10, Figures S6–S9 and Figure 11 (Blender renderings of the actual MD snapshots). |
| `legacy/` | The notebook used for the originally submitted Figure 9, kept for transparency (see `legacy/README.md`). |
| `forcefield/` | How to obtain the ReaxFF parameter file (identical to `potentials/ffield.reax.AB` of LAMMPS). |

## Software

* LAMMPS 29 Aug 2024 (6.8 nm; `reaxff/omp`, 24 OpenMP threads) and 27 Jun 2024 (10 and 15 nm, B-only; MPI + OpenMP),
  REAXFF package, `fix qeq/reaxff` (tolerance 10⁻⁶).
* LAMMPS 22 Jul 2025 with KOKKOS/CUDA (control simulations in `lammps/controls/`, RTX 5090).
* Python ≥ 3.10 with the packages in `requirements.txt`; Blender 3.5 for Figure 11.

## Reproducing the analyses

```bash
pip install -r requirements.txt
# download the Zenodo archives and extract them into one directory
for f in BoronCombustion_*.tar.zst; do tar --zstd -xf "$f" -C /path/to/data; done
BORON_DATA=/path/to/data ./run_analysis.sh          # results and figures are written to ./results
```

`run_analysis.sh` regenerates Figure 10, Figures S6–S9 and the values of Table S2. Expected O-uptake rates (atoms ps⁻¹,
Case 1 / Case 2): 6.8 nm 0.98 / 1.71, 10.0 nm 4.04 / 3.97, 15.0 nm 8.50 / 9.30; α-rhombohedral start (10 nm) 2.21 / 2.20.

Figure 11:
```bash
python figures/figure11/prep_snapshots.py
blender -b -P figures/figure11/render_blender.py -- results/figure11_snapshots results/figure11_renders 64
python figures/figure11/compose.py --micrographs <SEM/TEM image of the submitted Figure 10>
```

## Running the simulations

1. Build the structures (writes LAMMPS data files in the working directory):
   `python structures/generate_structures_multisize.py --size 10` (10 and 15 nm), `python structures/generate_structures.py` (6.8 nm).
   The shell placement is stochastic but seeded; the exact data files used in the paper are included in the Zenodo archive.
2. Copy the force-field file (see `forcefield/README.md`) and run, e.g.
   `mpirun -np 96 lmp -sf omp -pk omp 2 -in lammps/production/in.case1_10.0nm`.
3. Control simulations (GPU): `lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -in lammps/controls/<input>`.

Notes on file names in the Zenodo archive:
* The 6.8 nm inputs were run with the data-file names `case1_B_B2O3.data` / `case2_B_H3BO3_B2O3.data`; the archived files carry
  the size suffix (`case1_6.8nm_B_B2O3.data`, ...).
* The 10 nm Case 2 production run was restarted from `case2_10.0nm_equil.data` (`in.case2_10.0nm_restart`); the archived
  `case2_10.0nm_prod.lammpstrj` and `case2_10.0nm_species_prod.txt` are those of the complete restart run.
* The `*_bonds.txt` outputs of `fix reaxff/bonds` (~128 GB) are not archived; they were not used in the analyses.

## Notes on interpretation

* The FCC-type and α-rhombohedral starting cores are both far from equilibrium in this force field (+189 and
  +110 kcal mol⁻¹ atom⁻¹ relative to equilibrated amorphous boron) and lose their order within ~0.3 ps at 4000 K; the central
  cavity of the simulated particles forms at this stage, before oxygen enters the core (Figure 11g–j, Figure S9).
* Production temperatures (2500 K) exceed the melting point of bulk boron; the simulations are intended only as a qualitative
  comparison of Case 1 and Case 2 under identical conditions (single trajectory per system).

## License

Code: MIT (see `LICENSE`). Simulation data on Zenodo: CC BY 4.0.
