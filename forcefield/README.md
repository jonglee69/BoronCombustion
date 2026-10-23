# Force field

All simulations used the B/O/H ReaxFF parameter set of Weismiller et al. (J. Phys. Chem. A 2010, 114, 5485–5492).
The file used in this work (`ffield.reax.BOH`) is identical, apart from trailing whitespace, to `potentials/ffield.reax.AB`
distributed with LAMMPS (contributed by A. Thompson), so it is not redistributed here. To obtain it:

```bash
cp "$LAMMPS_DIR/potentials/ffield.reax.AB" ffield.reax.BOH
```

SHA-256 of the file as used in the simulations (`ffield.reax.BOH`, original line endings):
`7813cc55f15171b89a50f9f03b210b3e3a014fac794c157e3bcab1ecfdaa8d96`.
The LAMMPS input files refer to it as `ffield.reax.BOH` with the element mapping `H B O` (atom types 1, 2, 3).
