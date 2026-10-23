"""Oxidizer-free control particle: the 6.8 nm Case 1 structure with all O atoms removed (B core + shell B atoms).
Reproduces case_B_only.data (17,760 B atoms) exactly.  usage: python make_B_only.py case1_6.8nm_B_B2O3.data case_B_only.data"""
import sys
src, dst = sys.argv[1], sys.argv[2]
L = open(src).read().split('\n'); i = [k for k, l in enumerate(L) if l.startswith('Atoms')][0]
atoms = [l for l in L[i + 2:] if len(l.split()) >= 6 and l.split()[1] == '2']
box = [l for l in L[:i] if l.rstrip().endswith(('xhi', 'yhi', 'zhi'))]
with open(dst, 'w') as f:
    f.write('LAMMPS data file - Pure B nanoparticle\n\n%d atoms\n3 atom types\n\n%s\n\nMasses\n\n1 1.008\n2 10.811\n3 15.999\n\nAtoms # charge\n\n'
            % (len(atoms), '\n'.join(box)))
    for j, l in enumerate(atoms):
        s = l.split(); f.write(f'{j + 1} 2 0.0 {s[3]} {s[4]} {s[5]}\n')
print(len(atoms), 'B atoms written to', dst)
