import os


def run_pml_file(pml_file):
    #automatically runs a pml file (the current environment has to have the module 'pymol' installed
    # might need a 'conda activate master_1' beforehand

    #os.system(f'/usr/bin/pymol -c {pml_file}')
    os.system(f'pymol -c {pml_file}')