import pdb
import sys
import re

def get_next_molecule(data, splits):                                                                         
    start = 0              
    for idx, split in enumerate(splits):
        end = start + split
        atoms = data[start:end]
        assert len(atoms) == split, "# of atoms in molecule {} is {}, expected = {}".format(idx+1, len(atoms), split)
        yield atoms 
        start = end


def main(filename):
    """
    Counts the number of molecules in xyz file
    asserts # of entries of Z, # of entries of molecules is same
    """
    with open(filename) as file:
        xyz_data = file.read()
    
    # rexp=r"""^(\d)+\n"""; match = re.findall(rexp, xyz_data, re.MULTILINE);data=[print(m) for m in match] # https://docs.python.org/2/library/re.html#re.M
    get_splits = lambda x : map(int, re.findall(r"""^(\d+)\n""", x, re.MULTILINE))
    splits = get_splits(xyz_data)
    
    get_atoms = lambda x: re.findall(r"\n([A-Z]) (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+)[\s\n]*$", x, re.M)
    data = get_atoms(xyz_data)
    
    # for idx,val in enumerate(get_next_molecule(data, splits)):
    #     print(idx,len(val),val)
    return splits, [_ for _ in get_next_molecule(data, splits)]


if __name__ == "__main__":
    filename = sys.argv[1]
    atoms_per_molecule, molecules = main(filename)
    print("entries = %d" % len(atoms_per_molecule))
    assert len(atoms_per_molecule) == len(molecules), """Number of 
        'atoms_per_molecule'({}) != Number of molecules ({}).
        must be equal""".format(len(atoms_per_molecule), len(molecules))
    pdb.set_trace()
