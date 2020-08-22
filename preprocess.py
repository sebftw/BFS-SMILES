import copy
import os
import re

import numpy as np
from rdkit import Chem
import rdkit.Chem.Descriptors
from rdkit.Chem.rdchem import BondType

bond_types = (0, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE)  # 0 for NO_BOND.


def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    # This is taken from the GraphAF code.
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        print('converting radical electrons to H')
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m

def process_mol(mol):
    # Convert SMILES to the molecular form we want
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol)
    Chem.RemoveStereochemistry(mol)  # (- In case it is present)
    mol = Chem.RemoveHs(mol, sanitize=False)

    # From https://www.rdkit.org/docs/GettingStartedInPython.html:
    # "Note: as of this writing (Aug 2008), the smiles provided when one requests kekuleSmiles are not canonical.
    # The limitation is not in the SMILES generation, but in the kekulization itself."
    return mol

def reorder(mol):
    # Reverse Cuthill-McKee algorithm, with additional tie-breaking for molecular graphs.

    # adj_type = Chem.GetAdjacencyMatrix(mol, useBO=True)
    # csc = scipy.sparse.csc_matrix(adj_type)
    # order = np.ascontiguousarray(scipy.sparse.csgraph.reverse_cuthill_mckee(csc, symmetric_mode=True))  # [::-1]
    # return order

    canonical_order = list(Chem.CanonicalRankAtoms(mol))  # We need a tie-breaking rule.
    key = lambda atom: (atom.GetDegree(), canonical_order[atom.GetIdx()])  # Sign? mb. include atomicnumber or valency?

    x = min(mol.GetAtoms(), key=key)
    order = []
    queue = [x]
    R = {x.GetIdx()}
    while queue:
        x = queue.pop(0)
        for a in sorted(x.GetNeighbors(), key=key):
            if a.GetIdx() not in R:
                queue.append(a)
                R.add(a.GetIdx())
        order.append(x.GetIdx())

    return order

def smiles_to_string(smiles, full=True, separator=''):
    # Can also try full=True, separator=' ' for more explicit output.
    mol = process_mol(Chem.MolFromSmiles(smiles))

    # Extract the graph and reorder to minimize adjacency matrix bandwidth.
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    adj_type = Chem.GetAdjacencyMatrix(mol, useBO=True)

    order = reorder(mol)
    adj_type = adj_type[np.ix_(order, order)]
    atoms = [atoms[i] for i in order]

    # Done reordering. Ready to encode.
    code = ''
    for i in range(len(atoms)):
        code += atoms[i]
        nz = adj_type[i, :i].nonzero()[0]
        if len(nz):
            width = (i - nz).max()

            # We use int to convert to integer (requires kekulization!)
            addition = ''.join(str(int(x)) for x in np.flipud(adj_type[i, i - width:i]))
            if not full and (addition.endswith('01') or addition == '1'):
                # In this case shorten it, and let the final single-bond be implicit.
                addition = addition[:-1]
            code += addition

        if i != len(atoms) - 1:
            code += separator

    return code


def string_to_smiles(string):
    rw_mol = Chem.RWMol()

    # Assuming the string was created with full=True, we can simply split it using digits.
    # for atom_idx, atom_str in enumerate(string.split()):
    for atom_idx, atom_str in enumerate(re.findall('[A-Z][a-z]*\d*', string)):
        m = re.search(r"\d", atom_str)
        if m is None:
            digit_pos = len(atom_str)
        else:
            digit_pos = m.start()

        atom_name = atom_str[:digit_pos]
        rw_mol.AddAtom(Chem.Atom(atom_name))

        if atom_idx == 0:
            continue

        bonds = atom_str[digit_pos:]
        if bonds.endswith('0') or len(bonds) == 0:
            bonds += '1'  # Re-add implicit single bonds.

        for bond_pos, bond_type in enumerate(bonds, start=1):
            bond_type = bond_types[int(bond_type)]
            if bond_type:
                rw_mol.AddBond(atom_idx - bond_pos, atom_idx, bond_type)

    mol = rw_mol.GetMol()

    final_mol = convert_radical_electrons_to_hydrogens(mol)
    smile = None
    try:
        Chem.SanitizeMol(final_mol)
        smile = Chem.MolToSmiles(final_mol, isomericSmiles=False)
    except ValueError:  # The conversion failed. There was probably an issue with the BFS-SMILES string.
        pass

    return smile

if __name__ == "__main__":
    import moses
    import pandas as pd
    from tqdm import tqdm

    # Parameters
    moses_dataset = 'train'
    full = True

    smiles_input = moses.get_dataset(moses_dataset)
    string_output_file = 'BFS_SMILES' + str(int(full)) + '_' + moses_dataset + '.csv.gz'
    string_output_file = os.path.join('dataset', string_output_file)

    if not os.path.exists(string_output_file):
        #  Simply store the BFS-SMILES in the same manner as the moses SMILES datasets are stored.
        train_strings = np.array([smiles_to_string(smiles, full=full) for smiles in tqdm(smiles_input)])
        pd.DataFrame(data=train_strings, columns=["SMILES"]).to_csv(string_output_file, compression='gzip', index=False)
    else:
        # (Makefiles may be a better way, so we automatically ensure integrity.)
        print(string_output_file, 'already exists, so we will not do anything. (Delete it if it needs to be updated)')
