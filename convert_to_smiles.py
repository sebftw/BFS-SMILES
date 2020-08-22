import sys
from preprocess import string_to_smiles
import pandas as pd
import numpy as np
from tqdm import tqdm


def convert(smile):
    try:
        return string_to_smiles(smile)
    except Exception as e:
        # ValueError if MolToSmiles failed.
        # IndexError if a bond type was not in [0, 1, 2, 3].
        # OverflowError if there was given too many bonds, e.g. "CC11" or simply "C1".
        return "Fail with " + type(e).__name__

if __name__ == "__main__":
    fro = sys.argv[1]  # BFS-SMILES path
    to = sys.argv[2]   # SMILES result file

    bfs_smiles = pd.read_csv(fro)['SMILES'].values
    smiles = np.array([convert(smiles) for smiles in tqdm(bfs_smiles)])

    pd.DataFrame(data=smiles, columns=["SMILES"]).to_csv(to, index=False)