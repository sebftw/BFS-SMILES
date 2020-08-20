# BFS-SMILES
An alternative notation for describing molecules. Whereas SMILES is [DFS-based](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system#Graph-based_definition), this is BFS-based.

This idea was inspired by [GraphAF](https://arxiv.org/abs/2001.09382), (which in turn is inspired by [MolecularRNN](https://arxiv.org/abs/1905.13372)). Another alternative approach to a molecular graph language for deep learning is called [DeepSMILES](https://depth-first.com/articles/2019/03/19/chemical-line-notations-for-deep-learning-deepsmiles-and-beyond/).

## Examples

These examples were produced with `full=False`, meaning the final bond, if it is a single-bond, is left implicit.
| SMILES | BFS-SMILES |
| :------------- |:-------------|
| CCCCCC | CCCCCC |
| C=C=C=C=C=C | CC2C2C2C2C2 |
| c1ccccc1     | CCC02C02C0C21      |
| CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1 | CCCSO2C0C2C0C0C02C12N0N0C11N2CO2O0C      |

In case `full=True` the first example would read CC1C1C1C1C1; this means every C-atom is bonded with a single-bond '1', to the previous atom.
The character '0' is no-bond, it means to skip over the previous atom and go to the next one back.
