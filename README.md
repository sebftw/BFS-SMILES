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

## Results

Results so far. The networks have not been fully trained yet. We let the BFS versions run for as many epochs as the vanilla versions. The networks were trained for 24 hours.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th rowspan="2">Model</th>
      <th rowspan="2">Valid (↑)</th>
      <th rowspan="2">Unique@1k (↑)</th>
      <th rowspan="2">Unique@10k (↑)</th>
      <th colspan="2">FCD (↓)</th>
      <th colspan="2">SNN (↑)</th>
      <th colspan="2">Frag (↑)</th>
      <th colspan="2">Scaf (↑)</th>
      <th rowspan="2">IntDiv (↑)</th>
      <th rowspan="2">IntDiv2 (↑)</th>
      <th rowspan="2">Filters (↑)</th>
      <th rowspan="2">Novelty (↑)</th>
    </tr>
    <tr>
      <th>Test</th>
      <th>TestSF</th>
      <th>Test</th>
      <th>TestSF</th>
      <th>Test</th>
      <th>TestSF</th>
      <th>Test</th>
      <th>TestSF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><i>Train</i></td>
      <td><i>1.0</i></td>
      <td><i>1.0</i></td>
      <td><i>1.0</i></td>
      <td><i>0.008</i></td>
      <td><i>0.4755</i></td>
      <td><i>0.6419</i></td>
      <td><i>0.5859</i></td>
      <td><i>1.0</i></td>
      <td><i>0.9986</i></td>
      <td><i>0.9907</i></td>
      <td><i>0.0</i></td>
      <td><i>0.8567</i></td>
      <td><i>0.8508</i></td>
      <td><i>1.0</i></td>
      <td><i>1.0</i></td>
    </tr>
    <tr>
      <td>aae</td>
      <td>0.9719</td>
      <td><b>1.0</b></td>
      <td>0.9965</td>
      <td>0.6428</td>
      <td>1.0706</td>
      <td>0.6185</td>
      <td>0.5756</td>
      <td>0.2426</td>
      <td>0.3409</td>
      <td>0.8837</td>
      <td>0.0845</td>
      <td>0.8551</td>
      <td>0.8484</td>
      <td><b>0.9966</b></td>
      <td>0.7768</td>
    </tr>
    <tr>
      <td>aae_bfs</td>
      <td>0.9775</td>
      <td><b>1.0</b></td>
      <td>0.9978</td>
      <td>0.5941</td>
      <td>1.3597</td>
      <td>0.6016</td>
      <td>0.5543</td>
      <td>0.9951</td>
      <td>0.9925</td>
      <td>0.878</td>
      <td>0.0838</td>
      <td><b>0.8638</b></td>
      <td><b>0.8574</b></td>
      <td>0.9849</td>
      <td>0.7836</td>
    </tr>
    <tr>
      <td>char_rnn</td>
      <td>0.9818</td>
      <td><b>1.0</b></td>
      <td>0.9992</td>
      <td>0.0977</td>
      <td>0.5643</td>
      <td>0.5987</td>
      <td>0.5632</td>
      <td><b>0.9997</b></td>
      <td><b>0.9984</b></td>
      <td>0.9322</td>
      <td><b>0.0981</b></td>
      <td>0.8562</td>
      <td>0.8502</td>
      <td>0.9945</td>
      <td><b>0.8577</b></td>
    </tr>
    <tr>
      <td>char_rnn_bfs</td>
      <td><b>0.9911</b></td>
      <td><b>1.0</b></td>
      <td><b>0.9994</b></td>
      <td><b>0.0732</b></td>
      <td><b>0.5319</b></td>
      <td>0.6062</td>
      <td>0.5657</td>
      <td>0.9996</td>
      <td>0.998</td>
      <td>0.9345</td>
      <td>0.0916</td>
      <td>0.8571</td>
      <td>0.8511</td>
      <td>0.9893</td>
      <td>0.7987</td>
    </tr>
    <tr>
      <td>vae</td>
      <td>0.9792</td>
      <td><b>1.0</b></td>
      <td>0.9982</td>
      <td>0.1206</td>
      <td>0.6109</td>
      <td><b>0.6243</b></td>
      <td><b>0.5774</b></td>
      <td>0.9994</td>
      <td>0.9978</td>
      <td>0.9322</td>
      <td>0.0647</td>
      <td>0.8569</td>
      <td>0.8509</td>
      <td><b>0.9966</b></td>
      <td>0.6978</td>
    </tr>
    <tr>
      <td>vae_bfs</td>
      <td>0.9775</td>
      <td><b>1.0</b></td>
      <td>0.9983</td>
      <td>0.1001</td>
      <td>0.5503</td>
      <td>0.6121</td>
      <td>0.5677</td>
      <td><b>0.9997</b></td>
      <td>0.9982</td>
      <td><b>0.9372</b></td>
      <td>0.0788</td>
      <td>0.8562</td>
      <td>0.8503</td>
      <td>0.9862</td>
      <td>0.7344</td>
    </tr>
  </tbody>
</table>
