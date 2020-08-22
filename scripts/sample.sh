#!/bin/bash

#BSUB -J sample_moses
#BSUB -o out_%J
#BSUB -e err_%J
#BSUB -q hpc
#BSUB -W 9:00  # Time in minutes, hh:mm.
#BSUB -R "span[hosts=1]"

## How many CPU cores.
#BSUB -n 4

## How much memory per CPU slot (core)
#BSUB -R "rusage[mem=4GB]"

module load rdkit/2019_03_1-python-3.7.3

# GO
moses_path=../../moses

python3 $moses_path/scripts/sample.py ${MODEL%_bfs} \
            --model_load checkpoints/$MODEL/$MODEL.pt \
            --config_load checkpoints/$MODEL/$MODEL\_config.pt \
            --vocab_load checkpoints/$MODEL/$MODEL\_vocab.pt \
            --gen_save checkpoints/$MODEL/$MODEL\_sample.csv \
            --n_samples 10 \
            --device cpu \
            --max_len 200 \
            --seed 0

# Now if the model name contained the substring _bfs, assume we have to convert from BFS-SMILES to SMILES.
if [[ $MODEL == *"_bfs"* ]]; then
  mv checkpoints/$MODEL/$MODEL\_sample.csv checkpoints/$MODEL/$MODEL\_sample_BFS.csv
  python ../convert_to_smiles.py checkpoints/$MODEL/$MODEL\_sample_BFS.csv checkpoints/$MODEL/$MODEL\_sample.csv
fi

python3 $moses_path/scripts/eval.py \
            --gen_path checkpoints/$MODEL/$MODEL\_sample.csv \
            --n_jobs 4 > checkpoints/$MODEL/$MODEL\_metrics.csv
