#!/bin/bash

#BSUB -J train_moses
#BSUB -o out_%J
#BSUB -e err_%J
#BSUB -q gpuv100
#BSUB -W 23:59  # Time in minutes, hh:mm.
#BSUB -R "span[hosts=1]"

## How many GPU's, how many CPU cores.
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -n 12

## How much memory per CPU slot (core)
#BSUB -R "rusage[mem=4GB]"

module load rdkit/2019_03_1-python-3.7.3

# GO
moses_path=../../moses

mkdir -p checkpoints/$MODEL

bfs_smiles_train_path=../dataset/BFS_SMILES1_train.csv.gz
bfs_smiles_test_path=../dataset/BFS_SMILES1_test.csv.gz
MODEL_NAME=$MODEL

python3 $moses_path/scripts/train.py $MODEL_NAME \
            --model_save checkpoints/$MODEL/$MODEL.pt \
            --config_save checkpoints/$MODEL/$MODEL\_config.pt \
            --vocab_save checkpoints/$MODEL/$MODEL\_vocab.pt \
            --device cuda:0 \
            --save_frequency 1 \
            --log_file checkpoints/$MODEL/$MODEL\_log.txt \
            --n_jobs 6 \
            --seed 0 &
P1=$!

MODEL=$MODEL\_bfs

python3 $moses_path/scripts/train.py $MODEL_NAME \
            --train_load $bfs_smiles_train_path \
            --val_load $bfs_smiles_test_path \
            --model_save checkpoints/$MODEL/$MODEL.pt \
            --config_save checkpoints/$MODEL/$MODEL\_config.pt \
            --vocab_save checkpoints/$MODEL/$MODEL\_vocab.pt \
            --device cuda:1 \
            --save_frequency 1 \
            --log_file checkpoints/$MODEL/$MODEL\_log.txt \
            --n_jobs 6 \
            --seed 0 &
P2=$!

wait $P1 $P2
