#!/bin/bash

# bsub -env "all, MODEL=char_rnn" < train_bfs.sh
# bsub -env "all, MODEL=char_rnn" < train.sh

# bsub -env "all, MODEL=vae" < train_bfs.sh
# bsub -env "all, MODEL=vae" < train.sh

# bsub -env "all, MODEL=aae" < train_bfs.sh
# bsub -env "all, MODEL=aae" < train.sh

bsub -env "all, MODEL=char_rnn" < train_both.sh
bsub -env "all, MODEL=vae" < train_both.sh
bsub -env "all, MODEL=aae" < train_both.sh
