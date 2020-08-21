#!/bin/bash

bsub -env "all, MODEL=char_rnn" < train_bfs.sh