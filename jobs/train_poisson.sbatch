#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=poisson-train-radar-counter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="poisson-train-radar-counter-%j.out"
#SBATCH --err="poisson-train-radar-counter-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
module load cudnn/9.0v7.3.0.29
if [ $USER == "sk7898" ]; then
    source activate l3embedding-tf-12
else
    source activate l3embedding-new-cpu
fi

SRCDIR=.

python $SRCDIR/train_poisson_counter.py