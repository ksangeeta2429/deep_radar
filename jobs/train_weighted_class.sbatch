#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=weighted-class
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="train-weighted-class-%j.out"
#SBATCH --err="train-weighted-class-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
module load cudnn/10.0v7.6.2.24
source activate l3embedding-tf-14-gpu

SRCDIR=.

python $SRCDIR/train_weighted_classification.py