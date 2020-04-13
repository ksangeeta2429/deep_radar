#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=baseline-cls
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7:00:00
#SBATCH --mail-type=ALL
#SBATCH --output="baseline-cls-%j.out"
#SBATCH --err="baseline-cls-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

TYPE=$1

module purge
module load cudnn/10.1v7.6.5.32
source activate l3embedding-tf-2-gpu

SRCDIR=.

if [ $TYPE == "time" ]; then
    echo "TIME"
    python $SRCDIR/get_time_baseline_res.py
else
    echo "FREQUENCY"
    python $SRCDIR/get_freq_baseline_res.py
fi