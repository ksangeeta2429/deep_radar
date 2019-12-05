#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=time-hidden-gridsearch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="time-hidden-gridsearch-%j.out"
#SBATCH --err="time-hidden-gridsearch-%j.err"

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
#HIDDEN_1=$1
HIDDEN_2=$1
WINDOW=256
BATCH_SIZE=64
DATA_DIR=$SCRATCH/pedbike/window_256
OUTPUT_DIR=$SCRATCH/radar_counting/models
DATA_TYPE='time'
MODEL_TYPE='lstm_time'
CV=3

hidden1=( 32, 64 )

nvidia-cuda-mps-control -d
echo "start_server -uid `id -u`" | nvidia-cuda-mps-control

for h1 in ${hidden1[@]}
do
    python $SRCDIR/train_baseline_counter.py \
	    -base $DATA_DIR \
	    -datatype $DATA_TYPE \
	    -cv $CV \
	    -mt $MODEL_TYPE \
	    -window $WINDOW \
	    -outdir $OUTPUT_DIR \
	    -tbs $BATCH_SIZE \
	    -h1 $h1 \
	    -h2 $HIDDEN_2
done

echo "Waiting for processes to finish..."
wait
echo "quit" | nvidia-cuda-mps-control
