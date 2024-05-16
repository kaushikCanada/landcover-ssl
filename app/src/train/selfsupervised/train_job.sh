#!/bin/bash
GPUS_PER_NODE=1
NUMBER_OF_NODES=2
#SBATCH --time=0-10:00:00
#SBATCH --account=def-sh1352
#SBATCH --mem=32000M                    # memory per node
#SBATCH --nodes=$NUMBER_OF_NODES        # total number of nodes (N to be defined)
#SBATCH --gpus-per-node=$GPUS_PER_NODE  # number of GPUs reserved per node (here 1)
#SBATCH --cpus-per-task=8      # CPU cores/threads
#SBATCH --output=landcover.out
#SBATCH --tasks-per-node=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

# Load needed python and cuda modules
module load python/3.11 cuda cudnn gdal libspatialindex

source ~/royenv/bin/activate

# create the virtual environment on each node : 
# srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate

# pip install --no-index --upgrade pip
# pip install --no-index torchgeo tensorboard
# pip install lightly
# EOF

# Variables for readability
logdir=/home/karoy84/scratch/logs
datadir=/home/karoy84/scratch/data
# datadir=$SLURM_TMPDIR

export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

tensorboard --logdir=${logdir}/lightning_logs --host 0.0.0.0 --load_fast false & \
    srun python ~/scratch/landcover-ssl/app/src/train/selfsupervised/train.py \
    --batch_size 256 \
    --epoch 2 \
    --gpus_per_node $GPUS_PER_NODE \
    --number_of_nodes $NUMBER_OF_NODES \
    --num_workers 8 \
    --logdir ${logdir} \
    --data_dir  ${datadir}

