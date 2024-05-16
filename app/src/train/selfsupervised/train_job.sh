#!/bin/bash
#SBATCH --time=0-10:00:00
#SBATCH --account=def-sh1352
#SBATCH --mem=80000M            # memory per node
#SBATCH --nodes=5                # total number of nodes (N to be defined)
#SBATCH --gpus-per-node=1       # # number of GPUs reserved per node (here 1)
#SBATCH --cpus-per-task=10      # CPU cores/threads
#SBATCH --output=landcover.out
#SBATCH --tasks-per-node=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

# Load needed python and cuda modules
module load python/3.11 cuda cudnn gdal libspatialindex

# create the virtual environment on each node : 
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index torchgeo tensorboard
pip install lightly
EOF

# Variables for readability
logdir=/home/karoy84/scratch/logs
datadir=/home/karoy84/scratch/data
# datadir=$SLURM_TMPDIR

export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

tensorboard --logdir=${logdir}/lightning_logs --host 0.0.0.0 --load_fast false & \
    srun python /home/karoy84/scratch/app/src/train/selfsupervised/train.py \
    --batch_size 256 \
    --epoch 2 \
    --num_workers 10 \
    --logdir ${logdir} \
    --data_dir  ${datadir}

