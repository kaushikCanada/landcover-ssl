#!/bin/bash
GPUS_PER_NODE=1
NUMBER_OF_NODES=2
EPOCHS=2
BATCH_SIZE=256
#SBATCH --time=0-10:00:00
#SBATCH --account=def-sh1352
#SBATCH --mem=32000M            # memory per node
#SBATCH --nodes=$NUMBER_OF_NODES                # total number of nodes (N to be defined)
#SBATCH --gpus-per-node=$GPUS_PER_NODE       # # number of GPUs reserved per node (here 1)
#SBATCH --cpus-per-task=8      # CPU cores/threads
#SBATCH --output=landcover.out
#SBATCH --tasks-per-node=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

# Load needed python and cuda modules
module load python/3.11 cuda cudnn gdal libspatialindex

source ~/royenv/bin/activate
echo 'hello'
logdir=/home/karoy84/scratch/logs
datadir=/home/karoy84/scratch/data
export NCCL_BLOCKING_WAIT=1

tensorboard --logdir=${logdir}/lightning_logs --host 0.0.0.0 --load_fast false & \
    srun python ~/scratch/landcover-ssl/app/src/train/selfsupervised/train.py \
    --batch_size $BATCH_SIZE \
    --epoch $EPOCHS \
    --gpus_per_node $GPUS_PER_NODE \
    --number_of_nodes $NUMBER_OF_NODES \
    --num_workers 8 \
    --logdir ${logdir} \
    --data_dir  ${datadir}
