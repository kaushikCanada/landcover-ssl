#!/bin/bash
#SBATCH --job-name=landcover     # create a short name for your job
#SBATCH --time=0-10:00:00
#SBATCH --account=def-sh1352
#SBATCH --mem=64000M            # memory per node
#SBATCH --nodes=1                # total number of nodes (N to be defined)
#SBATCH --gpus-per-node=1       # # number of GPUs reserved per node (here 1)
#SBATCH --cpus-per-task=20      # CPU cores/threads
#SBATCH --output=landcover.out
#SBATCH --tasks-per-node=1
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
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
python ~/scratch/landcover-ssl/app/src/train/selfsupervised/train.py \
    --batch_size 180 \
    --epoch 20 \
    --gpus_per_node 1 \
    --number_of_nodes 1 \
    --num_workers 20 \
    --logdir ${logdir} \
    --data_dir  ${datadir}
