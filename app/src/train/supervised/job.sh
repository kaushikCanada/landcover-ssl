#!/bin/bash
#SBATCH --nodes=1             
#SBATCH --gpus-per-node=1          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=1   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10      # CPU cores/threads
#SBATCH --account=def-sh1352
#SBATCH --time=0-10:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=kaushik.roy@inrs.ca
#SBATCH --output=%N-%j.out

module load python/3.11 libspatialindex

echo "Hello World"
nvidia-smi

source ~/royenv/bin/activate

log_dir=/home/karoy84/scratch/output
data_dir=/home/karoy84/scratch/data

export TORCH_NCCL_BLOCKING_WAIT=1
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

# srun python ~/scratch/landcover-ssl/app/src/train/supervised/finetune.py \
#             --batch_size 32 \
#             --epochs 2 \
#             --workers 10 \
#             --checkpoint_dir ${log_dir} \
#             --data_dir  ${data_dir}

srun python ~/scratch/landcover-ssl/app/src/train/supervised/linear2D_probe.py \
            --batch_size 16 \
            --lr 0.001 \
            --max_epochs 5 \
            --num_workers 10 \
            --checkpoint_dir ${log_dir} \
            --data_dir  ${data_dir}
