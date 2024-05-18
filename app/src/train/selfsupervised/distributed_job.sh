#!/bin/bash
#SBATCH --nodes=2             
#SBATCH --gpus-per-node=4          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=4   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10      # CPU cores/threads
#SBATCH --account=def-sh1352
#SBATCH --time=0-05:00
#SBATCH --output=%N-%j.out

module load python/3.11 cuda cudnn gdal libspatialindex

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

# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

# srun python ~/scratch/landcover-ssl/app/src/train/selfsupervised/ddp-train.py \
#             --init_method tcp://$MASTER_ADDR:3456 \
#             --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) \
#             --batch_size 64 \
#             --start_epoch 0 \
#             --max_epochs 2 \
#             --num_workers 8 \
#             --limit 5 \
#             --log_dir ${log_dir} \
#             --data_dir  ${data_dir}
            
srun python ~/scratch/landcover-ssl/app/src/train/selfsupervised/barlow_twins.py \
            --batch_size 512 \
            --epochs 2 \
            --workers 10 \
            --checkpoint_dir ${log_dir} \
            --data_dir  ${data_dir}
