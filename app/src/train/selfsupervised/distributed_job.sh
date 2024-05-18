#!/bin/bash
#SBATCH --nodes=1             
#SBATCH --gpus-per-node=4          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=4   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8      # CPU cores/threads
#SBATCH --account=def-sh1352
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out

module load python/3.11 cuda cudnn gdal libspatialindex

echo "Hello World"
nvidia-smi


srun --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torchgeo tensorflow tensorboard --no-index
EOF

modeldir=/home/karoy84/scratch/output
datadir=/home/karoy84/scratch/data

export TORCH_NCCL_BLOCKING_WAIT=1
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

srun python ~/scratch/landcover-ssl/app/src/train/selfsupervised/ddp-train.py \
            --init_method tcp://$MASTER_ADDR:3456 \
            --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) \
            --batch_size 256 \
            --logdir ${logdir} \
            --data_dir  ${datadir}
