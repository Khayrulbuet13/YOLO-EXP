GPUS=$1
export OMP_NUM_THREADS=8
torchrun --nproc_per_node=$GPUS main.py ${@:2}