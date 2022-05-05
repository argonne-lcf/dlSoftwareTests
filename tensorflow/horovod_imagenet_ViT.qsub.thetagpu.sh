#!/bin/bash -l
#COBALT -n 1 
#COBALT -t 480
#COBALT -q full-node
#COBALT -A datascience

echo loading $1
#module load conda/2021-09-22
module load $1
conda activate

module list

echo python = $(which python)

NODES=`cat $COBALT_NODEFILE | wc -l`
GPUS_PER_NODE=8
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  PPN=$PPN  RANKS=$RANKS

export OMP_NUM_THREADS=16

#This is the one that I think works:
#mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np $RANKS -npernode $GPUS_PER_NODE --hostfile $COBALT_NODEFILE python horovod_mnist_ViT.py -i /lus/theta-fs0/software/datascience/datasets/mnist.npz --horovod --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS

#Original - works on 1 node
mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np $((COBALT_JOBSIZE*8)) -npernode 8 --hostfile ${COBALT_NODEFILE} python horovod_imagenet_ViT.py -i /lus/theta-fs0/software/datascience/datasets/mnist.npz --horovod --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS

#mpirun -hostfile $COBALT_NODEFILE -n $RANKS -npernode $GPUS_PER_NODE -- python horovod_mnist_ViT.py -i /lus/theta-fs0/software/datascience/datasets/mnist.npz --horovod --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS
