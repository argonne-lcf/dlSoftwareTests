#!/bin/bash -l
#COBALT -n 1
#COBALT -t 60
#COBALT -q full-node

echo loading module $1
module load $1
conda activate

module list

echo python = $(which python)

NODES=`cat $COBALT_NODEFILE | wc -l`
#NODES=1
GPUS_PER_NODE=8
#GPUS_PER_NODE=1
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  GPUS_PER_NODE=$GPUS_PER_NODE  RANKS=$RANKS

export OMP_NUM_THREADS=16

mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np ${RANKS} -npernode ${GPUS_PER_NODE} --hostfile ${COBALT_NODEFILE} python horovod_mnist.py -i /lus/theta-fs0/software/datascience/datasets/mnist.npz --horovod --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS
