#!/usr/bin/env bash
#COBALT -n 2
#COBALT -t 60
#COBALT -q debug-flat-quad

module load $1

export OMP_NUM_THREADS=64

aprun -n 2 -N 1 --cc none python horovod_mnist.py -i /lus/theta-fs0/software/datascience/datasets/mnist.npz --horovod --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS
