#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=2:system=polaris
#PBS -N hvd_tf_mnist

cd $PBS_O_WORKDIR

CONDAPATH=/home/parton/conda/2022-05-26/mconda3/
source $CONDAPATH/setup.sh

MNIST_URL=https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
if [ ! -f "mnist.npz" ]; then
   wget $MNIST_URL
fi

NODES=$(cat $PBS_NODEFILE | wc -l)
GPUS_PER_NODE=4
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  PPN=$GPUS_PER_NODE  RANKS=$RANKS

export OMP_NUM_THREADS=16

echo mpirun -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE  -- python horovod_mnist.py --horovod -i mnist.npz --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS
echo $PWD
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE  -- python ./horovod_mnist.py --horovod -i mnist.npz --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS 
