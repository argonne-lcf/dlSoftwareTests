#!/bin/bash -l
#PBS -l walltime=00:30:00
#PBS -l select=8:ncpus=64:ngpus=4
#PBS -N hvd_tf_mnist
#PBS -k doe
#PBS -j oe
#PBS -A datascience


cd $PBS_O_WORKDIR

module load conda/2022-07-19; conda activate

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
