#!/bin/bash -l
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=64:ngpus=4
#PBS -N hvd_pt_mnist
#PBS -k doe
#PBS -j oe
#PBS -A datascience

cd $PBS_O_WORKDIR

module load conda/2022-07-19; conda activate

#MNIST_URL=https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
#if [ ! -f "mnist.npz" ]; then
#   wget $MNIST_URL
#fi

NODES=$(cat $PBS_NODEFILE | wc -l)
GPUS_PER_NODE=4
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  PPN=$GPUS_PER_NODE  RANKS=$RANKS

export OMP_NUM_THREADS=16

horovodrun --check-build

echo mpirun -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE  -- python horovod_mnist.py
echo $PWD
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE  -- python ./horovod_mnist.py

echo test mpi4py
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE python -c "import mpi4py.MPI as mpi;import socket;print('host: ',socket.gethostname(), 'rank: ',mpi.COMM_WORLD.Get_rank(),' size: ',mpi.COMM_WORLD.Get_size())"
