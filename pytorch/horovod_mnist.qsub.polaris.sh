#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=2:system=polaris
#PBS -N hvd_pt_mnist
#PBS -k doe

cd $PBS_O_WORKDIR

#CONDAPATH=/home/parton/conda/2022-05-26/mconda3/
#source $CONDAPATH/setup.sh
module load conda/2022-05-26

#MNIST_URL=https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
#if [ ! -f "mnist.npz" ]; then
#   wget $MNIST_URL
#fi

NODES=$(cat $PBS_NODEFILE | wc -l)
GPUS_PER_NODE=4
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  PPN=$GPUS_PER_NODE  RANKS=$RANKS

export OMP_NUM_THREADS=16

echo mpirun -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE  -- python horovod_mnist.py
echo $PWD
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE  -- python ./horovod_mnist.py 

echo test mpi4py
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE python -c "import mpi4py.MPI as mpi;import socket;print('host: ',socket.gethostname(), 'rank: ',mpi.COMM_WORLD.Get_rank(),' size: ',mpi.COMM_WORLD.Get_size())"
