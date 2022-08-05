#!/bin/bash -l
#COBALT -n 2
#COBALT -t 60
#COBALT -q full-node
#COBALT -A datascience

module load conda/2022-07-01; conda activate

#MNIST_URL=https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
#if [ ! -f "mnist.npz" ]; then
#   wget $MNIST_URL
#fi

NODES=$(cat $COBALT_NODEFILE | wc -l)
GPUS_PER_NODE=8
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  GPUS_PER_NODE=$GPUS_PER_NODE  RANKS=$RANKS

export OMP_NUM_THREADS=16


echo mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np ${RANKS} -npernode ${GPUS_PER_NODE} --hostfile $COBALT_NODEFILE -- python ./horovod_mnist.py
echo $PWD
mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np ${RANKS} -npernode ${GPUS_PER_NODE} --hostfile $COBALT_NODEFILE -- python ./horovod_mnist.py

echo test mpi4py
mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np ${RANKS} -npernode ${GPUS_PER_NODE} --hostfile $COBALT_NODEFILE -- python -c "import mpi4py.MPI as mpi;import socket;print('host: ',socket.gethostname(), 'rank: ',mpi.COMM_WORLD.Get_rank(),' size: ',mpi.COMM_WORLD.Get_size())"
