#!/bin/bash
#SBATCH -p sched_mit_hill_cpsfr
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH --time=00:02:00
#
# Basic slurm job that runs in CPSFR 2023 nodes and checks GPU aware MPI is working.
#
#   To submit through Slurm use:
#
#   $ sbatch test_cuda_and_mpi.sbatch
#  
#   in terminal.


module load nvhpc/2023_233/nvhpc/23.3
culibdir=$NVHPC_ROOT/cuda/lib64
cuincdir=$NVHPC_ROOT/cuda/include

cat > test.c <<'EOFA'
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>
int main(int argc, char *argv[])
{
  int myrank, mpi_nranks; 
  int LBUF=1000000;
  float *sBuf_h, *rBuf_h;
  float *sBuf_d, *rBuf_d;
  int bSize;
  int i;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);                  // my MPI rank
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_nranks);

  if ( mpi_nranks != 2 ) { printf("Program requires exactly 2 ranks\n");exit(-1); }

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);               // How many GPUs?
  printf("Number of GPUs found = %d\n",deviceCount);
  int device_id = myrank % deviceCount;
  cudaSetDevice(device_id);                       // Map MPI-process to a GPU
  printf("Assigned GPU %d to MPI rank %d of %d.\n",device_id, myrank, mpi_nranks);

  // Allocate buffers on each host and device
  bSize = sizeof(float)*LBUF;
  sBuf_h = malloc(bSize);
  rBuf_h = malloc(bSize);
  for (i=0;i<LBUF;++i){
    sBuf_h[i]=(float)myrank;
    rBuf_h[i]=-1.;
  }
  if ( myrank == 0 ) {
   printf("rBuf_h[0] = %f\n",rBuf_h[0]);
  }

  cudaMalloc((void **)&sBuf_d,bSize);
  cudaMalloc((void **)&rBuf_d,bSize);

  cudaMemcpy(sBuf_d,sBuf_h,bSize,cudaMemcpyHostToDevice);

  if ( myrank == 0 ) {
   MPI_Recv(rBuf_d,LBUF,MPI_REAL,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  } 
  else if ( myrank == 1 ) {
   MPI_Send(sBuf_d,LBUF,MPI_REAL,0,0,MPI_COMM_WORLD);
  }
  else
  {
   printf("Unexpected myrank value %d\n",myrank);
   exit(-1);
  }

  cudaMemcpy(rBuf_h,rBuf_d,bSize,cudaMemcpyDeviceToHost);
  if ( myrank == 0 ) {
   printf("rBuf_h[0] = %f\n",rBuf_h[0]);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
EOFA

mpicc test.c -I${cuincdir} -L${culibdir} -lcudart

mpiexec --mca btl_openib_warn_no_device_params_found 0 -n 2 ./a.out 
