#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>

#define N 1000000
#define THREADS_PER_BLOCK 128

__global__ void add(int *, int *, int *); /* device function */

int main(int argc, char **argv){
  int i;
  int *a, *b, *c;
  int *dev_a, *dev_b, *dev_c;


  a = (int *) malloc(N*sizeof(int));
  b = (int *) malloc(N*sizeof(int));
  c = (int *) malloc(N*sizeof(int));

  for (i = 0; i < N; ++i){
    a[i] = - (i % 10);
    b[i] =    i % 11;
  }
  
  cudaMalloc( (void **) &dev_a, N*sizeof(int));
  cudaMalloc( (void **) &dev_b, N*sizeof(int));
  cudaMalloc( (void **) &dev_c, N*sizeof(int));

  cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

  /* launch N threads organized in blocks of size THREADS_PER_BLOCK */
  add<<<N/THREADS_PER_BLOCK + 1,THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_c);
  
  cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  //for (i=0;i<N;++i)
    //printf("c[%d]:%d\n", i, c[i]);

  free(a); free(b); free(c);
  cudaFree(dev_a); 
  cudaFree(dev_b); 
  cudaFree(dev_c); 

  
  exit(0);
}

__global__ void add(int *a, int *b, int *c){
  int myid;
  myid = threadIdx.x + blockIdx.x*blockDim.x;
  if (myid < N)
    c[myid] = a[myid] + b[myid];
  printf("here\n");
  return;
}
