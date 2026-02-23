#include <cstdio>
#include <cuda_runtime.h>

__global__ void k(int *out) { out[0] = 123; }

int main() {
  int *d=nullptr;
  int h=-1;

  cudaError_t e = cudaMalloc(&d, sizeof(int));
  if (e != cudaSuccess) { printf("cudaMalloc failed: %s\n", cudaGetErrorString(e)); return 1; }

  k<<<1,1>>>(d);
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("kernel failed: %s\n", cudaGetErrorString(e)); return 2; }

  e = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) { printf("cudaMemcpy failed: %s\n", cudaGetErrorString(e)); return 3; }

  printf("OK, got %d\n", h);
  cudaFree(d);
  return (h==123) ? 0 : 4;
}