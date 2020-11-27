#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>



//void cudaFindNearest(int numBlocks, int threadsPerBlock, double *P, double *Q, int nP, int nQ, double *Q_select, int *min_index_device);

void findNearest(int k,int m,int n,double *searchPoints,double *referencePoints,thrust::host_vector<int> *results);