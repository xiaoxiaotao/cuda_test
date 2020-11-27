#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#define random(a,b) (rand()%(b-a)+a)



int divup(int n, int m) 
{
    return n % m == 0 ? n/m : n/m + 1;
}




__global__ void mat_inv_kernel(
	const int k,
	const int n,
	const double *__restrict__ input,
	double *__restrict__ output)
{
	const int
		nInd = threadIdx.x + blockIdx.x * blockDim.x,
		kInd = threadIdx.y + blockIdx.y * blockDim.y;
	if (nInd < n && kInd < k)
	{
		const double a = input[nInd * k + kInd];
		output[nInd + kInd * n] = a;
	}
}

template <int BLOCK_DIM_X>
__global__ void
Compute_kernel_v4(
	const int k,
	const int m,
	const int n,
	const int result_size,
	const double *__restrict__ searchPoints,
	const double *__restrict__ referencePoints,
	int *__restrict__ result)
{
	const int ans_id = blockIdx.x * gridDim.y + blockIdx.y;
	if (ans_id >= result_size)
		return;
	__shared__ double dis_s[BLOCK_DIM_X];
	__shared__ int ind_s[BLOCK_DIM_X];
	dis_s[threadIdx.x] = INFINITY;
	for (int mInd = blockIdx.y, nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
		 nInd < n;
		 nInd += gridDim.x * BLOCK_DIM_X)
	{
		double squareSum = 0;
		for (int kInd = 0; kInd < k; ++kInd)
		{
			const double diff = searchPoints[kInd + mInd * k] - referencePoints[kInd * n + nInd];
			squareSum += diff * diff;
		}
		if (dis_s[threadIdx.x] > squareSum)
		{
			dis_s[threadIdx.x] = squareSum;
			ind_s[threadIdx.x] = nInd;
		}
	}
	__syncthreads();
	for (int offset = BLOCK_DIM_X >> 1; offset > 0; offset >>= 1)
	{
		if (threadIdx.x < offset)
			if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset])
			{
				dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
				ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
			}
		__syncthreads();
	}
	if (threadIdx.x == 0)
		result[ans_id] = ind_s[0];
}




int main(){

	int k=3;
	int m=5000;
	int n= 5000;
	// int *results;
	thrust::host_vector<int> result(m);
	double searchPoints[k*m];
	double referencePoints[k*n];

	for(int i=0;i<m;i++){

		searchPoints[i*k+0]=double(random(1,15));
		searchPoints[i*k+1]=double(random(1,15));
		searchPoints[i*k+2]=double(random(1,15));
	}
	for(int i=0;i<n;i++){
		referencePoints[i*k+0]=double(random(1,15));
		referencePoints[i*k+1]=double(random(1,15));
		referencePoints[i*k+2]=double(random(1,15));
	}
	
	auto start = std::chrono::steady_clock::now();
	thrust::device_vector<int> results_d(m);
	thrust::device_vector<double>
		s_d(searchPoints, searchPoints + k * m),
		r_d(referencePoints, referencePoints + k * n);
	
		
	
	auto start1 = std::chrono::steady_clock::now();
	const int BLOCK_DIM_X_v4 = 32, BLOCK_DIM_Y = 32;
	mat_inv_kernel<<<
		dim3(divup(n, BLOCK_DIM_X_v4), divup(k, BLOCK_DIM_Y)),
		dim3(BLOCK_DIM_X_v4, BLOCK_DIM_Y)>>>(
		k,
		n,
		thrust::raw_pointer_cast(s_d.data()),
		thrust::raw_pointer_cast(r_d.data()));
	
		const int BLOCK_DIM_X_v42 = 1024;
		Compute_kernel_v4<
		BLOCK_DIM_X_v42><<<
			dim3(results_d.size() / m, m),
			BLOCK_DIM_X_v42>>>(
			k,
			m,
			n,
			results_d.size(),
			thrust::raw_pointer_cast(s_d.data()),
			thrust::raw_pointer_cast(r_d.data()),
			thrust::raw_pointer_cast(results_d.data()));
	
	

	thrust::copy(
		results_d.begin(),
		results_d.end(),
		result.begin());

	auto end1 = std::chrono::steady_clock::now();
	std::chrono::duration<double, std::micro> elapsed1 = end1 - start1; // std::micro 表示以微秒为时间单位
	std::cout<< "time: "  << elapsed1.count()/1000. << "ms" << std::endl;




	// for(int i=0;i<m;i++){
	// 	std::cout<<result[i]<<" ";
	// }

	// std::cout<<std::endl;


	return 0;

}