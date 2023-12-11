#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

__global__ void compute_cuda(vector3** accels, double *hPos, double *hVel, double *mass){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int k;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    
	if (i < NUMENTITIES && j < NUMENTITIES) {
		//first compute the pairwise accelerations.  Effect is on the first argument.
		if (i==j) {
			FILL_VECTOR(accels[i][j],0,0,0);
		}
		else {
			vector3 distance;
			for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
			double magnitude=sqrt(magnitude_sq);
			double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
			FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
		}
	}
}

extern "C" void compute() {
	vector3* values;
	vector3** accels;
	size_t valuesSize = sizeof(vector3) * NUMENTITIES * NUMENTITIES;
	size_t accelsSize = sizeof(vector3*) * NUMENTITIES * NUMENTITIES;
	cudaMallocManaged(&values, valuesSize);
    cudaMallocManaged(&accels, accelsSize);
	for (int i=0;i<NUMENTITIES;i++) {
		accels[i]=&values[i*NUMENTITIES];
	}

	
	double *hPosDevice, *hVelDevice, *massDevice;
    size_t pvSize = sizeof(double) * NUMENTITIES * 3;
    size_t massSize = sizeof(double) * NUMENTITIES;
    cudaMalloc((void **)&hPosDevice, pvSize);
    cudaMalloc((void **)&hVelDevice, pvSize);
    cudaMalloc((void **)&massDevice, massSize);

	// Data -> device
    cudaMemcpy(hPosDevice, hPos, pvSize, cudaMemcpyHostToDevice);
    cudaMemcpy(hVelDevice, hVel, pvSize, cudaMemcpyHostToDevice);
    cudaMemcpy(massDevice, mass, massSize, cudaMemcpyHostToDevice);

	// execute the cuda call
	dim3 threadPerBlock(4, 4);
    dim3 numBlocks((NUMENTITIES/threadPerBlock.x) + 1, (NUMENTITIES/threadPerBlock.y) + 1);
	compute_cuda<<<numBlocks,threadPerBlock>>>(accels, hPosDevice, hVelDevice, massDevice);

	// Synchronize CUDA call
	// Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

	// Updated data -> host
    // cudaMemcpy(hPos, hPosDevice, pvSize, cudaMemcpyDeviceToHost);
    // cudaMemcpy(hVel, hVelDevice, pvSize, cudaMemcpyDeviceToHost);

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (int i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (int j=0;j<NUMENTITIES;j++){
			for (int k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (int k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}

    cudaFree(hPosDevice);
    cudaFree(hVelDevice);
    cudaFree(massDevice);
    cudaFree(values);
	cudaFree(accels);
}
